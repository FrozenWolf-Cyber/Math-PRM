from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer
import torch.nn.functional as F
import torch
import torch.nn as nn
import gc
from peft import LoraConfig, TaskType, get_peft_model
DEBUG = True
# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,             # Rank for dimensionality reduction (higher = better performance but more compute)
#     lora_alpha=16,   # Scaling factor for LoRA weights
#     target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to (GPT example)
#     lora_dropout=0.1,  # Dropout probability for LoRA layers
#     bias="none"      # Whether to apply LoRA to biases ("none", "all", or "lora_only")
# )


def get_best_dtype():
    capability = torch.cuda.get_device_capability()
    # Ampere and above (sm_80, 8.0) supports bfloat16
    supports_bfloat16 = capability >= (8, 0)
    return torch.bfloat16 if supports_bfloat16 else torch.float32


class QwenMathTokenClf_RM(nn.Module):
    def __init__(self, device, args):
        super(QwenMathTokenClf_RM, self).__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(
    args.reward_model, 
    device_map=device, 
    torch_dtype=get_best_dtype(),
    trust_remote_code=True,
)
        self.base_model.score = nn.Identity()
        self.args = args
        self.LN = nn.Linear(self.base_model.config.hidden_size, 2, device=device, dtype=get_best_dtype())
        self.model_path = args.reward_model
        if args.add_new_token:
            self.add_token()
        # model.config.num_labels = 2
        # self.base_model.score = nn.Linear(1536, 2, device=device, dtype=torch.bfloat16)

    def add_token(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        new_tokens = ['<PRM_STEP_SCORE>']
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        if num_added_tokens > 0:
            self.base_model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added_tokens} new tokens to the tokenizer and model. The token is {new_tokens[0]}.")
        else:
            print("No new tokens were added.")
            
        return tokenizer

    def forward(self, input_ids, attention_mask, labels=None, no_grad=False):
        if not no_grad:
            no_grad = self.args.freeze_till_last and (not self.args.add_new_token)
        
        with torch.set_grad_enabled(not no_grad):
            global DEBUG
            if DEBUG:
                print("Using no gradient mode:", no_grad)
                DEBUG = False
                
            if labels is None:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        logits = self.LN(logits).to(dtype=torch.float)  # Apply linear layer to logits
        # print(outputs)
        logits = F.softmax(logits, dim=-1)[..., 1]  # Assuming the second class is the one we want to predict
        # print(value_outputs)
        return logits.squeeze(dim=1)
        
        
        
class QwenMathCondGen_RM(nn.Module):
    def __init__(self, device, args):
        super(QwenMathCondGen_RM, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
    args.reward_model, 
    device_map=device, 
    torch_dtype=get_best_dtype(),
    trust_remote_code=True,
)
        self.model_path = args.reward_model
        self.args = args
        # self.lora_model = get_peft_model(base_model, lora_config)
        ### get dtype and set to linear layer
        dtype = self.base_model.dtype
        print(f"Model dtype: {dtype}")
        if args.add_new_token:
            self.add_token()
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1, device=device, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
            
        
    def add_token(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        new_tokens = ['<PRM_STEP_SCORE>']
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        if num_added_tokens > 0:
            self.base_model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added_tokens} new tokens to the tokenizer and model. The token is {new_tokens[0]}.")
        else:
            print("No new tokens were added.")
            
        return tokenizer

    def forward(self, input_ids, attention_mask, labels=None, no_grad=False):
        if not no_grad:
            no_grad = self.args.freeze_till_last and (not self.args.add_new_token)
        
        with torch.set_grad_enabled(not no_grad):
            global DEBUG
            if DEBUG:
                print("Using no gradient mode:", no_grad)
                DEBUG = False
   
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            
        outputs = outputs.logits[:, -1, :]
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)
    
def configure_module(args, device):
    if args.model_type == "token":
        model = QwenMathTokenClf_RM(device, args)
    else:
        model = QwenMathCondGen_RM(device, args)
        
    if args.peft_rank != -1:
            peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS if args.model_type == "token" else TaskType.CAUSAL_LM,
            r=args.peft_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        )
            model.base_model = get_peft_model(model.base_model, peft_config)
            print("Using PEFT model with LoRA configuration:\n","----"*10)
            print(peft_config)
            print("Trainable parameters in PEFT model:", model.base_model.print_trainable_parameters())
            print("LN parameters:", sum(p.numel() for p in model.LN.parameters() if p.requires_grad))
            
    if args.freeze_till_last:
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.LN.requires_grad = True
        if args.add_new_token:
            print("Unfreezing newly addded token")
            model.base_model.model.embed_tokens.weight[-1].requires_grad = True
            
        
    if args.freeze_tokens:
        print("Freezing all embeddings")
        model.base_model.model.embed_tokens.requires_grad = False
        if args.add_new_token:
            print("Freezing all embeddings except the newly added token")
            model.base_model.model.embed_tokens.weight[-1].requires_grad = True
            
        
        
    model.to(device)
    if args.freeze_all_but_bias:
        for param in model.parameters():
            param.requires_grad = False
            
        ### freeeze all parameters except last 1 bias paramater:
        bias_params = [p for n, p in model.named_parameters() if 'bias' in n]         
        last_bias = bias_params[-1]
        last_bias.requires_grad = True
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")
                
    ### Trainable parameters using numel out of total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("-----"*10)
    print(f"Trainable parameters: {trainable_params} out of {total_params} total parameters.")
    print("-----"*10)
                
    return model
    
    
def unbatch_process(batch, device, model, max_step_size, no_grad=False, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(batch['input_ids'])
    
    score_list = []
    for idx in range(start, end, max_step_size):
        print(f"Processing batch from {idx} to {min(end, idx + max_step_size)}")
        input_ids = batch['input_ids'][idx:min(end, idx + max_step_size)].to(device)
        attn_mask = batch['attention_mask'][idx:min(end,idx + max_step_size)].to(device)
        score = model(input_ids.to(device),
                        attn_mask.to(device), no_grad=no_grad)
        score_list.append(score)
        del input_ids, attn_mask
        gc.collect()
        torch.cuda.empty_cache()
    score = torch.cat(score_list, dim=0)
    return score

from collections import defaultdict

def round_robin_batch_ordering(batch):
    batch["index"] = torch.tensor(batch["index"])
    unique_ids = batch["index"].unique(sorted=True)

    # Step 1: Group indices by group_id
    group_to_indices = defaultdict(list)
    for idx, gid in enumerate(batch["index"].tolist()):
        group_to_indices[gid].append(idx)

    print(group_to_indices)
    # Step 2: Interleave in round-robin
    interleaved_indices = []
    while any(group_to_indices.values()):
        for gid in unique_ids.tolist():
            if group_to_indices[gid]:
                interleaved_indices.append(group_to_indices[gid].pop(0))

    # Step 3: Reindex all tensors
    print("Interleaved indices:", interleaved_indices, batch.keys())
    interleaved_indices = torch.tensor(interleaved_indices)

    for key in ['input_ids', 'attention_mask', 'index']:
        batch[key] = batch[key][interleaved_indices]
        
    
    reverse_indices = torch.empty_like(interleaved_indices)
    reverse_indices[interleaved_indices] = torch.arange(len(interleaved_indices))

        
    return reverse_indices


class DomainTable(nn.Module):
    def __init__(self, domain_to_idx):
        """
        Args:
            domain_to_idx (dict):
                Mapping from domain strings to integer indices, e.g., {"domain_a": 0, "domain_b": 1}.
        """
        super(DomainTable, self).__init__()
        self.domain_to_idx = domain_to_idx
        self.num_domains = len(domain_to_idx)

        # Create learnable raw weights (initialized to zero)
        self.raw_weights = nn.Parameter(torch.ones(self.num_domains))

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                Domain names for each sample in the batch. Length should match x's batch_size.
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1), containing a single value per sample.

        Returns:
            torch.Tensor:
                Output tensor of same shape (batch_size, 1), where each element is the original input
                multiplied by its corresponding domain weight.
        """
        # Apply softplus to ensure weights are positive
        positive_weights = torch.nn.functional.softplus(self.raw_weights)

        # Normalize weights by their mean to maintain scale
        mean_weights = positive_weights.mean()
        normalized_weights = positive_weights / mean_weights

        # Convert domain strings to indices matching batch order
        # idxes = [self.domain_to_idx[d] for d in domain_strings]
        idxes = domain_strings
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]
        # Retrieve domain weights for each sample in the batch [batch_size]
        domain_weights = normalized_weights[idxes]
        # Reshape weights to match input tensor dimensions [batch_size, 1]
        domain_weights = domain_weights.view(-1, 1)
        # Element-wise multiplication: each input value multiplied by its domain weight
        out = x * domain_weights
        return out
    
    
def get_pred(args, batch, score):
    score_cpy = score.clone()
    correctness = batch['correctness'] ## (B, )
    if args.model_type == "token":
        labels = batch['label'].to(dtype=torch.float).to("cpu")
        # score -> (B, T,)
        if args.dreamprm_loss:
            mask = (labels != -100)
            score[mask] = 0
            score = score / (1 - score)
            score[mask] = 1
            score = torch.log(score)
            score = score * mask.float()  # (B, T)
            mean_score = torch.mean(score, dim=1)
            outputs = torch.sigmoid(mean_score) ## (B, )
            
        else:
            # select last non -100 label
            non_filler = (labels != -100).float()
            reversed_non_filler = non_filler.flip(dims=[1])  # (B, T)
            # find index of the last non-(-100) (which is first True in the reversed tensor)
            reversed_index = torch.argmax(reversed_non_filler.float(), dim=1)  # (B,)
            # convert to correct index from the start
            index = labels.size(1) - 1 - reversed_index  # (B,)
            outputs = score[torch.arange(labels.size(0)),index]  # (B, )
        
        # extract score
        gt = labels[labels != -100].int().tolist()
        step_pred = score_cpy[labels != -100].tolist()

        
 
    else:
        # score -> (B * T*(T+1)/2) So [A_0_1, A_0_2,.. B_0_1, B_0_2,...] in cummulative order
        nproblems = set(batch['index']) # [0, 1, 2, B_Size-1]
        score_temp = score.clone()
        gt = []
        step_pred = []
        score = torch.log(score / (1 - score))
        outputs = []
        for i in nproblems:
            mean_score = 0
            step = 0
            for j in range(len(score)):
                if batch['index'][j] == i:
                    mean_score += score[j]
                    step_pred.append(score_temp[j].item())
                    gt.append(int(batch['label'][j]))
                    step += 1
            mean_score /= step
            outputs.append(mean_score)
        outputs = torch.stack(outputs) # (B, )
        outputs = torch.sigmoid(outputs)
    
    problem_pred = outputs.clone().tolist()
    ### round of predictions to 0 or 1
    step_pred = [1 if x >= 0.5 else 0 for x in step_pred]
    problem_pred = [1 if x >= 0.5 else 0 for x in problem_pred]
    print("Step pred:", step_pred)
    print("GT:", gt)
    print("problem_pred:", problem_pred)
    print("Correctness:", correctness)
    
    ## find metric for step_pred vs gt and outputs vs correctness
    ### Calculate TP, FP, TN, FN, ACC, F1, Precision, Recall

    return outputs, {'step_pred': step_pred,
            'gt': gt,
            'problem_pred': problem_pred,
            'correctness': correctness,}
        
import numpy as np

def binary_classification_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    
    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    accuracy = (TP + TN) / len(labels)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
