from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer
import torch.nn.functional as F
import torch
import torch.nn as nn

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
    return torch.bfloat16 if supports_bfloat16 else torch.float16


class QwenMathTokenClf_RM(nn.Module):
    def __init__(self, device, model_path = "Qwen/Qwen2.5-Math-7B-Instruct"):
        super(QwenMathTokenClf_RM, self).__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(
    model_path, 
    device_map=device, 
    torch_dtype=get_best_dtype()
,
    trust_remote_code=True,
)
        self.model_path = model_path
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

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is None:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits.to(dtype=torch.float)
        # print(outputs)
        logits = F.softmax(logits)[..., 1]  # Assuming the second class is the one we want to predict
        # print(value_outputs)
        return logits.squeeze(dim=1)
        
        
        
class QwenMathCondGen_RM(nn.Module):
    def __init__(self, device, model_path="Qwen/Qwen2.5-Math-7B-Instruct"):
        super(QwenMathCondGen_RM, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=device, 
    torch_dtype=get_best_dtype(),
    trust_remote_code=True,
)
        self.model_path = model_path
        # self.lora_model = get_peft_model(base_model, lora_config)
        ### get dtype and set to linear layer
        dtype = self.base_model.dtype
        print(f"Model dtype: {dtype}")
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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.logits[:, -1, :]
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)

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
        self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))

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