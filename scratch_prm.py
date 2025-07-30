# All code is original unless otherwise noted.

import argparse
import torch.optim as optim
from model import *
from prm_data import *
from utils import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
import wandb
from torch.optim import AdamW
import numpy as np
from copy import deepcopy
import pandas as pd
from eval_aime import *
import gc
from model import round_robin_batch_ordering
from peft import PeftModel
 
parser = argparse.ArgumentParser(description="DreamPRM")
parser.add_argument('--weights_path', type=str)
parser.add_argument("--iteration_num", type=int, default=10000)
parser.add_argument("--save_every_iterations", type=int, default=5000)
parser.add_argument("--unroll_steps", type=int, default=5)
parser.add_argument("--gradiant_accumulation", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--scheduler_step_size", type=int, default=5000)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_lr", type=float, default=0.01)
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--meta_batch_size", type=int, default=1)


parser.add_argument("--dreamprm_loss", action="store_true")
parser.add_argument("--model_type", type=str, default="token")
parser.add_argument("--meta_dataset", type=str, default="AIME", help="AIME or PRM800K or both")
parser.add_argument("--sanity_check", action="store_true")
parser.add_argument("--overfit", type=int, default=-1)
parser.add_argument("--add_new_token",  action="store_true", help="Whether to add new token <PRM_STEP_SCORE> to the tokenizer")
parser.add_argument("--freeze_till_last",  action="store_true", help="Freeze till last layer")
parser.add_argument("--freeze_tokens",  action="store_true", help="Freeze other than newly added tokens")
parser.add_argument("--max_step_size", type=int, default=-1)
parser.add_argument("--max_meta_steps_grad", type=int, default=-1)
parser.add_argument("--filter_dataset_steps", type=int, default=20, help="Max number of steps to filter dataset")
parser.add_argument("--filter_dataset_token_size", type=int, default=2000, help="Max tokens to filter dataset")
parser.add_argument("--wandb_mode", type=str, default="online", help="wandb mode")
parser.add_argument("--notes", type=str, default="", help="wandb notes")
parser.add_argument("--freeze_all_but_bias", action="store_true")
parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping value")
parser.add_argument("--peft_rank", type=int, default=-1, help="Rank for PEFT, -1 for no PEFT")
parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA")
parser.add_argument("--load_path", type=str, default="", help="Path to load the model from")
parser.add_argument("--evaluate_only", action="store_true")
parser.add_argument("--balance", action="store_true")


args = parser.parse_args()
print(args)
set_seed(args.seed)
domain_list = {'Algebra':0,
 'Counting & Probability':1,
 'Geometry':2,
 'Intermediate Algebra':3,
 'Number Theory':4,
 'Prealgebra':5,
 'Precalculus':6}

def get_best_dtype():
    capability = torch.cuda.get_device_capability()
    # Ampere and above (sm_80, 8.0) supports bfloat16
    supports_bfloat16 = capability >= (8, 0)
    return torch.bfloat16 if supports_bfloat16 else torch.float32

if get_best_dtype() == torch.bfloat16:
    print("Using bfloat16 precision")
else:
    args.precision = "fp32"
    print("Using fp32 precision")

inv_domain_list = {v: k for k, v in domain_list.items()}

print(domain_list)
sanity_check = args.sanity_check
if sanity_check:
    print("============ SANITY CHECK MODE ============")

if not os.path.exists(args.weights_path):
    os.makedirs(args.weights_path)

if sanity_check:
    args.save_every_iterations = 100
    args.iteration_num = 200
    if args.wandb_mode == "online":
        args.wandb_mode = "offline"
      
inner_log_every = 100  
if args.overfit != -1:
    inner_log_every = 1


tokenizer = AutoTokenizer.from_pretrained(args.reward_model, trust_remote_code=True)
if args.add_new_token:
    new_tokens = ['<PRM_STEP_SCORE>']
    num_added_tokens = tokenizer.add_tokens(new_tokens)


sampler = None
resume_idxes = None
resume_labels = None




(
    train_dataloader,
    meta_dataloader,
    dataloader_benchmark,
    validation_dataloader
) = build_dataloader(
    tokenizer=tokenizer,
    train_batch_size= args.train_batch_size,
    meta_batch_size= args.meta_batch_size,
    inf_batch_size=1,
    token_based=args.model_type == "token",
    add_new_token=args.add_new_token,
    meta_dataset=args.meta_dataset,
    sanity_check=sanity_check,
    filter_dataset_steps=args.filter_dataset_steps,
    filter_dataset_token_size=args.filter_dataset_token_size,
    overfit = args.overfit,
    balance=args.balance,
)


### log the configurations to wandb
mode = args.wandb_mode

wandb.init(project="DreamPRM-AIME", mode=mode, config=args)

if not sanity_check:
    run_name = wandb.run.name
    print(f"Run name: {run_name}")
    if run_name is None:
        run_name = "default_run"
    ### edit the save path to include the run name
    args.weights_path = os.path.join(args.weights_path, run_name)
    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)

device = torch.device(args.device)
criterion = nn.MSELoss(reduction='none')
criterion_meta = nn.MSELoss()
criterion_CE = nn.BCEWithLogitsLoss(reduction='none')

lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = 1000



class Lower:
    def __init__(self):
        self.module = configure_module(args, device)
        self.dataloader, self.optim = self.configure_train_data_loader(), self.configure_optimizer()
    
    def forward(self, input_ids, attention_mask, no_grad=False):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask, no_grad=no_grad)

    def training_step(self, batch):
        labels = batch['label'].to(dtype=torch.float).to(device)
        print("Lower shapes", batch['input_ids'].shape, labels.shape, batch['correctness'])
        domain_strings = batch['dataset']
        if args.max_step_size == -1:
            max_step_size = len(batch['input_ids'])
        else:
            max_step_size = args.max_step_size
        score = unbatch_process(batch, device, self.forward, max_step_size)
         
        gc.collect()
        torch.cuda.empty_cache()
        score = score.float()
                    ### clip score between 0 and 1
        score = torch.clamp(score, min=1e-3, max=1 - 1e-3)
        score = torch.nan_to_num(score, nan=0.5, posinf=1.0 - 1e-3, neginf=1e-3)

       
        # print("lower",score.shape, labels.shape, batch['correctness'])
        if args.model_type == "token":
            # if args.dreamprm_loss:
            # #     ### dreamprm loss, score -> (B, T,)
            # #     score = score[labels!=-100]
            # #     score = torch.log(score / (1 - score))
            # #     mean_score = torch.mean(score, dim=1) #  (B,)
            # #     outputs = torch.sigmoid(mean_score)  #  (B,)
            # #     loss = criterion(outputs, labels)
            # else:
            ### avg cross entropy loss
            mask = (labels != -100).float()
            labels = torch.clamp(labels, min=0) # (B, T)
            
            loss = criterion_CE(score, labels)
            loss = loss * mask.float()
            loss = torch.sum(loss, dim=1) / mask.sum(dim=1)  # (B, )
            del mask, labels
                
        else:
            print("lower score after clamp", score)
            # score -> (B, )
            # labels -> (B, T)
            ### take last label that is not -100
            # labels: (B, T)
            non_filler = (labels != -100)  # (B, T), bool
            # flip along the time dimension
            reversed_non_filler = non_filler.flip(dims=[1])  # (B, T)
            # find index of the last non-(-100) (which is first True in the reversed tensor)
            reversed_index = torch.argmax(reversed_non_filler.float(), dim=1)  # (B,)
            # convert to correct index from the start
            index = labels.size(1) - 1 - reversed_index  # (B,)
            labels = labels[torch.arange(labels.size(0)),index]  # (B, )
            # print("tokenclf,lower",labels, index)
            loss = criterion(score, labels)
            
            del non_filler, reversed_non_filler, reversed_index, index
        
        print("DEBUG", "LOWER", loss )
        print("Loss shape", loss.shape)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            ## clip the loss to avoid NaN
            print("NaN loss detected, clipping to zero, lower loss")
            loss = torch.clamp(loss, min=0, max=1e3)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        loss = torch.mean(loss)  # (B, )    
        wandb.log({"inner_loss": loss.cpu().item(),})
        return loss

        

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        model = configure_module(args, device)
        
        if args.load_path != "":
            if args.peft_rank != -1:
                model.base_model = PeftModel.from_pretrained(model.base_model, f"{args.load_path}/lower_weights")
            else:
                model.base_model.from_pretrained(f"{args.load_path}/lower_weights")
            model.LN.load_state_dict(torch.load(f"{args.load_path}/lower_weights_LN.pt"))
        
        model.base_model = model.base_model.to(device)
        model.LN = model.LN.to(device)
        print("Lower model configured with device:", model.base_model.device)
        return model
    
    def configure_optimizer(self):
        optimizer = AdamW(
            self.module.parameters(),
            lr=args.lr,
        )
        return optimizer
    
    
    @torch.no_grad()
    def validation(self):
        log_dict = {}
        step_pred, gt, problem_pred, correctness = [], [], [], []
        for batch in tqdm(validation_dataloader):
            if args.max_step_size == -1:
                max_step_size = len(batch['input_ids'])
            else:
                max_step_size = args.max_step_size
                
            score = unbatch_process(batch, device, self.module, max_step_size, no_grad=True).cpu()
            outputs, metric_preds = get_pred(args, batch, score)
            step_pred+=metric_preds['step_pred']
            gt+=metric_preds['gt']
            problem_pred+= metric_preds['problem_pred']
            correctness+= metric_preds['correctness']

            
        step_metrics = binary_classification_metrics(step_pred, gt) ## dict of metrics
        problem_metrics = binary_classification_metrics(problem_pred, correctness) ## dict of metrics
        
        print("Step Metrics:", step_metrics)
        print("Problem Metrics:", problem_metrics)
        
        log_dict = {f"step_{k}": v for k, v in step_metrics.items()}
        log_dict.update({f"problem_{k}": v for k, v in problem_metrics.items()})
        
        
        if (args.overfit == -1) and (not args.sanity_check):
            if args.peft_rank != -1:
                self.module.base_model.save_pretrained(f"{args.weights_path}/lower_weights")
            else:
                self.module.base_model.save_pretrained(f"{args.weights_path}/lower_weights")

            torch.save(self.module.LN.state_dict(), f"{args.weights_path}/lower_weights_LN.pt")

        


        all_scores = {}
        for ds_name in dataloader_benchmark:
            for model_name in tqdm(dataloader_benchmark[ds_name]):
                print(f"Evaluating {ds_name} with {model_name}")
                test_dataloader = dataloader_benchmark[ds_name][model_name]
                predictions = None
                
                for batch in test_dataloader:
                    if args.max_step_size == -1:
                        max_step_size = len(batch['input_ids'])
                    else:
                        max_step_size = args.max_step_size
                    score = unbatch_process(batch, device, self.module, max_step_size, no_grad=True).cpu()
      
                    outputs, _ = get_pred(args, batch, score)

                    outputs = outputs.to(dtype=torch.float32)
                    if predictions is None:
                        predictions = outputs.numpy()
                    else:
                        predictions = np.concatenate((predictions, outputs.numpy()), axis=0)

                dataset = test_dataloader.dataset.dataset
                predictions, score = eval(dataset, predictions, ds_name)

                print(f"Dataset: {ds_name}, Model: {model_name}, Score: {score}")
                df = pd.DataFrame(predictions)
                df.to_csv(f"{args.weights_path}/{ds_name}_{model_name.split('/')[-1]}_predictions.csv", index=False)
                log_dict.update({f"{ds_name}/{model_name}_score": score})
                all_scores[f"{ds_name}_{model_name}"] = score
                
        wandb.log(log_dict)


        all_scores["loss"] = 1
        return all_scores
    
    
lower = Lower()

iter = 0
while iter < args.iteration_num:
    for batch in tqdm(lower.dataloader, desc="Training Lower Model"):
        if iter % args.save_every_iterations == 0:
            lower.module.eval()
            print("Validating lower model at iteration", iter)
            lower.validation()

        lower.module.train()
        
        loss = lower.training_step(batch)
        loss /= args.gradiant_accumulation  # scale the loss for gradient accumulation
        loss.backward()
        
        if iter % inner_log_every == 0:
            wandb.log({"iter": iter, "loss": loss.cpu().item()})
        
        if (iter + 1) % args.gradiant_accumulation == 0 or (iter + 1 == len(train_dataloader)):
            torch.nn.utils.clip_grad_norm_(lower.module.parameters(), max_norm=1.0)  # optional
            lower.optim.step()
            lower.optim.zero_grad(set_to_none=True)
            

        if iter>args.iteration_num:
            print("Reached max iterations, stopping training")
            break
        iter += 1
