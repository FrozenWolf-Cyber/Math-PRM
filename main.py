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
from transformers import AdamW
import numpy as np
from copy import deepcopy
import pandas as pd
from eval_aime import *
import gc
from model import round_robin_batch_ordering
 
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
    dataloader_benchmark
) = build_dataloader(
    tokenizer=tokenizer,
    train_batch_size= args.train_batch_size,
    meta_batch_size= args.meta_batch_size,
    token_based=args.model_type == "token",
    add_new_token=args.add_new_token,
    meta_dataset=args.meta_dataset,
    sanity_check=sanity_check,
    filter_dataset_steps=args.filter_dataset_steps,
    filter_dataset_token_size=args.filter_dataset_token_size
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


class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        labels = batch['label'].to(device) ## (B, T)
        correctness = torch.tensor(batch['correctness'], dtype=torch.float).to(device) ## (B, )
        print("Upper shapes", batch['input_ids'].shape, labels.shape, correctness)
        max_meta_steps_grad = args.max_meta_steps_grad
        if args.max_step_size == -1:
            max_step_size = len(batch['input_ids'])
        else:
            max_step_size = args.max_step_size
        ### the last max_meta_steps_grad samples alone needs to be performaed with grad, rest without grad
        score_nograd = None
        if max_meta_steps_grad != -1 :
            max_meta_steps_grad*=len(set(batch['index']))
            
            reverse_indices = round_robin_batch_ordering(batch)
            with torch.set_grad_enabled(False):
                if len(batch['input_ids']) > max_meta_steps_grad:
                    print("Number of grads steps considered B*grad_Step",max_meta_steps_grad)
                    score_nograd = unbatch_process(batch, device, self.lower, max_step_size, no_grad=True, start=0, end=len(batch['input_ids'])-max_meta_steps_grad)
            with torch.set_grad_enabled(True):
                score = unbatch_process(batch, device, self.lower, max_step_size, no_grad=False, start=max(0,len(batch['input_ids'])-max_meta_steps_grad), end=len(batch['input_ids']))
            if score_nograd is not None:
                score = torch.cat((score_nograd, score), dim=0)
          
            batch['index'] = batch['index'][reverse_indices]
            batch['index'] = batch['index'].tolist()
            print("Batch index after reverse:", batch['index'])
            score = score[reverse_indices] # (B, T*(T+1)/2)
        else:
            score = unbatch_process(batch, device, self.lower, max_step_size)
            
        # print("Device:", score.device)
        ### clip score between 0 and 1
        score = score.float()
        score1 = torch.clamp(score, min=1e-3, max=1 - 1e-3)
        score = torch.clamp(score, min=1e-3, max=1 - 1e-3)
        score = torch.nan_to_num(score, nan=0.5, posinf=1.0 - 1e-3, neginf=1e-3)

        
        if args.model_type == "token":
            if args.dreamprm_loss: ### using overall problem solution correctness
                ### dreamprm loss, score -> (B, T,)
                mask = (labels != -100).float()
                score = score / (1 - score)
                score = torch.log(score) # (B, T)
                ### set nan scores mask t zero
                score[torch.isnan(score)] = 0
                score[torch.isinf(score)] = 0
                score = score* mask # (B, T)
                mean_score = torch.sum(score, dim=1) / mask.sum(dim=1) # (B, )
                outputs = torch.sigmoid(mean_score) # (B, )
                loss = criterion_meta(outputs, correctness)
            else:
                ### avg cross entropy loss -> per step annotations
                mask = (labels != -100).float()
                labels = torch.clamp(labels, min=0).float() # (B, T)
                
                loss = criterion_CE(score, labels)
                loss = loss * mask.float()
                loss = loss.sum() / mask.sum()
                
        else:
            print("Upper score ", score)
            # print("batch['index']:", batch['index'], score.device)
            ### using overall problem solution correctness
            nproblems = set(batch['index']) # [0, 1, 2, B_Size-1]
            # score -> (B * T*(T+1)/2) So [A_0_1, A_0_2,.. B_0_1, B_0_2,...] in cummulative order
            score = torch.log(score / (1 - score))
            print(score.shape)
            outputs = []
            for i in nproblems:
                mean_score = 0
                step = 0
                for j in range(len(score)):
                    if batch['index'][j] == i:
                        if torch.isnan(score[j]).any() or torch.isinf(score[j]).any():
                            print("!!!!!NaN subs score detected, skipping this score")
                            continue
                        mean_score += score[j]
                        step += 1
                        
                print("Step:", step, "Mean Score:", mean_score)
                mean_score /= max(step, 1)
                outputs.append(mean_score)
                
            ## outputs -> (B, )
            ## label -> ## (B * T*(T+1)/2)
            outputs = torch.stack(outputs) # (B, )
            print("Outputs shape:", outputs.shape, "Correctness shape:", correctness.shape)
            outputs = torch.sigmoid(outputs)
            print("Upper outputs:", outputs)
            loss = criterion_meta(outputs, correctness)
                
        print("DEBUG", "UPPER", loss )
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            ## clip the loss to avoid NaN
            print("NaN loss detected, clipping to zero upper loss")
            loss = torch.clamp(loss, min=0, max=1e3)
            
        upper_loss.append(loss.item())

        # torch.cuda.empty_cache()
        if len(upper_loss) == 5:
            mean_outer_loss = np.mean(upper_loss)
            print(f"Outer Loss: {mean_outer_loss}")
            wandb.log({"outer_loss": mean_outer_loss})
            upper_loss.clear()

        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        meta_net = DomainTable(
            domain_list
        )
        return meta_net

    def configure_optimizer(self):
        meta_optimizer = AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.weight_decay
        )
        return meta_optimizer


class Lower(ImplicitProblem):
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
        
        if args.baseline or args.retrain:
            loss = torch.mean(loss)  # (B, )    
            return loss

        loss = loss.unsqueeze(1)  # (B, 1)
        # print("lower loss",loss)
        weighted_loss = torch.mean(self.upper(domain_strings, loss))
        
        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            ## clip the loss to avoid NaN
            print("NaN loss detected, clipping to zero in lower weighted loss")
            weighted_loss = torch.clamp(weighted_loss, min=0, max=1e3)
        
        
        lower_loss.append(torch.mean(loss).item())
        lower_weighted_loss.append(torch.mean(weighted_loss).item())
        if len(lower_loss) == 100:
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({"inner_loss": mean_inner_loss,
                       "inner_weighted_loss": mean_inner_weighted_loss, })
            print(f"Inner Loss: {mean_inner_loss}, Inner Weighted Loss: {mean_inner_weighted_loss}")
            lower_loss.clear()
            lower_weighted_loss.clear()
        # torch.cuda.empty_cache()
        return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        model = configure_module(args, device)
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

    # def configure_scheduler(self):
    #     scheduler = optim.lr_scheduler.StepLR(
    #         self.optimizer, step_size = args.scheduler_step_size, gamma=args.scheduler_gamma
    #     )
    #     return scheduler



class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        if args.peft_rank != -1:
            torch.save(self.lower.module.state_dict(), f"{args.weights_path}/lower_weights.pt")
        else:
            self.lower.module.base_model.save_pretrained(f"{args.weights_path}/lower_weights")
            torch.save(self.lower.module.LN.state_dict(), f"{args.weights_path}/lower_weights_LN.pt")
        torch.save(
            self.upper.state_dict(),
            f"{args.weights_path}/domain_weights.pt",
        )
        
        #### log this domain weights to wandb # self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))
        wts = self.upper.module.raw_weights
        print("Raw Weights:", wts)
        positive_weights = torch.nn.functional.softplus(wts)
        print("Positive Weights:", positive_weights)
        mean_weights = positive_weights.mean()
        wts = (positive_weights / mean_weights).cpu().numpy()
        
        print("Domain Weights:", wts)
        ### separate line for each domain
        to_log = {inv_domain_list[i]: wts[i] for i in range(len(domain_list))}
        wandb.log(to_log)
            

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
                    score = unbatch_process(batch, device, self.lower, max_step_size, no_grad=True).cpu()
      
                    if args.model_type == "token":
                        # score -> (B, T,)
                        labels = batch['label'].to(dtype=torch.float).to("cpu")
                        mask = (labels != -100)
                        score[mask] = 0
                        score = score / (1 - score)
                        score[mask] = 1
                        score = torch.log(score)
                        score = score * mask.float()  # (B, T)
                        mean_score = torch.mean(score, dim=1)
                        outputs = torch.sigmoid(mean_score) ## (B, )


                    else:
                        # score -> (B * T*(T+1)/2) So [A_0_1, A_0_2,.. B_0_1, B_0_2,...] in cummulative order
                        nproblems = set(batch['index']) # [0, 1, 2, B_Size-1]
                        score = torch.log(score / (1 - score))
                        outputs = []
                        for i in nproblems:
                            mean_score = 0
                            step = 0
                            for j in range(len(score)):
                                if batch['index'][j] == i:
                                    mean_score += score[j]
                                    step += 1
                            mean_score /= step

                            outputs.append(mean_score)

                        outputs = torch.stack(outputs) # (B, )
                        outputs = torch.sigmoid(outputs)

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
                wandb.log({f"{ds_name}/{model_name}_score": score})
                all_scores[f"{ds_name}_{model_name}"] = score
        

        all_scores["loss"] = 1
        return all_scores


upper_config = Config(type="darts", precision=args.precision, retain_graph=True, gradient_clipping=args.gradient_clipping, gradient_accumulation=args.gradiant_accumulation)
lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps, gradient_accumulation=args.gradiant_accumulation, gradient_clipping=args.gradient_clipping)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)
upper = Upper(name="upper", config=upper_config)
lower = Lower(name="lower", config=lower_config)

if args.baseline or args.retrain:
    problems = [lower]
    u2l, l2u = {}, {}
else:
    problems = [upper, lower]
    u2l = {upper: [lower]}
    l2u = {lower: [upper]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()
