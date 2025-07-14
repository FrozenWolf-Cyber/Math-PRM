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
 
parser = argparse.ArgumentParser(description="DreamPRM")
parser.add_argument('--weights_path', type=str)
parser.add_argument("--iteration_num", type=int, default=10000)
parser.add_argument("--save_every_iterations", type=int, default=1000)
parser.add_argument("--unroll_steps", type=int, default=5)
parser.add_argument("--gradiant_accumulation", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--scheduler_step_size", type=int, default=5000)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_lr", type=float, default=0.01)
parser.add_argument("--meta_weight_decay", type=float, default=0.0)
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--num_meta", type=int, default=1000)
parser.add_argument("--imbalanced_factor", type=int, default=None)
parser.add_argument("--corruption_type", type=str, default=None)
parser.add_argument("--corruption_ratio", type=float, default=0.0)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--meta_batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=120)
parser.add_argument("--meta_interval", type=int, default=1)
parser.add_argument("--paint_interval", type=int, default=20)
parser.add_argument("--dreamprm_loss", action="store_true")
parser.add_argument("--model_type", type=str, default="token")
parser.add_argument("--meta_dataset", type=str, default="AIME", help="AIME or PRM800K or both")
parser.add_argument("--sanity_check", action="store_true")
parser.add_argument("--add_new_token",  action="store_true", help="Whether to add new token <PRM_STEP_SCORE> to the tokenizer")
parser.add_argument("--freeze_till_last",  action="store_true", help="Freeze till last layer")
parser.add_argument("--freeze_tokens",  action="store_true", help="Freeze other than newly added tokens")


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
    args.save_every_iterations = 2
    args.iteration_num = 5
    args.max_epoch = 2


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
)


### log the configurations to wandb

wandb.init(project="DreamPRM-AIME", mode="offline" if sanity_check else "online", config=args)

if not sanity_check:
    run_name = wandb.run.name
    print(f"Run name: {run_name}")
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
        score = self.lower(batch['input_ids'].to(device),
                                     batch['attention_mask'].to(device))
        
        if args.model_type == "token":
            if args.dreamprm_loss: ### using overall problem solution correctness
                ### dreamprm loss, score -> (B, T,)
                mask = (labels != -100).float()
                score[labels==-100] = 0
                score = score / (1 - score)
                score[labels==-100] = 1
                score = torch.log(score) # (B, T)
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
            ### using overall problem solution correctness
            print(batch['index'])
            nproblems = set(batch['index']) # [0, 1, 2, B_Size-1]
            # score -> (B * T*(T+1)/2) So [A_0_1, A_0_2,.. B_0_1, B_0_2,...] in cummulative order
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
                
            ## outputs -> (B, )
            ## label -> ## (B * T*(T+1)/2)
            print("upper",[len(i) for i in outputs])
            outputs = torch.stack(outputs) # (B, )
            outputs = torch.sigmoid(outputs)
            loss = criterion_meta(outputs, correctness)
                
        
        upper_loss.append(loss.item())

        # torch.cuda.empty_cache()
        if len(upper_loss) == 10:
            mean_outer_loss = np.mean(upper_loss)
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
    def forward(self, input_ids, attention_mask):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask)

    def training_step(self, batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(dtype=torch.float).to(device)
        domain_strings = batch['dataset']
        score = self.forward(input_ids=input_ids, attention_mask=attention_mask)
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
                
        else:
            # score -> (B, )
            # labels -> (B, T)
            ### take last label that is not -100
            non_filler = (labels != -100).float()
            index = torch.argmax(non_filler, dim=1)
            labels = labels[torch.arange(labels.size(0)),index]  # (B, )
            loss = criterion(score, labels)
            print("lower",score.shape, labels.shape, index.shape)
        
        if args.baseline or args.retrain:
            return loss

        loss = loss.unsqueeze(1)  # (B, 1)
        weighted_loss = torch.mean(self.upper(domain_strings, loss))
        lower_loss.append(torch.mean(loss).item())
        lower_weighted_loss.append(torch.mean(weighted_loss).item())
        if len(lower_loss) == 100:
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({"inner_loss": mean_inner_loss,
                       "inner_weighted_loss": mean_inner_weighted_loss, })
            lower_loss.clear()
            lower_weighted_loss.clear()
        # torch.cuda.empty_cache()
        return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        return configure_module(args, device)


        

    def configure_optimizer(self):
        optimizer = AdamW(
            self.module.parameters(),
            lr=args.lr,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size = args.scheduler_step_size, gamma=args.scheduler_gamma
        )
        return scheduler



class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        torch.save(self.lower.module.state_dict(), f"{args.weights_path}/lower_weights.pt")

        torch.save(
            self.upper.state_dict(),
            f"{args.weights_path}/domain_weights.pt",
        )
        
        #### log this domain weights to wandb # self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))
        wts = self.upper.module.raw_weights.cpu().numpy()
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
                    score = self.lower(batch['input_ids'].to(device),
                                        batch['attention_mask'].to(device))
                    
                    if args.model_type == "token":
                        # score -> (B, T,)
                        labels = batch['label'].to(dtype=torch.float).to(device)
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
                        predictions = outputs.cpu().numpy()
                    else:
                        predictions = np.concatenate((predictions, outputs.cpu().numpy()), axis=0)

                dataset = test_dataloader.dataset.dataset
                predictions, score = eval(dataset, predictions, ds_name)

                print(f"Dataset: {ds_name}, Model: {model_name}, Score: {score}")
                df = pd.DataFrame(predictions)
                df.to_csv(f"{args.weights_path}/{ds_name}_{model_name.split('/')[-1]}_predictions.csv", index=False)
                wandb.log({f"{ds_name}/{model_name}_score": score})
                all_scores[f"{ds_name}_{model_name}"] = score
        

        all_scores["loss"] = 1
        return all_scores


upper_config = Config(type="darts", precision=args.precision, retain_graph=True)
lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps, gradient_accumulation=args.gradiant_accumulation)
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
