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
parser.add_argument('--train_json_file', type=str)
parser.add_argument('--meta_json_file', type=str)
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
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument("--num_meta", type=int, default=1000)
parser.add_argument("--imbalanced_factor", type=int, default=None)
parser.add_argument("--corruption_type", type=str, default=None)
parser.add_argument("--corruption_ratio", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=120)
parser.add_argument("--meta_interval", type=int, default=1)
parser.add_argument("--paint_interval", type=int, default=20)
parser.add_argument("--prm_loss", action="store_true")
parser.add_argument("--model_type", type=str, default="token")

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
print(domain_list)



tokenizer = AutoTokenizer.from_pretrained(args.reward_model, trust_remote_code=True)
new_tokens = ['<PRM_STEP_SCORE>']
num_added_tokens = tokenizer.add_tokens(new_tokens)


sampler = None
resume_idxes = None
resume_labels = None

(
    train_dataloader,
    meta_dataloader,
) = build_dataloader(
    tokenizer=tokenizer,
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size,
    last_only=args.prm_loss,
)

dataloader_benchmark = build_inference_dataloader(
    tokenizer=tokenizer,
    batch_size= args.batch_size,
    last_only=args.prm_loss,
)


wandb.init(project="DreamPRM-AIME", mode="offline")

device = torch.device(args.device)
criterion = nn.MSELoss()
criterion_meta = nn.MSELoss()
lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = 1000


class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        labels = batch['labels'].to(device)
        score = self.inner(batch['input_ids'].to(device),
                                     batch['attention_mask'].to(device),
                                     labels if not args.prm_loss else None, last_nly=args.prm_loss)
        
        if args.model_type == "token":
            if args.prm_loss:
                #### DreamPRM Product Loss
                score = score[labels!=-100]
                score = torch.log(score / (1 - score))
                mean_score = torch.mean(score, dim=1)
                outputs = torch.sigmoid(mean_score)
                loss = criterion_meta(outputs, labels)
            else:
                ### avg cross entropy loss
                score, loss = score
        else:
            nproblems = set(batch['index'])
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
                
            outputs = torch.stack(outputs, device=device)
            outputs = torch.sigmoid(outputs)
            loss = criterion_meta(outputs, labels)
                
        
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
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask, pixel_values, image_grid_thw)

    def training_step(self, batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(dtype=torch.float).to(device)
        domain_strings = batch['index']
        score = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels if not args.prm_loss else None)
        
        if args.model_type == "token":
            if args.prm_loss:
                score = score[labels!=-100]
                score = torch.log(score / (1 - score))
                mean_score = torch.mean(score, dim=1)
                outputs = torch.sigmoid(mean_score)
                loss = criterion(outputs, labels)
            else:
                score, loss = score
                
        else:
            loss = criterion(score, labels)
        
        if args.baseline or args.retrain:
            return loss

        weighted_loss = self.upper(domain_strings, loss)
        lower_loss.append(loss.item())
        lower_weighted_loss.append(weighted_loss.item())
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
        if args.model_type == "token":
            return QwenMathTokenClf_RM(device, args.model_path)
        else:
            return QwenMathCondGen_RM(device, args.model_path)

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
        torch.save(
            self.inner.module.LN.state_dict(), f"{args.weights_path}/LN_weights.pt"
        )
        self.inner.module.base_model.save_pretrained(f"{args.weights_path}/base_model")
        torch.save(
            self.outer.state_dict(),
            f"{args.weights_path}/domain_weights.pt",
        )
        
        all_scores = {}
        for ds_name in dataloader_benchmark:
            for model_name in dataloader_benchmark[ds_name]:
                test_dataloader = dataloader_benchmark[ds_name][model_name]
                predictions = None

                for batch in test_dataloader:
                    score = self.inner(batch['input_ids'].to(device),
                                        batch['attention_mask'].to(device),
                                        last_nly=args.prm_loss)

                    if args.model_type == "token":
                        labels = batch['labels'].to(device)
                        score = score[labels!=-100]
                        score = torch.log(score / (1 - score))
                        mean_score = torch.mean(score, dim=1)
                        outputs = torch.sigmoid(mean_score) ## (B, )


                    else:
                        nproblems = set(batch['index'])
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

                        outputs = torch.stack(outputs, device=device)
                        outputs = torch.sigmoid(outputs)

                    if predictions is None:
                        predictions = outputs.cpu().numpy()
                    else:
                        predictions = np.concatenate((predictions, outputs.cpu().numpy()), axis=0)

                dataset = test_dataloader.dataset.data
                predictions, score = eval(dataset, predictions)

                print(f"Dataset: {ds_name}, Model: {model_name}, Score: {score}")
                df = pd.DataFrame(predictions)
                df.to_csv(f"{args.weights_path}/{ds_name}_{model_name}_predictions.csv", index=False)
                wandb.log({f"{ds_name}/{model_name}_score": score})
                all_scores[f"{ds_name}_{model_name}"] = score
        
        
            
        
        all_scores["loss"] = 1
        return all_scores


upper_config = Config(type="darts", precision=args.precision, retain_graph=True)
lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps, gradient_accumulation=args.gradiant_accumulation)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.parser.save_every_iterations,
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
