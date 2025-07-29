# train_prm.py
from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np
import wandb
import pandas as pd
from tqdm.auto import tqdm
from prm_data import *
from argparse import ArgumentParser
from peft import LoraConfig, TaskType, get_peft_model
parser = ArgumentParser()
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--peft_rank", type=int, default=-1, help="Rank for PEFT, -1 for no PEFT")
parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA")
parser.add_argument("--train_batch_size", type=int, default=4)
args = parser.parse_args()
print("Trraining with args:", args)

wandb.init(project="DreamPRM-AIME", mode="online")
run_name = wandb.run.name
print(f"Run name: {run_name}")

model = args.reward_model
tokenizer = AutoTokenizer.from_pretrained(model)
new_tokens = ['<PRM_STEP_SCORE>']
num_added_tokens = tokenizer.add_tokens(new_tokens)

dataloader_benchmark = build_vanilla_inference_dataloader(
    tokenizer=tokenizer,
    meta_batch_size=1
)

path = os.path.join("weights/", run_name)
if not os.path.exists(path):
    os.makedirs(path)


# class DebugTokenClassifier(torch.nn.Module):
#     def __init__(self, model_name, num_labels=2):
#         super().__init__()
#         self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
#         self.model.resize_token_embeddings(len(tokenizer))

#     def forward(self, input_ids=None, attention_mask=None, **kwargs):
#         print("Input IDs:\n", input_ids.shape)
#         print("Attention Mask:\n", attention_mask.shape)
#         return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


model = AutoModelForTokenClassification.from_pretrained(model, num_labels=2)
# model = DebugTokenClassifier(model, num_labels=2)
if args.peft_rank > 0:
    
    peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=args.peft_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
            )
    model = get_peft_model(model, peft_config)

train_dataset = load_dataset("FrozenWolf/prm800k")
val_dataset = train_dataset['test']
train_dataset = train_dataset['train']

training_args = PRMConfig(
    output_dir=f"{path}/Output-{args.reward_model.replace('/', '-')}",
    eval_steps=2000,    
    logging_dir="./logs",     
    logging_strategy="steps",     
    logging_steps=100,            
    report_to="wandb",         
    save_strategy="epoch",      
    save_total_limit=2,            
    num_train_epochs=5, 
    step_separator=new_tokens[0],      
    per_gpu_train_batch_size=args.train_batch_size,  per_gpu_eval_batch_size=args.train_batch_size
)
trainer = PRMTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()

### save the model
model.save_pretrained(f"{path}/vanilla_{model}")


from eval_aime import eval
device="cuda"
all_scores = {}
with torch.no_grad():
    for ds_name in dataloader_benchmark:
        for model_name in tqdm(dataloader_benchmark[ds_name]):
            print(f"Evaluating {ds_name} with {model_name}")
            test_dataloader = dataloader_benchmark[ds_name][model_name]
            predictions = None
            for batch in test_dataloader:
                
                score = model(batch['input_ids'].to(device),
                        batch['attention_mask'].to(device)).logits
                # score -> (B, T,)
                outputs = torch.argmax(score[:,-1], dim=-1).cpu()

                outputs = outputs.to(dtype=torch.float32)
                if predictions is None:
                    predictions = outputs.numpy()
                else:
                    predictions = np.concatenate((predictions, outputs.numpy()), axis=0)
                    
            dataset = test_dataloader.dataset.dataset
            predictions, score = eval(dataset, predictions, ds_name)
            
            print(f"Dataset: {ds_name}, Model: {model_name}, Score: {score}")
            df = pd.DataFrame(predictions)
            df.to_csv(f"{path}/vanilla_{ds_name}_{model_name.split('/')[-1]}_predictions.csv", index=False)
            wandb.log({f"{ds_name}/{model_name}_score": score})
            all_scores[f"{ds_name}_{model_name}"] = score