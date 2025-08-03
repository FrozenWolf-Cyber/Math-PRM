from datasets import load_dataset, load_from_disk
from prm_data import load_data_custom
from tqdm.auto import tqdm

device = "cuda" 
# ds = load_from_disk("../PRM800k_cleaned")
ds = load_data_custom("FrozenWolf/prm800k")
dst = ds['test'].to_pandas()


import pandas as pd
def select_top_20(group):
    # Ensure label length is available
    group = group.copy()
    group['label_len'] = group['labels'].apply(len)
    
    # Get top 10 True correctness with unique prompts
    true_rows = (
        group[group['correctness'] == True]
        .sort_values('label_len', ascending=False)
        .drop_duplicates('prompt')
        .head(10)
    )

    # Get top 10 False correctness with unique prompts
    false_rows = (
        group[group['correctness'] == False]
        .sort_values('label_len', ascending=False)
        .drop_duplicates('prompt')
        .head(10)
    )
    
    return pd.concat([true_rows, false_rows])

# Apply this for each subject
dst_selected = dst.groupby('subject', group_keys=False).apply(select_top_20)

# Optional cleanup
dst_selected = dst_selected.drop(columns='label_len').reset_index(drop=True)


import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
SEP_TOKEN = '<PRM_STEP_SCORE>'

def chat_template( question, steps):
        global SEP_TOKEN
        steps = f' {SEP_TOKEN} \n'.join(steps)
        
        prompt = f'''Question:
{question}
Answer:
{steps} {SEP_TOKEN}'''
        
        return prompt
    
def chat_template_no_special( question, steps):
        global SEP_TOKEN
        steps = '\n'.join(steps)
        
        prompt = f'''Question:
{question}
Answer:
{steps} {SEP_TOKEN}'''
        
        return prompt


def make_step_rewards(logits, token_masks):  
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities.cpu() * token_masks  # bs, seq_len

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0]  # get p(label=1)
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res

@torch.no_grad()
def forward(model, tokenizer, question, stepwise_solution, special_tokens, add_new_token):
    if not add_new_token:
        assert '<|im_end|>' in tokenizer.special_tokens_map['additional_special_tokens'], "Please check if <|im_end|> token to the tokenizer vocab."
        global SEP_TOKEN
        SEP_TOKEN = '<|im_end|>'

    if special_tokens:
        conversation_str = chat_template(question, stepwise_solution)
    else:
        conversation_str = chat_template_no_special(question, stepwise_solution)    
        
    # print("Conversation String:", conversation_str)
    
    model_inputs = tokenizer([conversation_str], return_tensors="pt")
    
    if SEP_TOKEN == '<PRM_STEP_SCORE>':
        SEP = len(tokenizer)-1
    else:
        SEP = tokenizer(SEP_TOKEN)['input_ids'][0]
            
    if special_tokens:
        token_masks = torch.ones_like(model_inputs['input_ids']).long()
        token_masks[(model_inputs['input_ids']!=SEP)] = 0
    else:
        token_masks = torch.zeros_like(model_inputs['input_ids']).long()
        token_masks[-1] = 1
            
    with torch.no_grad():
        outputs = model(input_ids=model_inputs['input_ids'].to(device),
                        attention_mask=model_inputs['attention_mask'].to(device),no_grad=True)
        

    return outputs, token_masks
@torch.no_grad()
def forward_no_tokens(model, tokenizer, question, stepwise_solution, add_new_token=False):
    step_score = []
    for i in range(1,len(stepwise_solution)+1):
        output, _ = forward(model, tokenizer, question, stepwise_solution[:i], special_tokens=False, add_new_token=add_new_token)
        step_score.append(output.item())
    return step_score
        

from peft import PeftModel    
from model import *

import argparse

parser = argparse.ArgumentParser(description="Model Configuration Arguments")
parser.add_argument("--load_path", type=str, default="", help="Path to load model from")
parser.add_argument("--peft_rank", type=int, default=-1, help="PEFT rank")
parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA")
parser.add_argument("--special_tokens", action="store_true", help="Use special tokens")
parser.add_argument("--add_new_token", action="store_true", help="Add new token")
parser.add_argument("--model_type", type=str, default="clf", help="Type of model")
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2-0.5B", help="Reward model name")
parser.add_argument("--freeze_till_last", action="store_true", help="Freeze all layers till last")
parser.add_argument("--freeze_tokens", action="store_true", help="Freeze token embeddings")
parser.add_argument("--freeze_all_but_bias", action="store_true", help="Freeze all layers except bias")


args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(args.reward_model, trust_remote_code=True)
if args.add_new_token:
    new_tokens = ['<PRM_STEP_SCORE>']
    num_added_tokens = tokenizer.add_tokens(new_tokens)


model = configure_module(args, device)



model = model.to(device)
special_tokens, add_new_token = args.special_tokens, args.add_new_token
special_tokens = args.model_type == "token"
if args.special_tokens!= special_tokens:
    print("!!!!!Warning: special_tokens argument does not match model type. Using model type's special tokens setting.")

print("Starting evaluation...")
history = []
all_step_pred, all_step_labels, all_problem_correctness, all_problem_pred = [], [], [], []
for questions, solutions, step_labels, correctness in tqdm(zip(
    dst_selected['prompt'].tolist(),
    dst_selected['completions'].tolist(),
    dst_selected['labels'].tolist(),
    dst_selected['correctness'].tolist()
), total=len(dst_selected)):
    
    raw_output = None
    if special_tokens:
        step_rewards, token_masks = forward(
        model=model,
        tokenizer=tokenizer,
        question=questions,
        stepwise_solution=solutions,
        special_tokens=True,
        add_new_token=add_new_token
        )
        raw_output = [step_rewards, token_masks]
        print(step_rewards.shape, token_masks)
        step_rewards = make_step_rewards(step_rewards, token_masks)[0]
    else:
        step_rewards = forward_no_tokens(
            model=model,
            tokenizer=tokenizer,
            question=questions,
            stepwise_solution=solutions,
            add_new_token=add_new_token
        )
        raw_output = step_rewards
        score = torch.tensor(step_rewards)
        score = torch.log(score / (1 - score))
        score = score.sum()
        score = torch.sigmoid(score).item()
        
    correctness = 1 if correctness else 0
        
    history.append({
		'question': questions,
		'solutions': solutions,
		'step_rewards': step_rewards,
		'step_labels': step_labels,
		'correctness': correctness,
        'raw_output': raw_output
	})
    all_step_pred+= [1 if i>=0.5 else 0 for i in step_rewards]
    all_step_labels+= [1 if i else 0 for i in step_labels.tolist()]
    all_problem_correctness+=[correctness]
    if special_tokens:
        a = step_rewards[-1]
    else:
        a = score
    all_problem_pred+=[1 if a>=0.5 else 0]
    
    
from model import binary_classification_metrics
step_metrics = binary_classification_metrics(all_step_pred, all_step_labels)
problem_metric = binary_classification_metrics(all_problem_pred, all_problem_correctness)

print("Step-wise Metrics:", step_metrics)
print("Problem-wise Metrics:", problem_metric)


## save with unique name
import os
wandb_name = args.load_path.split("/")[-1]
name = f"my_qwen_benchmark_history_{wandb_name}.pkl"
import pickle
with open(name, 'wb') as f:
    pickle.dump(history, f)