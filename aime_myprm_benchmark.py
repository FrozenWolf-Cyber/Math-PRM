from datasets import load_dataset, load_from_disk
from prm_data import load_data_custom
from tqdm.auto import tqdm
# ds = load_from_disk("../PRM800k_cleaned")
device = "cuda" 
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import gc
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
    if "7B" in args.reward_model:
        model_inputs['input_ids'] = model_inputs['input_ids'][:, :2300]
        model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :2300]
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
        print("ids", model_inputs['input_ids'].shape)
        outputs = model(input_ids=model_inputs['input_ids'].to(device),
                        attention_mask=model_inputs['attention_mask'].to(device)).cpu()
        
        output = outputs.clone()
        del outputs
        
    del model_inputs
    gc.collect()
    torch.cuda.empty_cache()
    return output, token_masks

@torch.no_grad()
def forward_no_tokens(model, tokenizer, question, stepwise_solution, add_new_token=False):
    step_score = []
    for i in range(1,len(stepwise_solution)+1):
        output, _ = forward(model, tokenizer, question, stepwise_solution[:i], special_tokens=False, add_new_token=add_new_token)
        step_score.append(output.item())
        del output
        gc.collect()
        torch.cuda.empty_cache()
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

if args.load_path != "":
    if args.peft_rank != -1:
        model.base_model = PeftModel.from_pretrained(model, f"{args.load_path}/lower_weights")
    else:
        model.base_model.from_pretrained(f"{args.load_path}/lower_weights")
    model.LN.load_state_dict(torch.load(f"{args.load_path}/lower_weights_LN.pt"))
        
special_tokens, add_new_token = args.special_tokens, args.add_new_token



import os
import pandas as pd
from eval_aime import eval
paths = "aime_outputs/"
    
dataloader_benchmark = {}
prediction_history = {}
for ds in os.listdir(paths):
    dataloader_benchmark[ds] = {}
    ds_ = os.path.join(paths, ds)
    prediction_history[ds] = {}
    
    for model_out in os.listdir(ds_):
        model_out = os.path.join(ds_, model_out)
        path = os.listdir(model_out)[0]
        path = os.path.join(model_out, path)
        output = pd.read_json(path_or_buf=path, lines=True)
        if path!="aime_outputs/test2025-I/gemini/Gemini_AIME_Dataset_2024_25_I.jsonl":
            continue
        print(f"Loading {path} with {len(output)} samples")
        output["index"] = output.index
        output = output.explode(["generated_responses", "answers_correctness"]).reset_index(drop=True)
        # sep = "\n" if "o4" in model_out else "\n\n"
        sep = "\n\n"
        output["generated_responses"] = output["generated_responses"].apply(lambda x: x.split(sep))
        output.rename(columns={"generated_responses": "completions"}, inplace=True)
        output["labels"] = [
                    [val] * len(completions)
                    for val, completions in zip(output["answers_correctness"], output["completions"])
                ]
        output["subject"] = "Others" ## filler values
        
        all_problem_pred = []
        idx = 0
        for questions, solutions in tqdm(zip(output['problem'].tolist(), output['completions'].tolist()), total=len(output)):
            idx+=1
            if idx!=19:
                continue
            print(f"Processing {idx}/{len(output)}")
            if special_tokens:
                step_rewards, token_masks = forward(
                model=model,
                tokenizer=tokenizer,
                question=questions,
                stepwise_solution=solutions,
                special_tokens=True,
                add_new_token=add_new_token
                )
                step_rewards = make_step_rewards(step_rewards, token_masks)[0]
            else:
                step_rewards = forward_no_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    question=questions,
                    stepwise_solution=solutions,
                    add_new_token=add_new_token
                )

                score = torch.tensor(step_rewards)
                score = torch.log(score / (1 - score))
                score = score.sum()
                score = torch.sigmoid(score)


            if special_tokens:
                a = step_rewards[-1]
            else:
                a = score
            all_problem_pred+=[1 if a>=0.5 else 0]
            
        predictions, score = eval(output, all_problem_pred, ds)
        print(f"Model {model_out} on dataset {ds} has score {score}")
        prediction_history[ds][model_out] = predictions
        
        
import pickle
with open("myprm_aime_prediction_history.pkl", "wb") as f:
    pickle.dump(prediction_history, f)