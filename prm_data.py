import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, RandomSampler, SequentialSampler
import re
from datasets import load_dataset
from datasets import Dataset as HF_Dataset
import pandas as pd
import os
import numpy as np

SANITY_CHECK = False

subjects_map = {'Algebra':0,
 'Counting & Probability':1,
 'Geometry':2,
 'Intermediate Algebra':3,
 'Number Theory':4,
 'Prealgebra':5,
 'Precalculus':6,
 'Others':7}

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





class QwenMathDataset(Dataset):
    '''
    This one generates a single trajectory per iteration and ALL steps accuracy like:
    [A, B, C] -> [T, F, T]
    special_tokens -> True for Token based model
    '''
    def __init__(self, data, tokenizer, special_tokens=True, inference=False, has_subjects=True):
        self.dataset = data
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.inference = inference
        self.has_subjects = has_subjects
        if SEP_TOKEN == '<PRM_STEP_SCORE>':
            self.SEP = len(self.tokenizer)-1
        else:
            print("Separator token is:", SEP_TOKEN, self.tokenizer(SEP_TOKEN))
            self.SEP = self.tokenizer(SEP_TOKEN)['input_ids'][0]
            
    def __len__(self):
        if SANITY_CHECK:
            print("Sanity check mode: returning 10 samples.")
            return 10
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]['prompt']
        completions = self.dataset[idx]['completions']
        raw_labels = self.dataset[idx]['labels']
        if  not self.has_subjects:
            dset = subjects_map['Others']
        else:
            dset = subjects_map[self.dataset[idx]['subject']]

        if self.special_tokens:
            text = chat_template(prompt, completions)
        else:
            text = chat_template_no_special(prompt, completions)    
        

        model_inputs = self.tokenizer([text], return_tensors="pt")
        labels = torch.ones_like(model_inputs['input_ids']).long()

        if self.special_tokens:
            labels[(model_inputs['input_ids']!=self.SEP)] = -100
            labels[(model_inputs['input_ids']==self.SEP)] = torch.tensor(raw_labels).long()
        else:
            labels *= -100
            labels[-1] = torch.tensor(raw_labels).long()[-1]

        model_inputs["dataset"] = dset
        model_inputs["labels"] = labels
        
        if self.inference:
            correctness = 1
        else:
            correctness = self.dataset[idx]['correctness']
            
        return [model_inputs['input_ids']], [model_inputs['attention_mask'].long()], [labels], [dset], [idx], [correctness]
    



class QwenMathMetaDataset(Dataset):
    '''
    This one generate all combinations of a trajectory per iteration and its LAST step accuracy ALONE like:
    [A, B, C] -> [T, F, T]
    [
        [A], -> [T],
        [A, B], -> [F],
        [A, B, C] -> [T]
    ]
    where A, B, C are the steps of the trajectory.
    '''
    def __init__(self, data, tokenizer, inference=False):
        self.dataset = data
        self.tokenizer = tokenizer
        self.inference = inference

    def __len__(self):
        if SANITY_CHECK:
            print("Sanity check mode: returning 10 samples.")
            return 10
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]['prompt']
        completions = self.dataset[idx]['completions']
        raw_labels = self.dataset[idx]['labels']

        inputs, attns, labels, index = [], [], [], []
        
        for step_idx in range(1, len(completions)+1):
            text = chat_template_no_special(prompt, completions[:step_idx])   
            model_inputs = self.tokenizer(text, return_tensors="pt")
            label = torch.ones_like(model_inputs['input_ids'])
            label *= -100
            label[-1] = torch.tensor(raw_labels[:step_idx])[-1].long()

            inputs.append(model_inputs['input_ids'])
            attns.append(model_inputs['attention_mask'].long())
            labels.append(label.long())
            index.append(idx)

        if self.inference:
            correctness = 1
        else:
            correctness = self.dataset[idx]['correctness']
        return inputs, attns, labels, index, [[correctness]]


    
    
def build_dataloader(
        tokenizer, train_batch_size, meta_batch_size, token_based=True, add_new_token=True, meta_dataset="AIME", # "AIME" or "PRM800K"
        sanity_check=False
):
    
    if not add_new_token:
        print(tokenizer.special_tokens_map['additional_special_tokens'])
        assert '<|im_end|>' in tokenizer.special_tokens_map['additional_special_tokens'], "Please check if <|im_end|> token to the tokenizer vocab."
        global SEP_TOKEN
        SEP_TOKEN = '<|im_end|>'

    if sanity_check:
        global SANITY_CHECK
        SANITY_CHECK = True

    def collate_merge_minibatch(batch):
        if len(batch[0]) == 6:
            names = ["input_ids", "attention_mask", "label", "dataset", "index", "correctness"]
        elif len(batch[0]) == 5:
            names = ["input_ids", "attention_mask", "label", "index", "correctness"]
        else:
            raise ValueError("Batch items must contain 4 or 5 elements.")

        merged = {}

        for name_idx, item in enumerate(zip(*batch)):
            j = []
            for i in item:
                j+=i

            if isinstance(j[0], torch.Tensor):
                max_len = max([i.shape[-1] for i in j ])
            for i in range(len(j)):    
                if isinstance(j[i], torch.Tensor):
                    if names[name_idx] == "input_ids":
                        j[i] = torch.cat((j[i], torch.ones(1, max_len - j[i].shape[-1], dtype=j[i].dtype)*tokenizer.pad_token_id), dim=-1)
                    elif names[name_idx] == "attention_mask":
                        j[i] = torch.cat((j[i], torch.zeros(1, max_len - j[i].shape[-1], dtype=j[i].dtype)), dim=-1)
                    elif names[name_idx] == "label":
                        j[i] = torch.cat((j[i], torch.ones(1, max_len - j[i].shape[-1], dtype=j[i].dtype)*-100), dim=-1)
                    elif names[name_idx] == "correctness":
                        j[i] = torch.tensor(j[i]).long()

            if isinstance(j[0], torch.Tensor):
                j = torch.cat(j, dim=0)

            merged[names[name_idx]] = j

        return merged


    data = load_dataset("FrozenWolf/prm800k")
    
    train_dataset = QwenMathDataset(data['train'], tokenizer, special_tokens=token_based)

    assert meta_dataset in ["AIME", "PRM800K", "both"], "Meta dataset must be specified as 'AIME', 'PRM800K', or 'both'."

    if meta_dataset == "AIME":
        meta_dataset = load_dataset("FrozenWolf/Gemini-AIME-Meta")['train']
    elif meta_dataset == "PRM800K":
        meta_dataset = data['test']
    elif meta_dataset == "both":
        meta_dataset1 = load_dataset("FrozenWolf/Gemini-AIME-Meta")['train']
        meta_dataset2 = data['test']
        meta_dataset = ConcatDataset([meta_dataset1, meta_dataset2])
    
    
    if not token_based:
        meta_dataset = QwenMathMetaDataset(meta_dataset, tokenizer)
    else:
        meta_dataset =  QwenMathDataset(meta_dataset, tokenizer, has_subjects=False)
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_merge_minibatch)
    next(iter(train_dataloader))  
    meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True, collate_fn=collate_merge_minibatch)
    next(iter(meta_dataloader))  

    paths = "aime_outputs/"
    
    dataloader_benchmark = {}


    for ds in os.listdir(paths):
        dataloader_benchmark[ds] = {}
        ds_ = os.path.join(paths, ds)
        
        for model_out in os.listdir(ds_):
            model_out = os.path.join(ds_, model_out)
            path = os.listdir(model_out)[0]
            path = os.path.join(model_out, path)
            output = pd.read_json(path_or_buf=path, lines=True)
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

            test_ds = HF_Dataset.from_pandas(output)
    
            if not token_based:
                dataset = QwenMathMetaDataset(test_ds, tokenizer, inference=True) 
            else:
                dataset = QwenMathDataset(test_ds, tokenizer, special_tokens=token_based, has_subjects=False, inference=True) 

            dataloader = DataLoader(dataset, batch_size=meta_batch_size, shuffle=False, collate_fn=collate_merge_minibatch)

            dataloader_benchmark[ds][model_out] = dataloader
            next(iter(dataloader))


    return train_dataloader, meta_dataloader, dataloader_benchmark
