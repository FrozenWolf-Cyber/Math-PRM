from datasets import load_dataset, load_from_disk
from prm_data import load_data_custom
from tqdm.auto import tqdm
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


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # get p(label=1)
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res


def evaluate_question_stepwise(model, tokenizer, system_prompt, question, stepwise_solution):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(stepwise_solution) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # print("Conversation String:", conversation_str)
    
    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt"
    ).to(model.device)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        step_rewards = make_step_rewards(outputs[0], token_masks)

    return step_rewards[0]


model_name = "Qwen/Qwen2.5-Math-PRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()


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
        for questions, solutions in tqdm(zip(output['problem'].tolist(), output['completions'].tolist()), total=len(output)):
            step_rewards = evaluate_question_stepwise(
                model=model,
                tokenizer=tokenizer,
                system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
                question=questions,
                stepwise_solution=solutions
            )

            all_problem_pred+=[step_rewards[-1]]
            
        predictions, score = eval(output, all_problem_pred, ds)
        print(f"Model {model_out} on dataset {ds} has score {score}")
        prediction_history[ds][model_out] = predictions
        
        
import pickle
with open("aime_prediction_history.pkl", "wb") as f:
    pickle.dump(prediction_history, f)