from datasets import load_dataset, load_from_disk
from prm_data import load_data_custom
from tqdm.auto import tqdm
# ds = load_from_disk("../PRM800k_cleaned")
ds = load_data_custom("FrozenWolf/prm800k")
dst = ds['test'].to_pandas()
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel, PeftConfig
import torch, os
import argparse
parser = argparse.ArgumentParser(description="Model Configuration Arguments")
parser.add_argument("--load_path", type=str, default="", help="Path to load model from")
args = parser.parse_args()
SEP_TOKEN = '<PRM_STEP_SCORE>'
def chat_template( question, steps):
        global SEP_TOKEN
        steps = f' {SEP_TOKEN} \n'.join(steps)
        
        prompt = f'''Question:
{question}
Answer:
{steps} {SEP_TOKEN}'''
        
        return prompt


dst['qindex'] = dst.index

def cumulative_lists(lst):
    return [lst[:i+1] for i in range(len(lst))]

# Apply to the completions column and explode
dst['cumulative_completions'] = dst['completions'].apply(cumulative_lists)
dst['cum_labels'] = dst['labels'].apply(cumulative_lists)
dst = dst.explode(['cum_labels', 'cumulative_completions'], ignore_index=True)

dst['completions'] = dst['cumulative_completions']
dst['labels'] = dst['cum_labels']
dst['last_label'] = dst['labels'].apply(lambda x: x[-1] )
dst['qindex'] = dst.index
df_shuffled = dst.sample(frac=1, random_state=42).drop_duplicates(subset='qindex')

# Step 2: Get unique question_number-subject pairs from shuffled df
unique_questions = df_shuffled[['qindex', 'subject', 'last_label']]

# Step 3: Sample n unique questions per subject
n = 50
sampled_questions = (
    unique_questions
    .groupby(['subject','last_label'])
    .sample(n=n, random_state=42)
)

# Step 4: Filter all original rows that belong to the sampled question_numbers
dst_selected = df_shuffled[df_shuffled['qindex'].isin(sampled_questions['qindex'])]
print(dst_selected.last_label.value_counts(), dst_selected.subject.value_counts())



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
    
    if args.load_path=="":
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
        step_sep_id = tokenizer.encode("<extra_0>")[0]
    else:
        step_sep_id = tokenizer.encode("<PRM_STEP_SCORE>")[0]
        conversation_str = chat_template(question, stepwise_solution)
    
    # print("Conversation String:", conversation_str)
    
    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt"
    ).to(model.device)

    
    token_masks = (input_ids == step_sep_id)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        step_rewards = make_step_rewards(outputs[0], token_masks)

    return step_rewards[0]

checkpoint_path = args.load_path
model_name = checkpoint_path
if checkpoint_path=="":
    model_name = "Qwen/Qwen2.5-Math-PRM-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda").eval()

else:
    # Load tokenizer from checkpoint (has added tokens)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load model from checkpoint (this will include any adapter if saved properly)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    print("Loaded model from checkpoint:", checkpoint_path)
    # If PEFT/LoRA was used (check if adapter_config.json and adapter_model.safetensors exist)
    # Then wrap with PEFT model
    if os.path.exists(f"{checkpoint_path}/adapter_config.json"):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("Loaded with LoRA adapter")

    # Move to GPU if needed
    model = model.cuda()
    model.eval()

history = []
all_step_pred, all_step_labels, all_problem_correctness, all_problem_pred = [], [], [], []
for questions, solutions, step_labels, correctness in tqdm(zip(
    dst_selected['prompt'].tolist(),
    dst_selected['completions'].tolist(),
    dst_selected['labels'].tolist(),
    dst_selected['correctness'].tolist()
), total=len(dst_selected)):
    step_rewards = evaluate_question_stepwise(
        model=model,
        tokenizer=tokenizer,
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
        question=questions,
        stepwise_solution=solutions
    )
    history.append({
		'question': questions,
		'solutions': solutions,
		'step_rewards': step_rewards,
		'step_labels': step_labels,
		'correctness': correctness
	})
    step_rewards = [step_rewards[-1]]
    all_step_pred+= [1 if i>=0.5 else 0 for i in step_rewards]
    all_step_labels+= [1 if i else 0 for i in step_labels.tolist()]
    all_problem_correctness+=[correctness]
    a = step_rewards[-1]
    all_problem_pred+=[1 if a>=0.5 else 0]
    
    
from model import binary_classification_metrics
step_metrics = binary_classification_metrics(all_step_pred, all_step_labels)
# problem_metric = binary_classification_metrics(all_problem_pred, all_problem_correctness)

print("Last Step-wise Metrics:", step_metrics)



import os
wandb_name = args.load_path.split("/")[-1]
name = f"qwen_benchmark_history_{wandb_name}.pkl"
import pickle
with open(name, 'wb') as f:
    pickle.dump(history, f)