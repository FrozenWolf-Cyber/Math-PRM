import json
from tqdm import tqdm
from aime_utils.parser import *
from aime_utils.data_loader import load_data
from aime_utils.grader import *

from math import comb
import pandas as pd



def parse_list(arg):
    return arg.split(',')



    '''
    '--model_name_or_path', type=str, default="./", help="model dir"
    '--data_name', type=str, default="math", help='identify how to extract answer'
    "--split", default="test", type=str
    "--data_dir", default="./data", type=str
    '--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]"
    '--start_idx', type=int, default=0, help="data[start:end]"
    "--k", type=int, default=1, help="Value of k for pass@k calculation"
    "--seed", default=0, type=int
    '''


def eval(dataset, predictions, ds_name):
    selected_responses = []
    curr_index = 0
    best_score = -10000
    selected_response = None
    for response, problem_index, pred in zip(dataset['completions'], dataset['index'], predictions):
        # print(f"Problem index: {problem_index}, Pred: {pred} Extracted response:",extract_answer('\n\n'.join(response), ds_name))
        if curr_index!= problem_index:
            # print("Inserting new response")
            # print(curr_index)
            selected_response = ['\n\n'.join(selected_response)]
            selected_responses.append({"generated_responses":selected_response})
            best_score = -10000
            selected_response = response
            curr_index = problem_index
        else:
            if pred > best_score:
                best_score = pred
                # print(f"New best score: {best_score} for problem index: {problem_index}")
                selected_response = response

    # print("Inserting new response")
    selected_response = response
    curr_index = problem_index
    selected_response = ['\n\n'.join(selected_response)]
    selected_responses.append({"generated_responses":selected_response})

    answers = [extract_answer('\n\n'.join(res['generated_responses']) if res['generated_responses'] else '', ds_name) for res in selected_responses]
    print(len(selected_responses), answers)
    return infer(selected_responses, split=ds_name, k=1)


def get_majority_voting(list_ans):
    ### get max repeated answer
    answer_count = {}
    for ans in list_ans:
        if ans not in answer_count:
            answer_count[ans] = 0
        answer_count[ans] += 1
    sorted_answers = sorted(answer_count.items(), key=lambda x: x[1], reverse=True)
    if sorted_answers:
        return sorted_answers[0][0]

def infer(file_outputs, data_name = "aime", split = "test", data_dir = "./", start_idx = 0, end_idx = -1, k = 1):
    print(f"Evaluating {data_name} {split} with k={k}")
    examples = load_data(data_name, split, data_dir)
    if end_idx == -1:
        end_idx = len(examples)
    examples = examples[start_idx:end_idx]

    pass_at_k_list = []
    k = k
    correct_cnt = 0
    majorty_voting_is_correct_cnt = 0
    a = []
    woorst_possible_score = 0
    for i in range(len(file_outputs)):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, data_name)
        generated_responses = file_outputs[i]['generated_responses']
        a.append(gt_ans)
        # generated_responses = ['\n'.join(file_outputs[i]['generated_responses'])]
        generated_answers = [extract_answer(generated_response, data_name) for generated_response in generated_responses]
        not_gt = [t!=gt_ans for t in generated_answers]
        if not any(not_gt):
            woorst_possible_score+=1
        majorty_voting = get_majority_voting(generated_answers)
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        majorty_voting_is_correct = check_is_correct(majorty_voting, gt_ans)
        if majorty_voting_is_correct:
            majorty_voting_is_correct_cnt += 1
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        file_outputs[i]['majority_voting'] = majorty_voting
        file_outputs[i]['majority_voting_is_correct'] = majorty_voting_is_correct
        file_outputs[i]['prompt'] = d['problem']
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
                
    # print(a)
    # print(f"Worst possible score: {woorst_possible_score}/{len(examples)} = {woorst_possible_score / len(examples):.4f}")
    print(f"Pass@{len(generated_answers)}:  {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
    print(f"Consensus (majority voting): {majorty_voting_is_correct_cnt}/{len(examples)} = {majorty_voting_is_correct_cnt / len(examples):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
        return file_outputs, average_pass_at_k
    else:
        print(f"Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
        return file_outputs, correct_cnt / len(examples)




