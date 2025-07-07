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


def eval(dataset, predictions):

    selected_responses = []
    curr_index = -1
    best_score = -10000
    selected_response = None
    for response, problem_index, pred in zip(dataset['generated_responses'], dataset['index'], predictions):
        if curr_index!= problem_index:
            best_score = -10000
            selected_response = response
            curr_index = problem_index
            if curr_index != -1:
                selected_responses.append({"generated_responses":[selected_response]})
        else:
            if pred > best_score:
                best_score = pred
                selected_response = response


    return infer(selected_responses)


def infer(file_outputs, data_name = "aime", split = "test", data_dir = "./data", start_idx = 0, end_idx = -1, k = 1):

    examples = load_data(data_name, split, data_dir)
    if end_idx == -1:
        end_idx = len(examples)
    examples = examples[start_idx:end_idx]

    pass_at_k_list = []
    k = k
    correct_cnt = 0
    
    for i in tqdm(range(len(examples)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, data_name)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(generated_response, data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        
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
                


    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
        return file_outputs, average_pass_at_k
    else:
        print(f"Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
        return file_outputs, correct_cnt / len(examples)




