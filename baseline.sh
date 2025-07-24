#!/bin/bash

{
echo "Running: aime_myprm_benchmark.py | token | Qwen2.5"
python aime_myprm_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: myprm_qwen_benchmark.py | token | Qwen2.5"
python myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: proocessbench_myprm_qwen_benchmark.py | token | Qwen2.5"
python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: aime_myprm_benchmark.py | clf | Qwen2.5"
python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: myprm_qwen_benchmark.py | clf | Qwen2.5"
python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: proocessbench_myprm_qwen_benchmark.py | clf | Qwen2.5"
python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 5
echo "Running: aime_myprm_benchmark.py | token | Qwen3"
python aime_myprm_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
sleep 5
echo "Running: myprm_qwen_benchmark.py | token | Qwen3"
python myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
sleep 5
echo "Running: proocessbench_myprm_qwen_benchmark.py | token | Qwen3"
python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
sleep 5
echo "Running: aime_myprm_benchmark.py | clf | Qwen3"
python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
sleep 5
echo "Running: myprm_qwen_benchmark.py | clf | Qwen3"
python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
sleep 5
echo "Running: proocessbench_myprm_qwen_benchmark.py | clf | Qwen3"
python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
} 2>&1 | tee output_log.txt
