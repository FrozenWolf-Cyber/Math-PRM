#!/bin/bash

{

	# qwen2.5-7B_clf_acc8_baseline_newtoken_deep-cloud-246
echo "Running: qwen2.5-7B_clf_acc8_baseline_newtoken_deep-cloud-246"
python aime_myprm_benchmark.py --load_path weights/deep-cloud-246 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python myprm_qwen_benchmark.py --load_path weights/deep-cloud-246 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python proocessbench_myprm_qwen_benchmark.py --load_path weights/deep-cloud-246 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2



# qwen2.5-7B_clf_acc8_scratch_newtoken_whole-sky-243
echo "Running: qwen2.5-7B_clf_acc8_scratch_newtoken_whole-sky-243"
python aime_myprm_benchmark.py --load_path weights/whole-sky-243 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python myprm_qwen_benchmark.py --load_path weights/whole-sky-243 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python proocessbench_myprm_qwen_benchmark.py --load_path weights/whole-sky-243 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2


# qwen2.5-7B_token_acc1_scratch_newtoken_leafy-morning-242
echo "Running: qwen2.5-7B_token_acc1_scratch_newtoken_leafy-morning-242"
python aime_myprm_benchmark.py --load_path weights/leafy-morning-242 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python myprm_qwen_benchmark.py --load_path weights/leafy-morning-242 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python proocessbench_myprm_qwen_benchmark.py --load_path weights/leafy-morning-242 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2


# qwen2.5-7B_clf_acc8_baseline_newtoken_silver-dawn-238
echo "Running: qwen2.5-7B_clf_acc8_baseline_newtoken_silver-dawn-238"
python aime_myprm_benchmark.py --load_path weights/silver-dawn-238 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python myprm_qwen_benchmark.py --load_path weights/silver-dawn-238 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2
python proocessbench_myprm_qwen_benchmark.py --load_path weights/silver-dawn-238 --peft_rank 16 --lora_dropout 0 --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct
sleep 2

	# qwen2.5-7B_token_acc16_baseline_lr1e-4_dropout0
# echo "Running: qwen2.5-7B_token_acc16_baseline_lr1e-4_dropout0"
# python aime_myprm_benchmark.py --load_path weights/elated-cosmos-198 --peft_rank 16 --lora_dropout 0 --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/elated-cosmos-198 --peft_rank 16 --lora_dropout 0 --special_tokens  --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/elated-cosmos-198 --peft_rank 16 --lora_dropout 0 --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2

# # qwen2.5-7B_clf_acc8_baseline_lr1e-4_dropout0
# echo "Running: qwen2.5-7B_clf_acc8_baseline_lr1e-4_dropout0"
# python aime_myprm_benchmark.py --load_path weights/tough-moon-199 --peft_rank 16 --lora_dropout 0  --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/tough-moon-199 --peft_rank 16 --lora_dropout 0 --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/tough-moon-199 --peft_rank 16 --lora_dropout 0  --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2

# echo "Running: token | Qwen2.5"
# python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5
# python myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5
# python aime_myprm_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5

# echo "Running: clf | Qwen2.5"
# python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5
# python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5
# python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct
# sleep 5

# echo "Running: baseline"
# python aime_qwen_benchmark.py
# python prm_qwen_benchmark.py
# python proocessbench_qwen_benchmark.py


# pip install --upgrade transformers

# echo "Running: token | Qwen3"
# python aime_myprm_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
# sleep 5
# python myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
# sleep 5
# python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens --model_type token --reward_model Qwen/Qwen3-8B
# sleep 5

# echo "Running: clf | Qwen3"
# python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
# sleep 5
# python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
# sleep 5
# python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B

# # qwen2.5-7B_clf_acc8_baseline_lr1e-4_dropout0.05
# echo "Running: qwen2.5-7B_clf_acc8_baseline_lr1e-4_dropout0.05"
# python aime_myprm_benchmark.py --load_path weights/dulcet-smoke-187 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/dulcet-smoke-187 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/dulcet-smoke-187 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2

# # qwen2.5-7B_token_acc16_baseline_lr1e-4_dropout0.05
# echo "Running: qwen2.5-7B_token_acc16_baseline_lr1e-4_dropout0.05"
# python aime_myprm_benchmark.py --load_path weights/charmed-breeze-185 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/charmed-breeze-185 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/charmed-breeze-185 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct --freeze_all_but_bias
# sleep 2



# # qwen2.5-1.5B_clf_acc4_filtered_steps20_tok1100_grad10
# echo "Running: qwen2.5-1.5B_clf_acc4_filtered_steps20_tok1100_grad10"
# python aime_myprm_benchmark.py --load_path weights/glowing-terrain-166 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/glowing-terrain-166 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/glowing-terrain-166 --peft_rank 16 --lora_dropout 0.05 --special_tokens --add_new_token --model_type classifier --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2

# # qwen2.5-1.5B_token_acc6_baseline
# echo "Running: qwen2.5-1.5B_token_acc6_baseline"
# python aime_myprm_benchmark.py --load_path weights/lucky-haze-180 --peft_rank -1 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2
# python myprm_qwen_benchmark.py --load_path weights/lucky-haze-180 --peft_rank -1 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2
# python proocessbench_myprm_qwen_benchmark.py --load_path weights/lucky-haze-180 --peft_rank -1 --lora_dropout 0.05 --special_tokens --add_new_token --model_type token --reward_model Qwen/Qwen2.5-Math-1.5B-Instruct --freeze_all_but_bias
# sleep 2


} 2>&1 | tee baselines_output_log.txt
