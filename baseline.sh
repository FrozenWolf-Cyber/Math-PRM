python aime_myprm_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct 
python myprm_qwen_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct 
python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen2.5-Math-7B-Instruct 

python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct 
python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct 
python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen2.5-Math-7B-Instruct 

python aime_myprm_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen3-8B
python myprm_qwen_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen3-8B
python proocessbench_myprm_qwen_benchmark.py --load_path "" --special_tokens  --model_type token --reward_model Qwen/Qwen3-8B

python aime_myprm_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
python myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B
python proocessbench_myprm_qwen_benchmark.py --load_path "" --model_type clf --reward_model Qwen/Qwen3-8B