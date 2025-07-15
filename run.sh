# python main.py --weights_path "weights" --reward_model "Qwen/Qwen2.5-Math-7B-Instruct" --batch_size 4 --prm_loss --model_type "classifier" --meta_dataset "AIME"

## model -> "Qwen/Qwen2.5-Math-7B-Instruct"
## model_type: token, classifier
## dreamprm_loss -> with and without for model_type=token
## meta_dataset: AIME, PRM800K, both

### SANITY CHECK
export TOKENIZERS_PARALLELISM=false
python main.py \
--weights_path "weights" \
--iteration_num 100000 \
--unroll_steps 5 \
--meta_batch_size 1 \
--train_batch_size 1 \
--reward_model "Qwen/Qwen2-0.5B" \
--model_type "classifier" \
--meta_dataset "both" \
--wandb_mode "offline" \
--freeze_all_but_bias \
--max_step_size 1 --max_meta_steps_grad 10 --filter_dataset_steps 20 --sanity_check 

