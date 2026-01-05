export CUDA_VISIBLE_DEVICES=1
base_model=/data/whsun/idrr/expt/rl_cold_start/pdtb2/top/qwen3-8b-exp_by_qwen3/epo1/lora_merged
lora_path=/data/whsun/idrr/checkpoints/verl_pdtb/Qwen3-8B-E1_by_qwen3_max-DAPO-lora/global_step_210/actor/lora_adapter
python src/lora_merge.py \
    --base_model $base_model \
    --tokenizer_base $base_model \
    --lora_path $lora_path \
    --output_dir $lora_path/lora_merged