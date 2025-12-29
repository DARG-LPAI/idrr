export CUDA_VISIBLE_DEVICES=2
lora_path=expt/rl_cold_start/pdtb2/Qwen2.5-7B-Instruct/epo1
python src/lora_merge.py \
    --base_model ../pretrained_models/Qwen/Qwen2.5-7B-Instruct \
    --lora_path $lora_path \
    --tokenizer_base $lora_path \
    --output_dir $lora_path/lora_merged