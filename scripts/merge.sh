export CUDA_VISIBLE_DEVICES=7
python src/lora_merge.py \
    --base_model /data/whsun/pretrained_models/Meta-Llama-3.1-8B-Instruct \
    --lora_path /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5 \
    --tokenizer_base /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5 \
    --output_dir /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5/merged