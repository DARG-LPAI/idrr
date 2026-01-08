ulimit -c 0
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/offline_infer.py \
    --data-format alpaca \
    --data-path /public/home/hongy/whsun/idrr_exp/data/alpaca/pdtb2_top_train_clue.json \
    --ckpt /public/home/hongy/pretrained_models/Qwen3-8B \
    --out /public/home/hongy/whsun/idrr_exp/results/clue/pdtb2_top_train_gen.json \
    --gpu_memory_utilization 0.8 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_new_tokens 2048 \
    # --ckpt /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5/merged \

# 如果是 dataset_infos 里面定义的 dataset，可直接用如下脚本运行
# save_name=results/rl_cold_start/pdtb2/top/qwen3-1.7b/epo1/generated_predictions.jsonl

# python /data/whsun/LLaMA-Factory/scripts/vllm_infer.py \
#     --dataset pdtb2_test_rl_cold_start \
#     --model_name_or_path /data/whsun/idrr/expt/rl_cold_start/pdtb2/top/qwen3-1.7b/epo1/lora_merged \
#     --adapter_name_or_path None \
#     --vllm_config {"gpu_memory_utilization":0.9} \
#     --template qwen3 \
#     --cutoff_len 1024 \
#     --max_new_tokens 2048 \
#     --top_k 20 \
#     --batch_size 2048 \
#     --save_name $save_name \

# 评测脚本
# python src/eval.py --data_path $save_name