export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/offline_infer.py \
    --data-format alpaca \
    --data-path /data/whsun/idrr/data/arg2def/pdtb2/aplaca/test.json \
    --ckpt /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5/merged \
    --out /data/whsun/idrr/result/arg2def/pdtb2/llama3/epo5/merged.vllm.pred.jsonl