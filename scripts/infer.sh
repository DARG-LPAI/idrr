#!/bin/bash

# 配置参数
export CUDA_VISIBLE_DEVICES=4,5
export WANDB_DISABLED=true
export SWANLAB_MODE="disabled"

DATASET_DIR="./data"
WORK="baseline"
WORK_PATH="pdtb2/top/temporal_aug/llama3-8b/epo5"
DEV_DATASET="pdtb2_dev_${WORK}"
TEST_DATASET="pdtb2_test_${WORK}"
MODEL_PATH="/data/sunwh/pretrained_models/Meta-Llama-3.1-8B-Instruct"
CHECKPOINTS_DIR="./expt/${WORK_PATH}"
OUTPUT_ROOT="./results/${WORK_PATH}"
CKPT_PATH=""
PER_DEVICE_TRAIN_BATCH_SIZE=4

llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path "${MODEL_PATH}" \
    --adapter_name_or_path "${CHECKPOINTS_DIR}/${CKPT_PATH}" \
    --eval_dataset "${TEST_DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template llama3 \
    --finetuning_type lora \
    --output_dir "${OUTPUT_ROOT}/${CKPT_PATH}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --predict_with_generate \
    --fp16

python src/eval.py --data_path "${OUTPUT_ROOT}/${CKPT_PATH}/generated_predictions.jsonl"