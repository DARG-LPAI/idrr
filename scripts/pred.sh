#!/bin/bash

# 配置参数
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export SWANLAB_MODE="disabled"
DATASET_DIR="./data"
WORK="arg2def"
WORK_PATH="${WORK}/pdtb2/llama3/epo5"
DEV_DATASET="pdtb2_dev_${WORK}"
TEST_DATASET="pdtb2_test_${WORK}"
MODEL_PATH="/data/whsun/pretrained_models/Meta-Llama-3.1-8B-Instruct"
CHECKPOINTS_DIR="./expt/${WORK_PATH}"
OUTPUT_ROOT="./results/${WORK_PATH}"
PER_DEVICE_TRAIN_BATCH_SIZE=1
TEMPLATE="llama3"

# 创建输出目录结构
mkdir -p "${OUTPUT_ROOT}/dev" "${OUTPUT_ROOT}/test"

# 阶段1: 在开发集上评估所有checkpoint
declare -A metrics
for checkpoint in "${CHECKPOINTS_DIR}"/checkpoint-*; do
    # 适配器检查点路径存在
    if [[ -d "${checkpoint}" ]]; then
        checkpoint_name=$(basename "${checkpoint}")
        output_dir="${OUTPUT_ROOT}/dev/${checkpoint_name}"
        # 是否已经评估过
        if [ ! -d $output_dir ]; then
            echo "------------------------------------------------------------"
            echo " Evaluating ${checkpoint_name} on ${DEV_DATASET}..."
            echo "------------------------------------------------------------"

            # 运行模型评估
            llamafactory-cli train \
                --stage sft \
                --do_predict \
                --model_name_or_path "${MODEL_PATH}" \
                --adapter_name_or_path "${checkpoint}" \
                --eval_dataset "${DEV_DATASET}" \
                --dataset_dir "${DATASET_DIR}" \
                --template "${TEMPLATE}" \
                --finetuning_type lora \
                --output_dir "${output_dir}" \
                --overwrite_cache \
                --overwrite_output_dir \
                --cutoff_len 1024 \
                --preprocessing_num_workers 16 \
                --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
                --predict_with_generate
   
        fi
        metric_file="${output_dir}/eval_results.json"
        
        # 检查评估结果文件是否存在
        if [[ ! -f "${metric_file}" ]]; then
            # 处理评估结果
            echo "Processing evaluation results..."
            python src/eval.py --data_path "${output_dir}" > "${output_dir}/eval_results.log"
        fi

        if [[ -f "${metric_file}" ]]; then
            # 使用 grep 和 awk 提取 JSON 文件中的 F1 字段值
            metric=$(grep -o '"F1":[^,]*' "${metric_file}" | awk -F ':' '{print $2}' | tr -d ' ')
            # 将提取的值存入 metrics 数组
            metrics["${checkpoint_name}"]=${metric}
            # 打印结果  
            echo "${checkpoint_name} F1 score: ${metric}"
        fi
    fi
done
# 阶段2: 选择最佳checkpoint
# for key in "${!metrics[@]}"; do
#     echo "Key: $key, Value: ${metrics[$key]}"
# done

best_metric=0
best_checkpoint=""
for checkpoint_name in "${!metrics[@]}"; do
    current_metric=${metrics["${checkpoint_name}"]}
    if (( $(awk "BEGIN {print (${current_metric} > ${best_metric})}") )); then
        best_metric=${current_metric}
        best_checkpoint="${CHECKPOINTS_DIR}/${checkpoint_name}"
    fi
done


echo "============================================================"
echo " Best Checkpoint: ${best_checkpoint}"
echo " Best Metric: ${best_metric}"
echo "============================================================"


# 阶段3: 在测试集上评估最佳checkpoint
if [[ -n "${best_checkpoint}" ]]; then
    best_ckpt_name=$(basename "${best_checkpoint}")
    output_dir="${OUTPUT_ROOT}/test/${best_ckpt_name}"
    mkdir -p "${output_dir}"

    echo "------------------------------------------------------------"
    echo " Evaluating best checkpoint on ${TEST_DATASET}..."
    echo "------------------------------------------------------------"

    llamafactory-cli train \
        --stage sft \
        --do_predict \
        --model_name_or_path "${MODEL_PATH}" \
        --adapter_name_or_path "${best_checkpoint}" \
        --eval_dataset "${TEST_DATASET}" \
        --dataset_dir "${DATASET_DIR}" \
        --template "${TEMPLATE}" \
        --finetuning_type lora \
        --output_dir "${output_dir}" \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --predict_with_generate

    # 处理最终测试结果
    echo "Processing final test results..."
    python src/eval.py --data_path "${output_dir}" > "${output_dir}/eval_results.log"
else
    
    echo "Error: No valid checkpoints found!"
    exit 1
fi