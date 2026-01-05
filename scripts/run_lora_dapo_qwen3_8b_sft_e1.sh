#!/usr/bin/env bash
# lora settings Ref: https://verl.readthedocs.io/en/v0.5.x/advance/ppo_lora.html
# perf tuning guide Ref: https://verl.readthedocs.io/en/v0.5.x/perf/perf_tuning.html

# set -eux pipefail

# for shuguang
ulimit -c 0
export VLLM_USE_V1=0
export SWANLAB_MODE=offline
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

# 显存相关
MODEL_PATH=expt/rl_cold_start/pdtb2/top/qwen3-8b-exp_by_qwen3/epo1/lora_merged
train_prompt_bsz=32 # 算法指标
train_prompt_mini_bsz=8 # 算法指标
vllm_memory_utilization=0.5
n_resp_per_prompt=16

max_prompt_length=512
max_response_length=256
max_seq_len=$((max_prompt_length + max_response_length))
enable_overlong_buffer=False
overlong_buffer_len=256
overlong_penalty_factor=1.0

n_gpus_per_node=2
sp_size=2
gen_tp=2

# TODO: 如何调整actor_ppo_max_token_len、infer_ppo_max_token_len和actor_rollout_ref.rollout.max_num_batched_tokens 才能最大化vLLM显存利用率
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_seq_len * 2)) # >= 2*max_seq_len
infer_ppo_max_token_len=$((max_seq_len * 4)) # >= 2*max_seq_len
offload=True


# RL算法相关
adv_estimator=grpo
use_kl_in_reward=False # True

kl_coef=0.0 # 0.001
use_kl_loss=False # True
kl_loss_coef=0.0 # 0.2

clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

# rollout相关
temperature=1.0 # 0.7
top_p=1.0
top_k=-1

val_top_p=0.95
val_temperature=0.6

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
gen_prompt_bsz=$((train_prompt_bsz * 4))

NOW=$(date +%Y%m%d_%H%M)
project_name='verl_pdtb'
exp_name='Qwen3-8B-E1_by_qwen3_max-DAPO-lora'
log_name="${exp_name}-${NOW}.log"
CKPTS_DIR=${CKPTS_DIR:-"checkpoints/${project_name}/${exp_name}"}

# actor_rollout_ref.rollout.enforce_eager=True \
python3 -m recipe.dapo.main_dapo \
    data.train_files="data/rl/verl/pdtb2/top/sft_rl_train.parquet" \
    data.val_files="data/rl/verl/pdtb2/top/sft_rl_test.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${offload} \
    \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_memory_utilization} \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_model_len=${max_seq_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_seq_len * 16)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    \
    critic.strategy=fsdp2 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.strategy=fsdp2 \
    \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    trainer.logger='swanlab' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.val_before_train=False \
    trainer.test_freq=2 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto $@ 2>&1 | tee dapo_${NOW}.log