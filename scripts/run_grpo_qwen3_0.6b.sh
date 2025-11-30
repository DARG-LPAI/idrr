# by https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh
# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export CUDA_VISIBLE_DEVICES=3,5
export PYTHONPATH=/data/whsun/idrr

use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 8))
infer_ppo_max_token_len=$((1024 * 8))

sp_size=2
gpu_nums=2
# CHECKPOINT_PATH=../pretrained_models/Qwen/Qwen3-0.6B
CHECKPOINT_PATH=expt/rl_cold_start/merged-qwen3-0.6B
project_name='verl_grpo_pdtb'
experiment_name='Qwen3-0.6B-SFT-E1-GRPO-weight'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files=./data/rl/verl/pdtb2/top/sft_rl_train.parquet \
    data.val_files=./data/rl/verl/pdtb2/top/sft_rl_test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2  \
    \
    actor_rollout_ref.model.path=${CHECKPOINT_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${gpu_nums} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 $@