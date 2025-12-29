set -x
export CUDA_VISIBLE_DEVICES=0

data_path=data/rl/verl/pdtb2/top/test.parquet
save_path=results/rl/verl/pdtb2/top/distill_qwen_1.5b_gen_test.parquet
model_path=checkpoints/verl_pdtb/Distill-Qwen-1.5B-GRPO-weighted_reward/global_step_420/actor/huggingface

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
    # rollout.top_k=50 \
