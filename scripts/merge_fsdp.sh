export CUDA_VISIBLE_DEVICES=0

python src/model_merger.py \
    --local_dir checkpoints/verl_pdtb/Distill-Qwen-1.5B-GRPO-weighted_reward/global_step_420/actor
