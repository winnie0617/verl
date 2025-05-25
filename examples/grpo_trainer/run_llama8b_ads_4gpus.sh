set -x

# normalized ppo_mini_batch_size 20 should be divisible by ppo_micro_batch_size_per_gpu 8
# data.train_batch_size: Batch size sampled for one training iteration of different RL algorithms.
# actor_rollout_ref.actor.ppo_mini_batch_size: One sample is split into multiple sub-batches with batch_size=ppo_mini_batch_size for PPO updates. The ppo_mini_batch_size is a global num across all workers/gpus
# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: Similar to gradient accumulation, the micro_batch_size_per_gpu for one forward

# Reference model will be enabled when actor.use_kl_loss or/and algorithm.use_kl_in_reward is/are True.
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: The batch size for one forward pass in the computation of ref_log_prob. The value represent the local num per gpu.
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: Micro batch size per gpu (The batch size for one forward pass) for recalculating log_prob. The value represent the local num per gpu.



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${PWD}/../../../data/rl_dataset_train.parquet \
    data.val_files=${PWD}/../../../data/rl_dataset_val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${PWD}/../../../src/utils/metrics.py\
    custom_reward_function.name=length_reward\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='ads_test' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 \
    ray_init.num_cpus=48 $@

    # null means using all CPUs, which might cause hang if limited in systems like SLURM
    # Got OOM when too many cpus are used, but too slow / hang when too few cpus are used