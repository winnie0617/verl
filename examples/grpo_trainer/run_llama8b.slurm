#!/bin/bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH -p preempt
#SBATCH -A marlowe-m000091
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load necessary modules
module load slurm
module unload nvhpc
module load cudnn/cuda12/9.3.0.75

#Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate ads


cd /users/wychow/ads_summarization/verl/examples/grpo_trainer
echo "Current directory: $(pwd)"

# Set huggingface cache directory
export HF_HOME=/projects/m000091/wychow/huggingface
echo "HF_HOME: ${HF_HOME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${PWD}/../../../data/rl_dataset_train.parquet \
    data.val_files=${PWD}/../../../data/rl_dataset_val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${PWD}/../../../src/utils/metrics.py\
    custom_reward_function.name=cross_entropy_reward\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='ads_test_bs32_rollout_4_cot_simple' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 \
    ray_init.num_cpus=48 $@

# # replace these information with your own
# verl_workdir=/path/to/verl
# train_files=/path/to/gsm8k/train.parquet
# val_files=/path/to/gsm8k/test.parquet
# apptainer_image_path=/path/to/verl-ngc.sif
# # replace these information with your own

# # Getting the node names
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=("$nodes")

# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$head_node_ip" == *" "* ]]; then
# IFS=' ' read -ra ADDR <<<"$head_node_ip"
# if [[ ${#ADDR[0]} -gt 16 ]]; then
#   head_node_ip=${ADDR[1]}
# else
#   head_node_ip=${ADDR[0]}
# fi
# echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
# fi

# port=6379
# ip_head=$head_node_ip:$port
# export ip_head
# echo "IP Head: $ip_head"

# # make sure we set environment variables before Ray initialization
# # If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# # export VLLM_ATTENTION_BACKEND=XFORMERS

# printenv

# echo "Starting HEAD at $head_node"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     apptainer run --nv --bind $verl_workdir $apptainer_image_path \
#         ray start --head --node-ip-address="$head_node_ip" --port=$port \
#         --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# # optional, though may be useful in certain versions of Ray < 1.0.
# sleep 10

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         apptainer run --nv --bind $verl_workdir $apptainer_image_path \
#             ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
#     sleep 5
# done

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=${PWD}/../../../data/rl_dataset_train.parquet \
#     data.val_files=${PWD}/../../../data/rl_dataset_val.parquet \
#     data.train_batch_size=48 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=48 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=48 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.0001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=48 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=48 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     custom_reward_function.path=${PWD}/../../../src/utils/metrics.py\
#     custom_reward_function.name=cross_entropy_reward\
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='verl_grpo_example_gsm8k' \
#     trainer.experiment_name='ads_test' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=15 \
#     ray_init.num_cpus=48 $@