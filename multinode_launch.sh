#!/bin/bash

# Get command line arguments
CONFIG_PATH=$1

# Check if second argument exists
if [ -n "$2" ]; then
    # Store the replicate
    replicate="$2"
fi

resolve_hosts_with_srun() {
  srun -N2 -n2 --ntasks-per-node=1  -l hostname \
  | sort -n -k1,1 \
  | awk '{print $2}' \
  | awk '!seen[$0]++' \
  | head -n 2
}

# Check if kill argument is provided
if [ "$1" = "kill" ]; then
    hosts="$(resolve_hosts_with_srun || true)"
    set -- $hosts
    NODE1_HOSTNAME=$1
    NODE2_HOSTNAME=$2
    echo "Stopping all processes..."
    ssh $NODE1_HOSTNAME 'pkill -f infer.py'
    ssh $NODE2_HOSTNAME 'pkill -f train.py'
    sleep 3
    ssh $NODE1_HOSTNAME 'pkill -f spawn_main'
    ssh $NODE2_HOSTNAME 'pkill -f spawn_main'
    echo "All processes stopped."
    exit 0
fi

hosts="$(resolve_hosts_with_srun || true)"
set -- $hosts
CURRENT_HOSTNAME=$(hostname)

# Assign current node to NODE1_HOSTNAME and the other to NODE2_HOSTNAME
if [ "$1" = "$CURRENT_HOSTNAME" ]; then
  NODE1_HOSTNAME=$1
  NODE2_HOSTNAME=$2
elif [ "$2" = "$CURRENT_HOSTNAME" ]; then
  NODE1_HOSTNAME=$2
  NODE2_HOSTNAME=$1
else
    exit 0
fi


bs=512
eval_interval=""
eval_steps=25
eval_steps=5
conf=MATH.toml
project=" --monitor.wandb.project TBA "

# Define the config file path
echo "Loading configuration from: ${CONFIG_PATH}"
# Source the config file to load variables
source "${CONFIG_PATH}"

# Remove "hparams/" prefix
name="${CONFIG_PATH}"
name="${name#refine/}"
name="${name/\//_}"
name="${name/llama_base/llama}"
process_bs=$((bs / 4))

if [ -n "${replicate+x}" ]; then
    name="${name}_iter${replicate}"
fi

train_seq_len=""
inf_seq_len=""
if [ -n "$seq_length" ]; then
    train_seq_len=" --data.seq_length $seq_length"
    inf_seq_len=" --model.max_model_len $seq_length"
fi


if [ -n "$eval_steps" ]; then
    eval_interval+=" --eval.online.interval $eval_steps"
fi

if [[ "$type" == "tb" ]]; then
    objective_args="--grpo.off-policy.type $type --grpo.off-policy.beta $beta  --grpo.off-policy.n $train_k "
elif [[ "$type" == "ratio" ]]; then
    objective_args="--grpo.off-policy.type $type"
elif [[ "$type" == "icepop" ]]; then
    objective_args="--grpo.off-policy.type $type"
else
    echo "Error: Invalid type '$type'. Expected 'tb', 'icepop', or 'ratio'." >&2
    exit 1
fi

if [ -n "$reset" ]; then
    objective_args+=" --grpo.reference-reset-interval $reset"
fi

if [ -n "$reset_opt" ]; then
    objective_args+=" --grpo.reference-reset-opt $reset_opt"
fi


if [ -n "$IS" ]; then
    objective_args+=$IS
fi

if [ -n "$beta_decay_end" ]; then
    objective_args+=" --grpo.off-policy.beta-decay-end $beta_decay_end"
fi

if [ -n "$final_beta" ]; then
    objective_args+=" --grpo.off-policy.final-beta $final_beta"
fi

inference_args=""
if [ -n "$target_length" ]; then
    inference_args=" --rewards.len-reward.reward-type exact --rewards.len-reward.target-lengths $target_length"
fi


  
# Display loaded configuration
echo "Configuration loaded successfully:"
echo "================================"
echo "NODE1_HOSTNAME=$NODE1_HOSTNAME"
echo "NODE2_HOSTNAME=$NODE2_HOSTNAME"
echo "model=$model"
echo "LR=$LR"
echo "sampled_k=$sampled_k"
echo "async_level=$async_level"
echo "steps=$steps"
echo "name=$name"
echo "type=$type"
echo "objective_args=$objective_args"
echo "process_bs (bs/4)=$process_bs"
echo "================================"


echo "Starting multi-node training setup..."
echo "This script assumes you're running it from a system that can SSH to both nodes"
echo "Node 1 (${NODE2_HOSTNAME}): 4 GPUs for inference"
echo "Node 2 (${NODE1_HOSTNAME}): 4 GPUs for training"

echo "Starting inference workers..."

# Node 1: Use all 4 GPUs for inference
ssh $NODE2_HOSTNAME << EOF
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
nohup uv run python src/zeroband/infer.py @ configs/inference/${conf} $eval_interval $inf_seq_len --max_batch_size ${process_bs} --model.name ${model} --parallel.dp 4 --rl.async-level $async_level   --rl.ckpt-path /p/vast1/bartolds/tba-prime/${name}_checkpoints  --rollout-path /p/vast1/bartolds/tba-prime/${name}_rollouts  --max-steps $steps  --sampling.n $sampled_k  --monitor.wandb.name infer_${name} $project $inference_args > inference_${name}.log 2>&1 &
echo "Inference started on Node 1 with 4 GPUs"
EOF

# Wait a bit for inference to start up
echo "Waiting 5 seconds for inference workers to initialize..."
sleep 5

# Start training on Node 2 (uses all 4 GPUs)
echo "Starting training worker..."
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1,2,3
ulimit -n 65536
uv run torchrun --nproc_per_node=4 src/zeroband/train.py @ configs/training/${conf} $train_seq_len  --optim.batch_size $bs --model.name ${model} --data.num_workers 1  --optim.optim.lr ${LR}  --stop-after-steps $steps  --max-async-level $async_level  --data.path /p/vast1/bartolds/tba-prime/${name}_rollouts  --ckpt.rollout-path /p/vast1/bartolds/tba-prime/${name}_checkpoints ${objective_args}  --monitor.wandb.name ${name} $project > training_${name}.log 2>&1

echo "Multi-node setup complete!"
