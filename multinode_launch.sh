#!/bin/bash

# Resolve repo dir from script location so the launcher works from any clone
# (override with REPO_DIR=/path/to/clone if needed).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

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


# IS_OFF, when set in the experiment config, carries the flag to disable
# importance sampling (e.g. "--no-grpo.off-policy.importance-sample"). Leave
# it unset to keep IS on (the TBA default).
if [ -n "$IS_OFF" ]; then
    objective_args+=$IS_OFF
fi

if [ -n "$beta_decay_end" ]; then
    objective_args+=" --grpo.off-policy.beta-decay-end $beta_decay_end"
fi

if [ -n "$final_beta" ]; then
    objective_args+=" --grpo.off-policy.final-beta $final_beta"
fi

if [ -n "$kl_mean_source" ]; then
    objective_args+=" --grpo.off-policy.kl-mean-source $kl_mean_source"
fi

if [ -n "$kl_per_sample_source" ]; then
    objective_args+=" --grpo.off-policy.kl-per-sample-source $kl_per_sample_source"
fi

if [ -n "$reference_mode" ]; then
    objective_args+=" --grpo.reference-mode $reference_mode"
fi

if [ -n "$ema_alpha" ]; then
    objective_args+=" --grpo.ema-alpha $ema_alpha"
fi

if [ -n "$probe_extra_train_forward" ] && [ "$probe_extra_train_forward" = "true" ]; then
    objective_args+=" --grpo.probe-extra-train-forward"
fi

if [ -n "$kl_approx" ]; then
    objective_args+=" --grpo.kl-approx $kl_approx"
fi

if [ -n "$kl_centering" ] && [ "$kl_centering" = "false" ]; then
    objective_args+=" --no-grpo.kl-centering"
fi

if [ -n "$kl_is" ]; then
    objective_args+=" --grpo.off-policy.kl-is $kl_is"
fi

# --- FP4 emulation knobs (Phase A: quantized-rollout RL) ---
# rollout_quant=nvfp4: QDQ eligible weights in save_ckpt_for_rollout, so the
# sampler serves NVFP4-grid weights on BF16 kernels.
if [ -n "$rollout_quant" ]; then
    objective_args+=" --ckpt.rollout-quant $rollout_quant"
fi

# fake_quant=nvfp4: symmetric arm — trainer forward QDQs its linear weights
# (STE) so trainer and sampler share the quantization grid bit-exactly.
if [ -n "$fake_quant" ]; then
    objective_args+=" --fake-quant-forward $fake_quant"
fi

if [ -n "$four_over_six" ] && [ "$four_over_six" = "true" ]; then
    objective_args+=" --ckpt.rollout-quant-four-over-six"
fi

if [ -n "$quant_skip_last_frac" ]; then
    objective_args+=" --ckpt.rollout-quant-skip-last-frac $quant_skip_last_frac"
fi

# vllm_logprobs=true: use the sampler's own returned logprobs as the IS/KL
# anchor (preserves sampler numerics incl. quantization error) instead of
# recomputing them with trainer numerics.
if [ -n "$vllm_logprobs" ] && [ "$vllm_logprobs" = "true" ]; then
    objective_args+=" --use-vllm-logprobs --no-recompute-logprobs"
fi

# --- Heterogeneous staleness knobs ---
# kl_approx_delta_source=per_rollout: calibrate the approx-KL coefficient with
# each sample's true lag Δ_i instead of the global max_async_level.
if [ -n "$kl_approx_delta_source" ]; then
    objective_args+=" --grpo.kl-approx-delta-source $kl_approx_delta_source"
fi

# staleness_offsets=[1,4,10,32]: per-DP-rank checkpoint lags on the inference
# side, so batches mix rollouts from policies of different ages.
infer_args=""
if [ -n "$staleness_offsets" ]; then
    infer_args+=" --rl.staleness-offsets $staleness_offsets"
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
cd ${REPO_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
nohup uv run --no-sync python src/zeroband/infer.py @ configs/inference/${conf} $eval_interval $inf_seq_len ${infer_args} --max_batch_size ${process_bs} --model.name ${model} --parallel.dp 4 --rl.async-level $async_level   --rl.ckpt-path /p/vast1/bartolds/tba-prime/${name}_checkpoints  --rollout-path /p/vast1/bartolds/tba-prime/${name}_rollouts  --max-steps $steps  --sampling.n $sampled_k  --monitor.wandb.name infer_${name} $project > inference_${name}.log 2>&1 &
echo "Inference started on Node 1 with 4 GPUs"
EOF

# Wait a bit for inference to start up
echo "Waiting 5 seconds for inference workers to initialize..."
sleep 5

# Start training on Node 2 (uses all 4 GPUs)
echo "Starting training worker..."
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
ulimit -n 65536
uv run --no-sync torchrun --nproc_per_node=4 src/zeroband/train.py @ configs/training/${conf} $train_seq_len  --optim.batch_size $bs --model.name ${model} --data.num_workers 1  --optim.optim.lr ${LR}  --stop-after-steps $steps  --max-async-level $async_level  --data.path /p/vast1/bartolds/tba-prime/${name}_rollouts  --ckpt.rollout-path /p/vast1/bartolds/tba-prime/${name}_checkpoints ${objective_args}  --monitor.wandb.name ${name} $project > training_${name}.log 2>&1

echo "Multi-node setup complete!"
