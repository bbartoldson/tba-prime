#!/bin/bash
#
# Single-node launcher for Countdown / MATH experiments.
#
# Usage:
#   ./launch.sh <config_path> [replicate]
#   ./launch.sh kill
#
# Reads the same per-experiment config files as multinode_launch.sh
# (e.g. Countdown_experiments/TBA_qwen3_b005_async10_reset50). Splits the
# visible GPUs between inference and training on this single node.
#
# Override the GPU split with INFER_GPUS / TRAIN_GPUS env vars, e.g.:
#   INFER_GPUS=0,1,2 TRAIN_GPUS=3 ./launch.sh Countdown_experiments/...
#
# NOTE: keep the objective_args composition below in sync with
# multinode_launch.sh.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ "$1" = "kill" ]; then
    echo "Stopping all processes..."
    pkill -f infer.py; pkill -f train.py
    pkill -f spawn_main
    echo "All processes stopped."
    exit 0
fi

CONFIG_PATH=$1
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <config_path> [replicate]" >&2
    exit 1
fi

if [ -n "$2" ]; then
    replicate="$2"
fi

# Default GPU split: half infer, half train.
if [ -z "$INFER_GPUS" ] && [ -z "$TRAIN_GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        n_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        n_gpus=0
    fi
    if [ "$n_gpus" -lt 2 ]; then
        echo "Need at least 2 visible GPUs for single-node infer+train, found $n_gpus." >&2
        echo "Set INFER_GPUS / TRAIN_GPUS env vars to override." >&2
        exit 1
    fi
    half=$(( n_gpus / 2 ))
    INFER_GPUS=$(seq -s, 0 $((half - 1)))
    TRAIN_GPUS=$(seq -s, $half $((n_gpus - 1)))
fi

infer_dp=$(awk -F, '{print NF}' <<<"$INFER_GPUS")
train_nproc=$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")

# Defaults (overridable in the sourced config)
bs=512
eval_steps=5
conf=MATH.toml
project=" --monitor.wandb.project TBA "

echo "Loading configuration from: ${CONFIG_PATH}"
source "${CONFIG_PATH}"

name="${CONFIG_PATH}"
name="${name#refine/}"
name="${name/\//_}"
name="${name/llama_base/llama}"
process_bs=$(( bs / train_nproc ))

if [ -n "${replicate+x}" ]; then
    name="${name}_iter${replicate}"
fi

train_seq_len=""
inf_seq_len=""
if [ -n "$seq_length" ]; then
    train_seq_len=" --data.seq_length $seq_length"
    inf_seq_len=" --model.max_model_len $seq_length"
fi

eval_interval=""
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

if [ -n "$IS_OFF" ]; then
    objective_args+=$IS_OFF
fi

if [ -n "$beta_decay_end" ]; then
    objective_args+=" --grpo.off-policy.beta-decay-end $beta_decay_end"
fi

if [ -n "$final_beta" ]; then
    objective_args+=" --grpo.off-policy.final-beta $final_beta"
fi

echo "Configuration loaded:"
echo "================================"
echo "REPO_DIR=$REPO_DIR"
echo "INFER_GPUS=$INFER_GPUS  (dp=$infer_dp)"
echo "TRAIN_GPUS=$TRAIN_GPUS  (nproc=$train_nproc)"
echo "model=$model"
echo "LR=$LR"
echo "sampled_k=$sampled_k"
echo "async_level=$async_level"
echo "steps=$steps"
echo "name=$name"
echo "type=$type"
echo "objective_args=$objective_args"
echo "process_bs (bs/${train_nproc})=$process_bs"
echo "================================"

cd "${REPO_DIR}"

echo "Starting inference on GPUs ${INFER_GPUS}..."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
CUDA_VISIBLE_DEVICES=${INFER_GPUS} nohup uv run python src/zeroband/infer.py @ configs/inference/${conf} \
    $eval_interval $inf_seq_len \
    --max_batch_size ${process_bs} \
    --model.name ${model} \
    --parallel.dp ${infer_dp} \
    --rl.async-level $async_level \
    --rl.ckpt-path /p/vast1/bartolds/tba-prime/${name}_checkpoints \
    --rollout-path /p/vast1/bartolds/tba-prime/${name}_rollouts \
    --max-steps $steps \
    --sampling.n $sampled_k \
    --monitor.wandb.name infer_${name} \
    $project > inference_${name}.log 2>&1 &

echo "Waiting 5 seconds for inference workers to initialize..."
sleep 5

echo "Starting training on GPUs ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} uv run torchrun --nproc_per_node=${train_nproc} src/zeroband/train.py \
    @ configs/training/${conf} $train_seq_len \
    --optim.batch_size $bs \
    --model.name ${model} \
    --data.num_workers 1 \
    --optim.optim.lr ${LR} \
    --stop-after-steps $steps \
    --max-async-level $async_level \
    --data.path /p/vast1/bartolds/tba-prime/${name}_rollouts \
    --ckpt.rollout-path /p/vast1/bartolds/tba-prime/${name}_checkpoints \
    ${objective_args} \
    --monitor.wandb.name ${name} \
    $project > training_${name}.log 2>&1

echo "Single-node launch complete."
