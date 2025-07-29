#!/bin/bash

# Check if kill argument is provided
if [ "$1" = "kill" ]; then
    echo "Stopping all processes..."
    pkill -f infer.py; pkill -f train.py
    pkill -f spawn_main
    echo "All processes stopped."
    exit 0
fi

echo "Starting 1-node training setup..."

# Start inference on Node 1 (uses all 4 GPUs) and Node 2 (uses 2 GPUs)
echo "Starting inference workers..."

# Node 1: Use all 4 GPUs for inference
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/TBA_qwen2.5_7b_math.toml > inference.log 2>&1 &
echo "Inference started on Node 1 with 3 GPUs"

# Wait a bit for inference to start up
echo "Waiting 10 seconds for inference workers to initialize..."
sleep 5

# Start training on Node 2 (uses remaining 2 GPUs)
echo "Starting training worker..."
ulimit -n 65536
CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/TBA_qwen2.5_7b_math.toml --data.num_workers 2 > training.log 2>&1 &
echo "Training started on Node 1 with 2 GPUs"

echo "1-node setup complete!"
