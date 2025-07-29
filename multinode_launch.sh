#!/bin/bash

# Launch script for 2-node setup with 4 H100s each
# Node 1: 6 GPUs for inference (1.5 nodes worth)
# Node 2: 2 GPUs for training (0.5 nodes worth)

# Job 1
#export HF_HOME=/p/vast1/bartolds/tba-prime/hf_cache_job1
#export MASTER_PORT=29500

# Configuration
NODE1_HOSTNAME="matrix21"  # Replace with actual hostname
NODE2_HOSTNAME="matrix22"  # Replace with actual hostname

# Check if kill argument is provided
if [ "$1" = "kill" ]; then
    echo "Stopping all processes..."
    ssh $NODE1_HOSTNAME 'pkill -f infer.py'
    ssh $NODE2_HOSTNAME 'pkill -f train.py'
    sleep 3
    ssh $NODE1_HOSTNAME 'pkill -f spawn_main'
    ssh $NODE2_HOSTNAME 'pkill -f spawn_main'
    echo "All processes stopped."
    exit 0
fi

echo "Starting multi-node training setup..."
echo "This script assumes you're running it from a system that can SSH to both nodes"
echo "Node 1 (${NODE1_HOSTNAME}): 4 GPUs for inference"
echo "Node 2 (${NODE2_HOSTNAME}): 4 GPUs for training"

echo "Starting inference workers..."

# Node 1: Use all 4 GPUs for inference
ssh $NODE1_HOSTNAME << 'EOF'
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
nohup uv run python src/zeroband/infer.py @ configs/inference/TBA_qwen2.5_7b_math.toml --parallel.dp 4 > inference_TBA.log 2>&1 &
echo "Inference started on Node 1 with 4 GPUs"
EOF

# Wait a bit for inference to start up
echo "Waiting 5 seconds for inference workers to initialize..."
sleep 5

# Start training on Node 2 (uses all 4 GPUs)
echo "Starting training worker..."
ssh $NODE2_HOSTNAME << 'EOF'
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1,2,3
ulimit -n 65536
nohup uv run torchrun --nproc_per_node=4 src/zeroband/train.py @ configs/training/TBA_qwen2.5_7b_math.toml --data.num_workers 1 > training_TBA.log 2>&1 &
echo "Training started on Node 2 with 4 GPUs"
EOF

echo "Multi-node setup complete!"
echo ""
echo "To monitor logs:"
echo "  Inference Node 1: tail -f /usr/workspace/bartolds/tba-prime/inference.log"
echo "  Training Node 2: tail -f /usr/workspace/bartolds/tba-prime/training.log"
