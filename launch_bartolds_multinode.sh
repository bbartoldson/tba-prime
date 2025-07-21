#!/bin/bash

# Launch script for 2-node setup with 4 H100s each
# Node 1: 6 GPUs for inference (1.5 nodes worth)
# Node 2: 2 GPUs for training (0.5 nodes worth)

# Configuration
NODE1_HOSTNAME="matrix17"  # Replace with actual hostname
NODE2_HOSTNAME="matrix16"  # Replace with actual hostname

# Check if kill argument is provided
if [ "$1" = "kill" ]; then
    echo "Stopping all processes..."
    ssh $NODE1_HOSTNAME 'pkill -f infer.py'
    ssh $NODE2_HOSTNAME 'pkill -f infer.py; pkill -f train.py'
    echo "All processes stopped."
    exit 0
fi

echo "Starting multi-node training setup..."
echo "This script assumes you're running it from a system that can SSH to both nodes"
echo "Node 1 (${NODE1_HOSTNAME}): 6 GPUs for inference"
echo "Node 2 (${NODE2_HOSTNAME}): 2 GPUs for training"

# Start inference on Node 1 (uses all 4 GPUs) and Node 2 (uses 2 GPUs)
echo "Starting inference workers..."

# Node 1: Use all 4 GPUs for inference
ssh $NODE1_HOSTNAME << 'EOF'
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
nohup uv run python src/zeroband/infer.py @ configs/inference/math_1b.toml --parallel.dp 4 --max-batch-size 64 > inference_node1.log 2>&1 &
echo "Inference started on Node 1 with 4 GPUs"
EOF

# Node 2: Use 2 GPUs for inference, 2 for training
ssh $NODE2_HOSTNAME << 'EOF'
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536
nohup uv run python src/zeroband/infer.py @ configs/inference/math_1b.toml --parallel.dp 2 --max-batch-size 64 > inference_node2.log 2>&1 &
echo "Inference started on Node 2 with 2 GPUs"
EOF

# Wait a bit for inference to start up
echo "Waiting 30 seconds for inference workers to initialize..."
sleep 30

# Start training on Node 2 (uses remaining 2 GPUs)
echo "Starting training worker..."
ssh $NODE2_HOSTNAME << 'EOF'
cd /usr/workspace/bartolds/tba-prime  # Update this path
export CUDA_VISIBLE_DEVICES=2,3
ulimit -n 65536
nohup uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/math_1b.toml --data.num_workers 2 > training_node2.log 2>&1 &
echo "Training started on Node 2 with 2 GPUs"
EOF

echo "Multi-node setup complete!"
echo ""
echo "To monitor logs:"
echo "  Inference Node 1: ssh $NODE1_HOSTNAME 'tail -f /usr/workspace/bartolds/tba-prime/inference_node1.log'"
echo "  Inference Node 2: ssh $NODE2_HOSTNAME 'tail -f /usr/workspace/bartolds/tba-prime/inference_node2.log'"
echo "  Training Node 2: ssh $NODE2_HOSTNAME 'tail -f /usr/workspace/bartolds/tba-prime/training_node2.log'"
echo ""
echo "To stop all processes:"
echo "  ssh $NODE1_HOSTNAME 'pkill -f infer.py'"
echo "  ssh $NODE2_HOSTNAME 'pkill -f infer.py; pkill -f train.py'"
