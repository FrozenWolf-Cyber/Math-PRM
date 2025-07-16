#!/bin/bash

# Function to extract GPU memory usage and utilization
get_gpu_stats() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

# Get GPU stats and find the best one (lowest memory used + low util)
best_gpu=$(get_gpu_stats | awk -F',' '
{
    mem=$2 + 0; util=$3 + 0;
    score = mem * 2 + util;  # You can tweak this weight
    print $1, score;
}' | sort -k2 -n | head -n1 | awk '{print $1}')

# Export the selected GPU
export CUDA_VISIBLE_DEVICES=$best_gpu
echo "✅ Exported CUDA_VISIBLE_DEVICES=$best_gpu"
