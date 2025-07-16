#!/bin/bash

if [[ "$1" == "--get" ]]; then
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '
    {
        mem=$2 + 0; util=$3 + 0;
        score = mem * 2 + util;
        print $1, score;
    }' | sort -k2 -n | head -n1 | awk '{print $1}'
else
    best_gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '
    {
        mem=$2 + 0; util=$3 + 0;
        score = mem * 2 + util;
        print $1, score;
    }' | sort -k2 -n | head -n1 | awk '{print $1}')
    export CUDA_VISIBLE_DEVICES=$best_gpu
    echo "✅ Exported CUDA_VISIBLE_DEVICES=$best_gpu"
fi
