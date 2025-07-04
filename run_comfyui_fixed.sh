#!/bin/bash
# Launch ComfyUI with fixes for PyTorch 2.7.1 CUDA graph compatibility

echo "Starting ComfyUI with CUDA graph fixes..."

# Disable CUDA graphs to prevent checkPoolLiveAllocations error
export DISABLE_CUDA_GRAPH=1
export DISABLE_TORCH_COMPILE=1

# Additional PyTorch settings for stability
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export TORCH_CUDNN_V8_API_DISABLED=0

echo "Environment variables set:"
echo "  DISABLE_CUDA_GRAPH=1"
echo "  DISABLE_TORCH_COMPILE=1"
echo "  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
echo ""

# Change to ComfyUI directory
cd /mnt/c/Users/LIZ/Desktop/LATENTSYNC1.6/comfyui

# Activate conda environment
source /home/wolvend/miniconda3/etc/profile.d/conda.sh
conda activate comfyui

# Launch ComfyUI
echo "Launching ComfyUI..."
python main.py --listen --port 8189 --verbose