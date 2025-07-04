#!/usr/bin/env python
"""
Fix for PyTorch 2.7.1 CUDA graph compatibility issue

This script sets environment variables to disable CUDA graphs and torch.compile
to prevent the "cudaMallocAsync does not yet support checkPoolLiveAllocations" error.
"""

import os
import sys

def fix_cuda_graph_error():
    """Set environment variables to disable problematic features"""
    
    print("Fixing CUDA graph compatibility issue...")
    
    # Disable CUDA graphs
    os.environ['DISABLE_CUDA_GRAPH'] = '1'
    
    # Disable torch compile
    os.environ['DISABLE_TORCH_COMPILE'] = '1'
    
    # Also set PyTorch-specific flags
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '0'
    
    print("✓ Environment variables set:")
    print("  - DISABLE_CUDA_GRAPH=1")
    print("  - DISABLE_TORCH_COMPILE=1")
    print("  - PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync")
    print("\nThese settings will prevent the CUDA graph error.")
    print("\nTo make this permanent, add these to your ComfyUI launch script:")
    print("export DISABLE_CUDA_GRAPH=1")
    print("export DISABLE_TORCH_COMPILE=1")

if __name__ == "__main__":
    fix_cuda_graph_error()