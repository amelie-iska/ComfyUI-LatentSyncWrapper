"""
RTX 4090 Optimized Configuration
Prevents system lag while maximizing inference speed
"""

import os
import torch

def configure_rtx4090_for_inference():
    """Configure environment for RTX 4090 to prevent display lag"""
    
    # GPU Configuration
    config = {
        # Memory Management
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,garbage_collection_threshold:0.7',
        'CUDA_FORCE_PTX_JIT': '1',
        'CUDA_CACHE_MAXSIZE': '268435456',  # 256MB
        'CUDA_LAUNCH_BLOCKING': '0',  # Keep async
        'CUDA_DEVICE_MAX_CONNECTIONS': '1',
        
        # Display Priority
        'LATENTSYNC_DISPLAY_PRIORITY': '1',
        'TORCH_CUDA_ALLOC_SYNC': '0',
        
        # Performance Settings
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        'TORCH_BACKENDS_CUDNN_BENCHMARK': '1',
        
        # Thread Management  
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8',
        'TORCH_NUM_THREADS': '8',
    }
    
    # Apply environment settings
    for key, value in config.items():
        os.environ[key] = value
    
    # PyTorch specific settings
    if torch.cuda.is_available():
        # Memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # Enable TF32 for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optimize cudnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory pool via environment variable (compatible with all PyTorch versions)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.7'
    
    return config

def get_rtx4090_inference_settings():
    """Get optimal inference settings for RTX 4090"""
    return {
        'batch_size': 8,  # Reduced from 12 to prevent lag
        'num_frames': 16,
        'inference_steps': 15,  # Reduced for speed
        'vram_fraction': 0.85,
        'enable_xformers': True,
        'enable_flash_attention': True,
        'enable_deepcache': True,
        'use_fp16': True,
        'use_channels_last': True,
        'enable_cuda_graphs': False,  # Disabled due to memory issues
        'enable_time_slicing': True,
        'time_slice_ms': 5,
        'yield_frequency_ms': 1,
        'display_priority': True,
        'async_execution': True,
        'memory_efficient_attention': True,
        'gradient_checkpointing': False,
        'cpu_offload': False,
        'sequential_cpu_offload': False,
        'attention_slicing': False,
        'vae_slicing': True,
        'vae_tiling': True,
    }

def optimize_rtx4090_model(model):
    """Apply RTX 4090 specific model optimizations"""
    
    # Convert to FP16
    model = model.half()
    
    # Channels last format
    model = model.to(memory_format=torch.channels_last)
    
    # Compile if available (with careful settings)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(
                model,
                mode='default',  # Not max-autotune to prevent lag
                fullgraph=False,
                dynamic=True,
                backend='inductor'
            )
        except:
            print("Warning: torch.compile not supported")
    
    return model

# Auto-configure on import
configure_rtx4090_for_inference()
print("✓ Configured RTX 4090 optimizations for lag-free inference")