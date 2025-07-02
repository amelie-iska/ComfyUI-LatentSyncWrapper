import os
import torch
import gc

# Ensure allocator config is set before CUDA initialization
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8",
)


def limit_gpu_memory(memory_fraction=None):
    """Limit GPU memory usage to a fraction of total VRAM."""
    if torch.cuda.is_available():
        if memory_fraction is None:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 24:
                memory_fraction = 0.75  # Reduced from 0.95
            elif total_gb >= 16:
                memory_fraction = 0.75  # Reduced from 0.9
            else:
                memory_fraction = 0.70  # Reduced from 0.83
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.empty_cache()
        gc.collect()


# Automatically limit memory upon module import
try:
    limit_gpu_memory()
except Exception:
    pass

def clear_cache_periodically():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def log_memory_usage(stage=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"[{stage}] VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free (Total: {total:.1f}GB)")
        
        if allocated > 18:
            print(f"⚠️ WARNING: High VRAM usage!")
            clear_cache_periodically()


if __name__ == "__main__":
    limit_gpu_memory()
