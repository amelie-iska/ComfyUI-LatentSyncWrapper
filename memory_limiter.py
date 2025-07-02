import os
import torch
import gc

# Ensure allocator config is set before CUDA initialization
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8",
)

import torch
import gc


def limit_gpu_memory(memory_fraction=None):
    """Limit GPU memory usage to a fraction of total VRAM."""
    if torch.cuda.is_available():
        if memory_fraction is None:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 24:
                memory_fraction = 0.95
            elif total_gb >= 16:
                memory_fraction = 0.9
            else:
                memory_fraction = 0.83
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


if __name__ == "__main__":
    limit_gpu_memory()
