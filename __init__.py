import os

# Configure PyTorch memory settings before any CUDA initialization
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8",
)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gpu_config_node import GPU_NODE_CLASS_MAPPINGS, GPU_NODE_DISPLAY_NAME_MAPPINGS

# Merge the mappings
NODE_CLASS_MAPPINGS.update(GPU_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GPU_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']