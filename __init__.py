import os

# Configure PyTorch memory settings before any CUDA initialization
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:True",
)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gpu_config_node import GPU_NODE_CLASS_MAPPINGS, GPU_NODE_DISPLAY_NAME_MAPPINGS
from .video_length_adjuster_node import VIDEO_NODE_CLASS_MAPPINGS, VIDEO_NODE_DISPLAY_NAME_MAPPINGS
from .video_loader_node import VIDEO_LOADER_NODE_CLASS_MAPPINGS, VIDEO_LOADER_NODE_DISPLAY_NAME_MAPPINGS
from .copy_video_node import COPY_VIDEO_NODE_CLASS_MAPPINGS, COPY_VIDEO_NODE_DISPLAY_NAME_MAPPINGS

# Try to import ultra speed nodes
try:
    from .nodes_ultra import NODE_CLASS_MAPPINGS_ULTRA, NODE_DISPLAY_NAME_MAPPINGS_ULTRA
    ultra_available = True
except ImportError:
    NODE_CLASS_MAPPINGS_ULTRA = {}
    NODE_DISPLAY_NAME_MAPPINGS_ULTRA = {}
    ultra_available = False

# Try to import auto-optimization nodes
try:
    from .auto_optimize_node import NODE_CLASS_MAPPINGS_AUTO, NODE_DISPLAY_NAME_MAPPINGS_AUTO
    auto_available = True
except ImportError:
    NODE_CLASS_MAPPINGS_AUTO = {}
    NODE_DISPLAY_NAME_MAPPINGS_AUTO = {}
    auto_available = False

# Merge the mappings
NODE_CLASS_MAPPINGS.update(GPU_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GPU_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VIDEO_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VIDEO_LOADER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_LOADER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(COPY_VIDEO_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(COPY_VIDEO_NODE_DISPLAY_NAME_MAPPINGS)

# Add ultra nodes if available
if ultra_available:
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ULTRA)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_ULTRA)
    print("✨ LatentSync Ultra nodes loaded with speed optimizations!")

# Add auto-optimization nodes if available
if auto_available:
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_AUTO)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_AUTO)
    print("🤖 LatentSync Auto-Optimization nodes loaded!")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']