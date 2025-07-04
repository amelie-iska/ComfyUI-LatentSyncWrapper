"""
Speed Control Node for LatentSync
Provides fine-grained control over speed optimizations
"""

import torch
import folder_paths


class SpeedBoostControlNode:
    """Fine-grained control over speed optimizations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["maximum_speed", "balanced", "quality_priority", "custom"], {"default": "balanced"}),
                "turbo_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "cache_aggressiveness": ("INT", {"default": 3, "min": 1, "max": 10}),
                "skip_similar_frames": ("BOOLEAN", {"default": True}),
                "similarity_threshold": ("FLOAT", {"default": 0.95, "min": 0.8, "max": 0.99, "step": 0.01}),
            }
        }
        
    RETURN_TYPES = ("SPEED_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "LatentSync/Optimization"
    
    def create_config(self, mode, turbo_strength, cache_aggressiveness, 
                     skip_similar_frames, similarity_threshold):
        config = {
            "mode": mode,
            "turbo_strength": turbo_strength,
            "cache_aggressiveness": cache_aggressiveness,
            "skip_similar_frames": skip_similar_frames,
            "similarity_threshold": similarity_threshold
        }
        
        # Apply mode presets
        if mode == "maximum_speed":
            config.update({
                "min_steps": 3,
                "max_steps": 10,
                "cache_interval": 2,
                "temporal_weight": 0.8
            })
        elif mode == "quality_priority":
            config.update({
                "min_steps": 15,
                "max_steps": 30,
                "cache_interval": 5,
                "temporal_weight": 0.3
            })
            
        return (config,)


# Update node mappings - Only SpeedBoostControlNode remains
NODE_CLASS_MAPPINGS_ULTRA = {
    "LatentSyncSpeedControl": SpeedBoostControlNode,
}

NODE_DISPLAY_NAME_MAPPINGS_ULTRA = {
    "LatentSyncSpeedControl": "LatentSync Speed Control",
}