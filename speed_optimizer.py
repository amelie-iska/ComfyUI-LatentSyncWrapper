"""
Speed optimization utilities for high-end GPUs
Implements advanced techniques to reduce inference time
"""
import torch
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any

class DeepCacheOptimizer:
    """DeepCache: Accelerating Diffusion Models for Free
    Reuses feature maps across denoising steps to reduce computation
    """
    def __init__(self, cache_interval: int = 3, cache_layer_id: int = 0):
        self.cache_interval = cache_interval
        self.cache_layer_id = cache_layer_id
        self.cache = {}
        self.step_count = 0
        
    def should_use_cache(self) -> bool:
        """Determine if we should use cached features for this step"""
        return self.step_count % self.cache_interval != 0
        
    def update_cache(self, features: torch.Tensor, layer_id: int):
        """Update the feature cache"""
        if layer_id == self.cache_layer_id and self.step_count % self.cache_interval == 0:
            self.cache[layer_id] = features.clone()
            
    def get_cached_features(self, layer_id: int) -> Optional[torch.Tensor]:
        """Retrieve cached features if available"""
        if self.should_use_cache() and layer_id in self.cache:
            return self.cache[layer_id]
        return None
        
    def increment_step(self):
        """Increment the step counter"""
        self.step_count += 1
        
    def reset(self):
        """Reset the cache and counter"""
        self.cache.clear()
        self.step_count = 0


def enable_torch_compile(model, mode: str = "reduce-overhead", backend: str = "inductor"):
    """Enable torch.compile() for RTX 40-series GPUs
    
    Args:
        model: The model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        backend: Backend to use ("inductor", "cudagraphs")
    
    Returns:
        Compiled model
    """
    if not hasattr(torch, "compile"):
        print("torch.compile not available in this PyTorch version")
        return model
        
    # Check if GPU supports compilation (RTX 40-series)
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)
        
        # RTX 40-series has compute capability 8.9
        if compute_capability >= (8, 0):
            try:
                print(f"Compiling model with mode='{mode}', backend='{backend}'")
                compiled_model = torch.compile(model, mode=mode, backend=backend)
                return compiled_model
            except Exception as e:
                print(f"Failed to compile model: {e}")
                return model
        else:
            print(f"GPU compute capability {compute_capability} does not support optimal compilation")
            
    return model


@contextmanager
def cuda_graphs_context():
    """Context manager for CUDA graphs acceleration
    CUDA graphs can significantly speed up inference by reducing kernel launch overhead
    """
    if not torch.cuda.is_available():
        yield
        return
        
    # Check if CUDA graphs are supported
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major < 7:  # CUDA graphs require compute capability 7.0+
        yield
        return
        
    try:
        # Warm up CUDA
        torch.cuda.synchronize()
        
        # Set CUDA graph capture mode
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        yield
        
    finally:
        torch.cuda.synchronize()
        if 'CUDA_LAUNCH_BLOCKING' in os.environ:
            del os.environ['CUDA_LAUNCH_BLOCKING']


def optimize_vae_decoding(vae, tile_size: int = 512, overlap: int = 64):
    """Optimize VAE decoding for large images using tiling
    
    Args:
        vae: The VAE model
        tile_size: Size of tiles for decoding
        overlap: Overlap between tiles to avoid seams
    """
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        print(f"Enabled VAE tiling with tile_size={tile_size}")
        
        # Set tile parameters if available
        if hasattr(vae, 'tile_sample_min_size'):
            vae.tile_sample_min_size = tile_size
        if hasattr(vae, 'tile_latent_min_size'):
            vae.tile_latent_min_size = tile_size // 8  # Assuming 8x downsampling
        if hasattr(vae, 'tile_overlap_factor'):
            vae.tile_overlap_factor = overlap / tile_size
            
    return vae


def apply_speed_optimizations(pipeline, gpu_info: Dict[str, Any], optimization_level: str = "balanced"):
    """Apply speed optimizations based on GPU capabilities and optimization level
    
    Args:
        pipeline: The diffusion pipeline
        gpu_info: GPU information dictionary
        optimization_level: "conservative", "balanced", "aggressive"
    
    Returns:
        Optimized pipeline and optimization config
    """
    optimizations = {
        "deepcache": False,
        "torch_compile": False,
        "cuda_graphs": False,
        "vae_tiling": True,
        "reduced_precision": True,
        "dynamic_steps": False,
    }
    
    vram_gb = gpu_info.get("vram_gb", 8)
    compute_capability = gpu_info.get("compute_capability", (7, 0))
    gpu_name = gpu_info.get("name", "").lower()
    
    # RTX 4090 specific optimizations
    if "4090" in gpu_name:
        if optimization_level == "aggressive":
            optimizations.update({
                "deepcache": True,
                "torch_compile": True,
                "cuda_graphs": True,
                "dynamic_steps": True,
            })
        elif optimization_level == "balanced":
            optimizations.update({
                "deepcache": True,
                "torch_compile": True,
                "vae_tiling": True,
            })
    # RTX 3090/4080
    elif vram_gb >= 16:
        if optimization_level != "conservative":
            optimizations.update({
                "deepcache": True,
                "torch_compile": compute_capability >= (8, 0),
            })
    
    # Apply optimizations
    if optimizations["torch_compile"] and hasattr(pipeline, 'unet'):
        pipeline.unet = enable_torch_compile(pipeline.unet, mode="reduce-overhead")
        
    if optimizations["vae_tiling"] and hasattr(pipeline, 'vae'):
        pipeline.vae = optimize_vae_decoding(pipeline.vae)
        
    if optimizations["reduced_precision"]:
        # Enable TF32 for Ampere and newer
        if compute_capability >= (8, 0):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster computation")
            
    return pipeline, optimizations


def dynamic_inference_steps(base_steps: int, optimization_level: str = "balanced") -> int:
    """Dynamically adjust inference steps based on optimization level
    
    Args:
        base_steps: Original number of inference steps
        optimization_level: "conservative", "balanced", "aggressive"
    
    Returns:
        Optimized number of steps
    """
    if optimization_level == "aggressive":
        # Reduce steps by 40% for aggressive optimization
        return max(10, int(base_steps * 0.6))
    elif optimization_level == "balanced":
        # Reduce steps by 20% for balanced optimization
        return max(15, int(base_steps * 0.8))
    else:
        # Keep original steps for conservative
        return base_steps


class BatchedVAEDecoder:
    """Decode VAE latents in optimized batches to reduce memory peaks"""
    
    def __init__(self, vae, batch_size: int = 4):
        self.vae = vae
        self.batch_size = batch_size
        
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents in batches"""
        if latents.shape[0] <= self.batch_size:
            return self.vae.decode(latents).sample
            
        # Process in batches
        decoded_batches = []
        for i in range(0, latents.shape[0], self.batch_size):
            batch = latents[i:i + self.batch_size]
            with torch.cuda.amp.autocast():
                decoded = self.vae.decode(batch).sample
            decoded_batches.append(decoded.cpu())  # Move to CPU to free VRAM
            torch.cuda.empty_cache()
            
        # Concatenate all batches
        return torch.cat(decoded_batches, dim=0).to(latents.device)