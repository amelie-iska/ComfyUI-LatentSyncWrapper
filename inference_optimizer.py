"""
Inference optimization wrapper for LatentSync to reduce lag and memory usage
"""
import torch
import gc
import os
from contextlib import contextmanager

@contextmanager
def optimized_inference_context(device='cuda'):
    """Context manager for optimized inference with aggressive memory management"""
    
    # Record initial state
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        
        # Set memory allocator to be more aggressive about releasing memory
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'
        
        # Enable deterministic algorithms for consistent memory usage
        torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic mode
        
        # Disable cudnn benchmarking during inference
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        yield
    finally:
        # Aggressive cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            final_memory = torch.cuda.memory_allocated()
            memory_used = (final_memory - initial_memory) / 1024**3
            print(f"Inference used {memory_used:.2f}GB additional memory")


def optimize_inference_pipeline(pipeline):
    """Optimize the inference pipeline for lower memory usage"""
    
    if hasattr(pipeline, 'unet'):
        # Enable memory efficient attention if available
        if hasattr(pipeline.unet, 'set_attn_processor'):
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipeline.unet.set_attn_processor(AttnProcessor2_0())
                print("Enabled memory efficient attention (Flash Attention)")
            except:
                print("Could not enable memory efficient attention")
        
        # Enable gradient checkpointing if available
        if hasattr(pipeline.unet, 'enable_gradient_checkpointing'):
            pipeline.unet.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing for UNet")
    
    if hasattr(pipeline, 'vae'):
        # Enable sliced decoding for VAE
        if hasattr(pipeline.vae, 'enable_slicing'):
            pipeline.vae.enable_slicing()
            print("Enabled VAE slicing")
        
        # Enable tiled decoding for VAE
        if hasattr(pipeline.vae, 'enable_tiling'):
            pipeline.vae.enable_tiling()
            print("Enabled VAE tiling")
    
    # Set the pipeline to use less memory
    if hasattr(pipeline, 'enable_attention_slicing'):
        pipeline.enable_attention_slicing(1)
        print("Enabled attention slicing")
    
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
        print("Enabled VAE slicing via pipeline")
    
    if hasattr(pipeline, 'enable_sequential_cpu_offload'):
        # This moves models to CPU after each step
        pipeline.enable_sequential_cpu_offload()
        print("Enabled sequential CPU offload")
    elif hasattr(pipeline, 'enable_model_cpu_offload'):
        # Alternative CPU offload method
        pipeline.enable_model_cpu_offload()
        print("Enabled model CPU offload")
    
    return pipeline


def process_frames_in_chunks(frames, chunk_size=4):
    """Process frames in smaller chunks to reduce memory usage"""
    total_frames = len(frames) if isinstance(frames, list) else frames.shape[0]
    
    for i in range(0, total_frames, chunk_size):
        end_idx = min(i + chunk_size, total_frames)
        if isinstance(frames, list):
            yield frames[i:end_idx]
        else:
            yield frames[i:end_idx]
        
        # Clear cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reduce_inference_lag(config, args):
    """Reduce lag during inference by optimizing memory and processing"""
    
    # Further reduce batch size if experiencing lag
    if hasattr(args, 'batch_size') and args.batch_size > 4:
        print(f"Reducing batch size from {args.batch_size} to 4 to reduce lag")
        args.batch_size = 4
    
    # Reduce number of frames processed at once
    if hasattr(config, 'data') and hasattr(config.data, 'num_frames'):
        if config.data.num_frames > 8:
            print(f"Reducing num_frames from {config.data.num_frames} to 8")
            config.data.num_frames = 8
    
    # Enable mixed precision if not already
    if hasattr(args, 'use_mixed_precision'):
        args.use_mixed_precision = True
    
    return config, args