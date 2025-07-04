"""Memory optimization specifically for 16GB systems - stable and quality-preserving"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import psutil
from typing import Dict, List, Tuple, Optional
import os

class Memory16GBOptimizer:
    """Specialized optimizer to keep memory usage under 16GB while maintaining quality"""
    
    def __init__(self):
        self.system_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.target_vram_gb = 14.0  # Leave 2GB headroom on 16GB cards
        self.critical_vram_gb = 15.0  # Emergency threshold
        
        # Safe operating parameters for 16GB
        self.safe_params = {
            'max_frames_in_memory': 8,  # Process max 8 frames at once
            'vae_batch_size': 2,  # VAE decode 2 frames at a time
            'face_batch_size': 4,  # Process 4 faces at once
            'latent_precision': torch.float16,  # Use fp16 for latents
            'attention_slice_size': 4,  # Slice attention computation
        }
        
    def check_memory_availability(self) -> Dict:
        """Check current memory status"""
        if torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            
            # System RAM
            ram_used = psutil.virtual_memory().percent
            
            return {
                'gpu_allocated_gb': allocated,
                'gpu_free_gb': free,
                'gpu_total_gb': total,
                'ram_used_percent': ram_used,
                'safe_to_proceed': free > 2.0,  # Need at least 2GB free
                'emergency_mode': allocated > self.critical_vram_gb
            }
        return {'safe_to_proceed': True, 'emergency_mode': False}
    
    def calculate_safe_batch_size(self, frame_size: Tuple[int, int], current_memory: Dict) -> int:
        """Calculate safe batch size based on available memory"""
        h, w = frame_size
        
        # Estimate memory per frame (rough calculation)
        # Latent: (h/8) * (w/8) * 4 channels * 2 bytes (fp16)
        latent_size_mb = (h/8) * (w/8) * 4 * 2 / (1024**2)
        
        # Decoded frame: h * w * 3 channels * 2 bytes
        frame_size_mb = h * w * 3 * 2 / (1024**2)
        
        # Total per frame with overhead
        total_per_frame_mb = (latent_size_mb + frame_size_mb) * 3  # 3x overhead for processing
        
        # Available memory in MB
        available_mb = current_memory['gpu_free_gb'] * 1024 - 1024  # Leave 1GB buffer
        
        # Calculate safe batch size
        safe_batch = max(1, int(available_mb / total_per_frame_mb))
        
        # Apply hard limits for 16GB systems
        if current_memory['gpu_total_gb'] <= 16:
            safe_batch = min(safe_batch, self.safe_params['max_frames_in_memory'])
        
        return safe_batch
    
    def optimize_pipeline_for_16gb(self, pipeline):
        """Apply 16GB-specific optimizations to pipeline"""
        
        # 1. Enable memory efficient attention
        if hasattr(pipeline.unet, 'set_attention_slice'):
            pipeline.unet.set_attention_slice(self.safe_params['attention_slice_size'])
            print("✓ Enabled sliced attention for memory efficiency")
        
        # 2. Enable VAE tiling for large images
        if hasattr(pipeline.vae, 'enable_tiling'):
            pipeline.vae.enable_tiling()
            print("✓ Enabled VAE tiling")
        
        # 3. Enable CPU offload for sequential modules
        if hasattr(pipeline, 'enable_sequential_cpu_offload'):
            pipeline.enable_sequential_cpu_offload()
            print("✓ Enabled sequential CPU offload")
        
        # 4. Set optimal dtypes
        pipeline.vae.to(dtype=torch.float16)
        pipeline.unet.to(dtype=torch.float16)
        
        return pipeline
    
    def process_video_memory_safe(self, video_frames: List[np.ndarray], 
                                 process_func, 
                                 audio_features: Optional[torch.Tensor] = None) -> List[np.ndarray]:
        """Process video with strict memory management for 16GB systems"""
        
        total_frames = len(video_frames)
        processed_frames = []
        
        # Initial memory check
        mem_status = self.check_memory_availability()
        if mem_status['emergency_mode']:
            print("⚠️ Emergency mode: Clearing all GPU memory")
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate initial batch size
        if video_frames:
            h, w = video_frames[0].shape[:2]
            batch_size = self.calculate_safe_batch_size((h, w), mem_status)
        else:
            batch_size = self.safe_params['max_frames_in_memory']
        
        print(f"Processing {total_frames} frames with batch size: {batch_size}")
        
        # Process in chunks
        for i in range(0, total_frames, batch_size):
            chunk_end = min(i + batch_size, total_frames)
            chunk = video_frames[i:chunk_end]
            
            # Memory check before processing
            mem_status = self.check_memory_availability()
            if not mem_status['safe_to_proceed']:
                # Emergency: Reduce batch size
                print(f"⚠️ Memory pressure detected. Reducing batch size.")
                batch_size = max(1, batch_size // 2)
                chunk = video_frames[i:i+batch_size]
                chunk_end = i + batch_size
                
                # Force cleanup
                torch.cuda.empty_cache()
                gc.collect()
            
            # Process chunk
            try:
                if audio_features is not None:
                    audio_chunk = audio_features[i:chunk_end] if len(audio_features) > i else audio_features[-1:]
                    processed_chunk = process_func(chunk, audio_chunk)
                else:
                    processed_chunk = process_func(chunk)
                
                processed_frames.extend(processed_chunk)
                
            except torch.cuda.OutOfMemoryError:
                print("❌ OOM Error! Attempting recovery...")
                # Clear everything
                torch.cuda.empty_cache()
                gc.collect()
                
                # Process one frame at a time
                for j, frame in enumerate(chunk):
                    try:
                        if audio_features is not None:
                            audio_frame = audio_features[i+j:i+j+1] if len(audio_features) > i+j else audio_features[-1:]
                            result = process_func([frame], audio_frame)
                        else:
                            result = process_func([frame])
                        processed_frames.extend(result)
                    except Exception as e:
                        print(f"Failed to process frame {i+j}: {e}")
                        # Use previous frame as fallback
                        if processed_frames:
                            processed_frames.append(processed_frames[-1])
                        else:
                            processed_frames.append(frame)
            
            # Cleanup after each chunk
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                
            # Progress
            progress = (chunk_end / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Memory: {mem_status['gpu_allocated_gb']:.1f}GB / {mem_status['gpu_total_gb']:.1f}GB")
        
        return processed_frames


class StreamingVAEDecoder:
    """VAE decoder that streams results to disk to save memory"""
    
    def __init__(self, temp_dir: str, batch_size: int = 2):
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        os.makedirs(temp_dir, exist_ok=True)
        
    def decode_to_disk(self, vae, latents: torch.Tensor, file_prefix: str = "frame") -> List[str]:
        """Decode latents directly to disk files"""
        num_frames = latents.shape[0]
        file_paths = []
        
        # Process in small batches
        for i in range(0, num_frames, self.batch_size):
            batch_end = min(i + self.batch_size, num_frames)
            batch_latents = latents[i:batch_end]
            
            # Decode batch
            with torch.cuda.amp.autocast(enabled=True):
                decoded = vae.decode(batch_latents).sample
            
            # Convert to images and save
            decoded = (decoded + 1) / 2  # [-1, 1] to [0, 1]
            decoded = decoded.clamp(0, 1)
            
            for j, frame in enumerate(decoded):
                frame_idx = i + j
                file_path = os.path.join(self.temp_dir, f"{file_prefix}_{frame_idx:06d}.pt")
                
                # Save as fp16 tensor to disk
                torch.save(frame.half().cpu(), file_path)
                file_paths.append(file_path)
                
                # Free memory immediately
                del frame
            
            del decoded
            torch.cuda.empty_cache()
        
        return file_paths
    
    def load_frame(self, file_path: str) -> torch.Tensor:
        """Load a single frame from disk"""
        return torch.load(file_path, map_location='cpu')


class SmartLatentProcessor:
    """Process latents in a memory-efficient way for 16GB systems"""
    
    def __init__(self):
        self.chunk_size = 4  # Process 4 latents at a time
        
    def process_latents_sequential(self, latents: torch.Tensor, unet, 
                                  timesteps, encoder_hidden_states=None,
                                  guidance_scale=1.0) -> torch.Tensor:
        """Process latents sequentially to save memory"""
        
        batch_size = latents.shape[0]
        processed_latents = []
        
        # Process in chunks
        for i in range(0, batch_size, self.chunk_size):
            chunk_end = min(i + self.chunk_size, batch_size)
            latent_chunk = latents[i:chunk_end]
            
            # Process chunk through all timesteps
            current_latents = latent_chunk
            for t in timesteps:
                # Add encoder states if provided
                if encoder_hidden_states is not None:
                    encoder_chunk = encoder_hidden_states[i:chunk_end]
                else:
                    encoder_chunk = None
                
                # Single step
                with torch.cuda.amp.autocast(enabled=True):
                    noise_pred = unet(current_latents, t, encoder_hidden_states=encoder_chunk).sample
                
                # Apply guidance if needed
                if guidance_scale > 1.0 and encoder_chunk is not None:
                    # Classifier free guidance
                    noise_pred_uncond = unet(current_latents, t).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                
                # Scheduler step (pseudo-code, adjust based on your scheduler)
                current_latents = self.scheduler_step(noise_pred, t, current_latents)
            
            processed_latents.append(current_latents)
            
            # Clear intermediate tensors
            del current_latents, noise_pred
            if i % 8 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate results
        return torch.cat(processed_latents, dim=0)
    
    def scheduler_step(self, noise_pred, timestep, latents):
        """Placeholder for scheduler step - implement based on your scheduler"""
        # This is a simplified example
        alpha = 0.001  # Would come from scheduler
        return latents - alpha * noise_pred


def apply_16gb_optimizations(pipeline, video_length: int, resolution: Tuple[int, int]):
    """Apply all 16GB optimizations to a pipeline"""
    
    optimizer = Memory16GBOptimizer()
    
    # Check if we need special handling
    total_pixels = video_length * resolution[0] * resolution[1]
    needs_optimization = total_pixels > 50 * 512 * 512  # More than 50 512x512 frames
    
    if needs_optimization:
        print(f"📊 Applying 16GB memory optimizations for {video_length} frames at {resolution}")
        
        # Apply pipeline optimizations
        pipeline = optimizer.optimize_pipeline_for_16gb(pipeline)
        
        # Set conservative parameters
        if hasattr(pipeline, 'set_progress_bar_config'):
            pipeline.set_progress_bar_config(disable=False, desc="Processing (16GB Mode)")
        
        # Ensure we're using optimal settings
        if video_length > 100:
            print("📌 Long video detected - enabling streaming mode")
            pipeline._use_streaming = True
            pipeline._streaming_batch_size = 4
        
        # Memory monitoring
        def memory_callback(step, timestep, latents):
            mem = optimizer.check_memory_availability()
            if mem['emergency_mode']:
                print(f"⚠️ Step {step}: High memory usage - {mem['gpu_allocated_gb']:.1f}GB")
                torch.cuda.empty_cache()
        
        pipeline._memory_callback = memory_callback
    
    return pipeline


# Integration function for existing pipeline
def integrate_16gb_optimizations(nodes_instance):
    """Integrate 16GB optimizations into existing LatentSync node"""
    
    # Add optimizer to node instance
    nodes_instance.memory_optimizer_16gb = Memory16GBOptimizer()
    nodes_instance.streaming_decoder = None  # Initialize on demand
    nodes_instance.latent_processor = SmartLatentProcessor()
    
    # Override inference method to use optimizations
    original_inference = nodes_instance.inference
    
    def optimized_inference(*args, **kwargs):
        # Check memory before starting
        mem_check = nodes_instance.memory_optimizer_16gb.check_memory_availability()
        
        if mem_check['gpu_total_gb'] <= 16:
            print("🎯 16GB GPU detected - applying memory optimizations")
            
            # Force conservative settings
            if 'memory_mode' in kwargs:
                kwargs['memory_mode'] = 'conservative'
            if 'output_mode' in kwargs and kwargs.get('output_mode') == 'auto':
                kwargs['output_mode'] = 'video_file'  # Force video file output
            if 'vram_fraction' in kwargs and kwargs['vram_fraction'] == 0.0:
                kwargs['vram_fraction'] = 0.75  # Set safe VRAM limit
            
            # Add streaming decoder if needed
            if nodes_instance.streaming_decoder is None:
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="latentsync_16gb_")
                nodes_instance.streaming_decoder = StreamingVAEDecoder(temp_dir)
        
        return original_inference(*args, **kwargs)
    
    nodes_instance.inference = optimized_inference
    
    return nodes_instance