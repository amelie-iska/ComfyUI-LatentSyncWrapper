"""
Patches the LipsyncPipeline to prevent system lag during inference
"""

import torch
import time
from functools import wraps

def patch_lipsync_pipeline_for_speed():
    """Monkey patch the pipeline to prevent GPU monopolization"""
    
    try:
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from .ultra_fast_denoising import ChunkedDenoising, fast_inference_mode
        from .gpu_throttle import GPUThrottleManager
        
        # Store original method
        original_call = LipsyncPipeline.__call__
        
        # Create throttle manager
        throttle_manager = GPUThrottleManager()
        chunked_denoising = ChunkedDenoising(yield_every_n_steps=1)
        
        @wraps(original_call)
        def throttled_call(self, *args, **kwargs):
            """Wrapped call that prevents system lag"""
            
            # Enable display priority mode
            with throttle_manager.throttled_compute(priority='low'):
                # Run original inference
                return original_call(self, *args, **kwargs)
        
        # Patch the denoising loop specifically
        def patched_denoising_loop(self, latents, timesteps, audio_embeds, 
                                 mask_latents, masked_image_latents, ref_latents,
                                 do_classifier_free_guidance, guidance_scale,
                                 num_inference_steps, callback=None, callback_steps=1,
                                 extra_step_kwargs=None, generator=None):
            """Optimized denoising loop that prevents lag"""
            
            if extra_step_kwargs is None:
                extra_step_kwargs = {}
                
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            
            with fast_inference_mode():
                for j, t in enumerate(timesteps):
                    # CRITICAL: Yield to display every step on RTX 4090
                    if j > 0:  # Skip first step
                        torch.cuda.synchronize()
                        time.sleep(0.001)  # 1ms yield prevents display freezing
                    
                    if j == 0:
                        print(f"[LatentSync] Denoising with display priority: {num_inference_steps} steps")
                    
                    # Prepare inputs
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    unet_input = self.scheduler.scale_model_input(unet_input, t)
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                    
                    # Use chunked denoising with micro-yields
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        # Add another yield before heavy computation
                        if j % 2 == 0:
                            throttle_manager.yield_to_display()
                            
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                    
                    # Guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                    
                    # Scheduler step with yield
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    
                    # Yield after each step to maintain display responsiveness
                    throttle_manager.yield_to_display()
                    
                    # Callback
                    if callback is not None and j % callback_steps == 0:
                        callback(j, t, latents)
                        
                    # Print sparse progress updates
                    if j % 5 == 0:
                        progress = (j + 1) / len(timesteps) * 100
                        print(f"\r[Denoising] {progress:.1f}%", end="", flush=True)
                        
            print()  # New line after progress
            return latents
        
        # Store patched methods
        LipsyncPipeline.__call__ = throttled_call
        LipsyncPipeline._patched_denoising_loop = patched_denoising_loop
        
        # Also patch the main loop_video method to add yields
        original_loop_video = LipsyncPipeline.loop_video
        
        @wraps(original_loop_video) 
        def throttled_loop_video(self, *args, **kwargs):
            """Add yields in video processing loop"""
            # Yield before starting
            throttle_manager.yield_to_display()
            
            # Call original with periodic yields
            result = original_loop_video(self, *args, **kwargs)
            
            # Yield after completion
            throttle_manager.yield_to_display()
            
            return result
            
        LipsyncPipeline.loop_video = throttled_loop_video
        
        print("✓ Patched LipsyncPipeline for display-friendly inference")
        return True
        
    except Exception as e:
        print(f"Warning: Could not patch pipeline for speed: {e}")
        return False

# Additional optimization for VAE decoding
def patch_vae_for_speed():
    """Patch VAE to prevent lag during decoding"""
    try:
        from diffusers import AutoencoderKL
        
        original_decode = AutoencoderKL.decode
        
        @wraps(original_decode)
        def throttled_decode(self, z, return_dict=True):
            """VAE decode with display yields"""
            # Yield before decode
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                time.sleep(0.0005)  # 0.5ms yield
                
            result = original_decode(self, z, return_dict)
            
            # Yield after decode  
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                time.sleep(0.0005)
                
            return result
            
        AutoencoderKL.decode = throttled_decode
        print("✓ Patched VAE for display-friendly decoding")
        
    except:
        pass

# Auto-patch on import
patch_lipsync_pipeline_for_speed()
patch_vae_for_speed()