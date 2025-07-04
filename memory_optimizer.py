"""
Enhanced memory optimizer for preventing end-stage inference lag
"""
import torch
import gc
import psutil
import os
from typing import List, Optional, Tuple
import numpy as np

class FrameBufferManager:
    """Manages frame accumulation to prevent memory buildup during long inferences"""
    
    def __init__(self, max_frames_in_memory: int = 100, use_disk_buffer: bool = False, temp_dir: Optional[str] = None):
        self.max_frames_in_memory = max_frames_in_memory
        self.use_disk_buffer = use_disk_buffer
        self.temp_dir = temp_dir or "/tmp"
        self.frames_buffer = []
        self.disk_frame_paths = []
        self.frame_count = 0
        
    def add_frames(self, frames: torch.Tensor) -> None:
        """Add frames to buffer, automatically managing memory"""
        if self.use_disk_buffer and len(self.frames_buffer) >= self.max_frames_in_memory:
            # Write to disk to free memory
            self._flush_to_disk()
        
        # Convert to CPU and add to buffer
        if frames.is_cuda:
            frames = frames.cpu()
        
        self.frames_buffer.append(frames)
        self.frame_count += frames.shape[0]
        
        # Aggressive memory cleanup
        if self.frame_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _flush_to_disk(self) -> None:
        """Write frames to disk to free memory"""
        if not self.frames_buffer:
            return
            
        # Concatenate frames in buffer
        buffer_frames = torch.cat(self.frames_buffer, dim=0)
        
        # Save to disk
        disk_path = os.path.join(self.temp_dir, f"frames_buffer_{len(self.disk_frame_paths)}.pt")
        torch.save(buffer_frames, disk_path)
        self.disk_frame_paths.append(disk_path)
        
        # Clear buffer
        self.frames_buffer.clear()
        del buffer_frames
        gc.collect()
        
    def get_all_frames(self) -> torch.Tensor:
        """Retrieve all frames, loading from disk if needed"""
        all_frames = []
        
        # Load from disk first
        for disk_path in self.disk_frame_paths:
            frames = torch.load(disk_path)
            all_frames.append(frames)
            os.remove(disk_path)  # Clean up
            
        # Add remaining buffer frames
        if self.frames_buffer:
            buffer_frames = torch.cat(self.frames_buffer, dim=0)
            all_frames.append(buffer_frames)
            
        # Concatenate all
        result = torch.cat(all_frames, dim=0) if all_frames else torch.empty(0)
        
        # Cleanup
        self.cleanup()
        
        return result
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        # Remove any remaining disk files
        for disk_path in self.disk_frame_paths:
            if os.path.exists(disk_path):
                os.remove(disk_path)
        
        self.frames_buffer.clear()
        self.disk_frame_paths.clear()
        self.frame_count = 0
        gc.collect()


class InferenceMemoryOptimizer:
    """Optimizes memory during inference to prevent end-stage lag"""
    
    def __init__(self, memory_mode: str = "balanced", enable_disk_cache: bool = False):
        self.memory_mode = memory_mode
        self.enable_disk_cache = enable_disk_cache
        
        # Memory thresholds
        self.memory_thresholds = {
            "aggressive": {"warning": 0.85, "critical": 0.95},
            "balanced": {"warning": 0.75, "critical": 0.85},
            "conservative": {"warning": 0.65, "critical": 0.75}
        }
        
        self.iteration_count = 0
        self.last_cleanup = 0
        
    def should_clear_cache(self, iteration: int) -> bool:
        """Determine if cache should be cleared based on iteration and memory usage"""
        self.iteration_count = iteration
        
        # Get current memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            thresholds = self.memory_thresholds[self.memory_mode]
            
            # Critical threshold - always clear
            if memory_used > thresholds["critical"]:
                return True
                
            # Warning threshold - clear based on iteration
            if memory_used > thresholds["warning"]:
                # More aggressive clearing as we progress
                clear_interval = max(1, 5 - (iteration // 20))
                return (iteration - self.last_cleanup) >= clear_interval
        
        # Default clearing schedule
        if self.memory_mode == "aggressive":
            return iteration % 5 == 0
        elif self.memory_mode == "balanced":
            return iteration % 3 == 0
        else:  # conservative
            return iteration % 2 == 0
    
    def clear_memory(self, aggressive: bool = False) -> None:
        """Clear memory with varying levels of aggressiveness"""
        self.last_cleanup = self.iteration_count
        
        # Basic cleanup
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if aggressive:
                # Force synchronize to ensure all operations complete
                torch.cuda.synchronize()
                
                # Additional aggressive cleanup
                torch.cuda.reset_peak_memory_stats()
                
                # Clear any cached allocator blocks
                torch.cuda.empty_cache()
                
        # System-level garbage collection
        gc.collect(2)  # Full collection
        
    def optimize_inference_iteration(self, iteration: int, total_iterations: int) -> None:
        """Optimize memory for a specific inference iteration"""
        # Progressive optimization - more aggressive as we near the end
        progress = iteration / total_iterations
        
        if progress > 0.8:  # Last 20% of iterations
            # Very aggressive clearing
            self.clear_memory(aggressive=True)
        elif progress > 0.6:  # 60-80% through
            # Moderate clearing
            if self.should_clear_cache(iteration):
                self.clear_memory(aggressive=False)
        else:
            # Normal clearing schedule
            if iteration > 0 and iteration % 3 == 0:
                self.clear_memory(aggressive=False)
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        stats = {
            "iteration": self.iteration_count,
            "mode": self.memory_mode
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,
                "gpu_utilization": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            })
        
        # System memory
        vm = psutil.virtual_memory()
        stats.update({
            "ram_used_gb": vm.used / 1024**3,
            "ram_available_gb": vm.available / 1024**3,
            "ram_percent": vm.percent
        })
        
        return stats


def optimize_end_stage_inference(pipeline, num_iterations: int, memory_mode: str = "balanced", 
                                enable_disk_cache: bool = False) -> None:
    """
    Apply end-stage optimization to prevent lag during long inferences
    
    Args:
        pipeline: The inference pipeline
        num_iterations: Total number of inference iterations
        memory_mode: Memory management mode
        enable_disk_cache: Whether to use disk caching for frames
    """
    # Check if pipeline has the inference loop
    if hasattr(pipeline, '__call__') or hasattr(pipeline, 'forward'):
        # Monkey-patch memory optimization into the pipeline
        original_call = pipeline.__call__ if hasattr(pipeline, '__call__') else pipeline.forward
        memory_optimizer = InferenceMemoryOptimizer(memory_mode, enable_disk_cache)
        
        def optimized_call(*args, **kwargs):
            # Track iterations if possible
            if 'callback' in kwargs:
                original_callback = kwargs['callback']
                def optimized_callback(step, timestep, latents):
                    # Apply memory optimization
                    memory_optimizer.optimize_inference_iteration(step, num_iterations)
                    # Call original callback if exists
                    if original_callback:
                        original_callback(step, timestep, latents)
                kwargs['callback'] = optimized_callback
            
            return original_call(*args, **kwargs)
        
        if hasattr(pipeline, '__call__'):
            pipeline.__call__ = optimized_call
        else:
            pipeline.forward = optimized_call
        
    return pipeline