"""
Dynamic Batch Size Optimizer for LatentSync
Automatically finds optimal batch size based on available VRAM
"""

import torch
import gc
import time
from typing import Tuple, Optional, Dict, Any
import numpy as np


class DynamicBatchOptimizer:
    """Dynamically optimize batch size based on available VRAM"""
    
    def __init__(self, safety_margin: float = 0.9):
        """
        Initialize the optimizer
        
        Args:
            safety_margin: Use only this fraction of available VRAM (0.9 = 90%)
        """
        self.safety_margin = safety_margin
        self.benchmark_results = {}
        self.optimal_batch_sizes = {}
        
    def get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        if not torch.cuda.is_available():
            return 0.0
            
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        free_vram = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
        return free_vram * self.safety_margin
        
    def benchmark_batch_size(self, 
                           model: torch.nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int,
                           num_iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark a specific batch size
        
        Returns:
            Dict with 'success', 'time_per_batch', 'memory_used'
        """
        torch.cuda.empty_cache()
        gc.collect()
        
        # Prepare test input
        test_shape = (batch_size,) + input_shape
        device = next(model.parameters()).device
        
        try:
            # Warmup
            with torch.no_grad():
                test_input = torch.randn(test_shape, device=device, dtype=torch.float16)
                _ = model(test_input)
                torch.cuda.synchronize()
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024**3)
            
            # Time the forward pass
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        _ = model(test_input)
                torch.cuda.synchronize()
            
            end_time = time.time()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Cleanup
            del test_input
            torch.cuda.empty_cache()
            
            return {
                'success': True,
                'time_per_batch': (end_time - start_time) / num_iterations,
                'memory_used': peak_mem - start_mem,
                'batch_size': batch_size
            }
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'success': False,
                'batch_size': batch_size
            }
        except Exception as e:
            print(f"Benchmark error for batch size {batch_size}: {e}")
            return {
                'success': False,
                'batch_size': batch_size,
                'error': str(e)
            }
            
    def find_optimal_batch_size(self,
                               model: torch.nn.Module,
                               input_shape: Tuple[int, ...],
                               min_batch: int = 1,
                               max_batch: int = 32,
                               target_memory_usage: float = 0.8) -> int:
        """
        Find optimal batch size using binary search
        
        Args:
            model: The model to benchmark
            input_shape: Shape of single input (without batch dimension)
            min_batch: Minimum batch size to try
            max_batch: Maximum batch size to try
            target_memory_usage: Target fraction of available VRAM to use
            
        Returns:
            Optimal batch size
        """
        available_vram = self.get_available_vram()
        target_vram = available_vram * target_memory_usage
        
        print(f"Finding optimal batch size... (Available VRAM: {available_vram:.1f}GB)")
        
        # Binary search for optimal batch size
        left, right = min_batch, max_batch
        best_batch = min_batch
        best_result = None
        
        while left <= right:
            mid = (left + right) // 2
            result = self.benchmark_batch_size(model, input_shape, mid)
            
            if result['success']:
                best_batch = mid
                best_result = result
                
                # Check if we're using memory efficiently
                if result['memory_used'] < target_vram * 0.7:
                    # Can go higher
                    left = mid + 1
                else:
                    # Close to target, but check if we can go a bit higher
                    test_higher = self.benchmark_batch_size(model, input_shape, mid + 1)
                    if test_higher['success'] and test_higher['memory_used'] <= target_vram:
                        best_batch = mid + 1
                        best_result = test_higher
                    break
            else:
                # Failed, go lower
                right = mid - 1
                
        print(f"✅ Optimal batch size: {best_batch}")
        if best_result:
            print(f"   Memory usage: {best_result['memory_used']:.2f}GB")
            print(f"   Time per batch: {best_result['time_per_batch']:.3f}s")
            
        return best_batch
        
    def adaptive_batch_size(self,
                           current_batch: int,
                           current_memory: float,
                           target_memory: float,
                           success: bool) -> int:
        """
        Adaptively adjust batch size based on current performance
        
        Args:
            current_batch: Current batch size
            current_memory: Current memory usage in GB
            target_memory: Target memory usage in GB
            success: Whether the current batch completed successfully
            
        Returns:
            New recommended batch size
        """
        if not success:
            # Reduce batch size by 25%
            return max(1, int(current_batch * 0.75))
            
        # Calculate memory efficiency
        memory_ratio = current_memory / target_memory
        
        if memory_ratio < 0.6:
            # Significantly under-utilizing, increase by 50%
            return int(current_batch * 1.5)
        elif memory_ratio < 0.8:
            # Some room to grow, increase by 25%
            return int(current_batch * 1.25)
        elif memory_ratio > 0.95:
            # Too close to limit, reduce by 10%
            return max(1, int(current_batch * 0.9))
        else:
            # Just right
            return current_batch
            
    def get_video_optimal_settings(self,
                                  video_length: int,
                                  resolution: Tuple[int, int],
                                  model_memory_mb: int = 2000) -> Dict[str, Any]:
        """
        Get optimal settings for a specific video
        
        Args:
            video_length: Number of frames
            resolution: (height, width)
            model_memory_mb: Estimated model memory in MB
            
        Returns:
            Dict with optimal settings
        """
        available_vram = self.get_available_vram() * 1024  # Convert to MB
        
        # Estimate memory per frame
        height, width = resolution
        # Rough estimates based on LatentSync architecture
        latent_size = (height // 8) * (width // 8) * 4 * 4  # 4 channels, float32
        frame_memory = latent_size / (1024 * 1024)  # MB
        
        # Add overhead for attention, intermediate activations
        frame_memory *= 3.5  # Empirical multiplier
        
        # Calculate batch size
        available_for_frames = (available_vram - model_memory_mb) * 0.8
        optimal_batch = max(1, int(available_for_frames / frame_memory))
        
        # Calculate number of chunks
        num_chunks = max(1, video_length // optimal_batch)
        if video_length % optimal_batch > optimal_batch * 0.5:
            num_chunks += 1
            
        # Adjust batch size for even distribution
        optimal_batch = max(1, video_length // num_chunks)
        
        return {
            'batch_size': optimal_batch,
            'num_chunks': num_chunks,
            'frames_per_chunk': optimal_batch,
            'estimated_memory_gb': (model_memory_mb + optimal_batch * frame_memory) / 1024,
            'video_length': video_length,
            'resolution': resolution
        }


class SmartBatchProcessor:
    """Smart batch processing with memory monitoring"""
    
    def __init__(self, optimizer: DynamicBatchOptimizer):
        self.optimizer = optimizer
        self.memory_history = []
        self.batch_history = []
        
    def process_with_adaptive_batching(self,
                                      process_func,
                                      items: list,
                                      initial_batch_size: int,
                                      progress_callback=None) -> list:
        """
        Process items with adaptive batch sizing
        
        Args:
            process_func: Function that processes a batch
            items: List of items to process
            initial_batch_size: Starting batch size
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed results
        """
        results = []
        current_batch_size = initial_batch_size
        processed = 0
        total = len(items)
        
        available_vram = self.optimizer.get_available_vram()
        target_memory = available_vram * 0.8
        
        while processed < total:
            # Get current batch
            batch_end = min(processed + current_batch_size, total)
            batch = items[processed:batch_end]
            
            # Monitor memory before processing
            torch.cuda.synchronize()
            before_memory = torch.cuda.memory_allocated() / (1024**3)
            
            try:
                # Process batch
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                # Monitor memory after processing
                torch.cuda.synchronize()
                after_memory = torch.cuda.memory_allocated() / (1024**3)
                memory_used = after_memory - before_memory
                
                # Update batch size for next iteration
                current_batch_size = self.optimizer.adaptive_batch_size(
                    current_batch_size,
                    after_memory,
                    target_memory,
                    success=True
                )
                
                # Record history
                self.memory_history.append(after_memory)
                self.batch_history.append(len(batch))
                
                processed = batch_end
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed / total, 
                                    f"Processed {processed}/{total} | "
                                    f"Batch: {len(batch)} | "
                                    f"VRAM: {after_memory:.1f}GB")
                                    
            except torch.cuda.OutOfMemoryError:
                # Reduce batch size and retry
                torch.cuda.empty_cache()
                gc.collect()
                
                current_batch_size = self.optimizer.adaptive_batch_size(
                    current_batch_size,
                    available_vram,
                    target_memory,
                    success=False
                )
                
                if current_batch_size < 1:
                    raise RuntimeError("Cannot process even single item - out of memory")
                    
                print(f"⚠️  OOM - Reducing batch size to {current_batch_size}")
                continue
                
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.memory_history:
            return {}
            
        return {
            'avg_memory_gb': np.mean(self.memory_history),
            'max_memory_gb': np.max(self.memory_history),
            'min_memory_gb': np.min(self.memory_history),
            'avg_batch_size': np.mean(self.batch_history),
            'total_batches': len(self.batch_history)
        }


# Integration function for LatentSync
def optimize_latentsync_batch_size(pipeline, video_length: int, resolution: Tuple[int, int]) -> Dict[str, Any]:
    """
    Optimize batch size for LatentSync pipeline
    
    Args:
        pipeline: LatentSync pipeline instance
        video_length: Number of frames
        resolution: (height, width)
        
    Returns:
        Optimal settings dictionary
    """
    optimizer = DynamicBatchOptimizer(safety_margin=0.85)
    
    # Quick estimate based on video properties
    settings = optimizer.get_video_optimal_settings(
        video_length=video_length,
        resolution=resolution,
        model_memory_mb=2500  # LatentSync UNet estimate
    )
    
    print(f"\n📊 Optimal settings for {video_length} frames at {resolution[0]}x{resolution[1]}:")
    print(f"   Batch size: {settings['batch_size']} frames")
    print(f"   Number of chunks: {settings['num_chunks']}")
    print(f"   Estimated VRAM: {settings['estimated_memory_gb']:.1f}GB")
    
    return settings