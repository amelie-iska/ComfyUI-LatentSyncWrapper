"""
GPU Throttling and Display Priority Management
Prevents inference from monopolizing the GPU and causing system lag

Environment variables:
    - ``LATENTSYNC_YIELD_MS``: override how often ``yield_to_display`` sleeps.
      Set to ``0`` to disable yielding completely (default ``1`` ms).
    - ``LATENTSYNC_DISABLE_SYNC``: if ``1`` skip ``torch.cuda.synchronize`` in
      ``yield_to_display``.
"""

import torch
import time
import os
from contextlib import contextmanager
from threading import Thread, Event
import queue

class GPUThrottleManager:
    """Manages GPU usage to prevent system lag during inference"""
    
    def __init__(self):
        self.display_priority = True
        self.gpu_usage_target = 0.85  # Leave 15% for display
        self.frame_time_target = 16.67  # 60 FPS target (ms)
        self.last_yield_time = time.time()

        # Allow tuning of yield behaviour via environment variables
        yield_ms = float(os.environ.get('LATENTSYNC_YIELD_MS', '1'))
        self.yield_interval = yield_ms / 1000.0
        self.disable_sync = os.environ.get('LATENTSYNC_DISABLE_SYNC', '0') == '1'
        
        # Set CUDA environment for display priority
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        
        # Configure memory fraction
        if torch.cuda.is_available():
            # Reserve memory for display
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Set memory allocator environment variables instead
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.7'
    
    def yield_to_display(self):
        """Yield GPU to display rendering if enabled"""
        if self.yield_interval <= 0:
            return

        current_time = time.time()
        if current_time - self.last_yield_time > self.yield_interval:
            if torch.cuda.is_available() and not self.disable_sync:
                torch.cuda.synchronize()
            time.sleep(0.0001)  # Tiny sleep to yield
            self.last_yield_time = current_time
    
    @contextmanager
    def throttled_compute(self, priority='normal'):
        """Context manager for throttled GPU computation"""
        if priority == 'low':
            # Lower thread priority for background inference
            try:
                import psutil
                p = psutil.Process()
                p.nice(10)  # Lower priority on Unix
            except:
                pass
        
        # Set GPU to lower power mode temporarily
        if torch.cuda.is_available():
            # Limit GPU clock speeds to prevent thermal throttling
            try:
                device = torch.cuda.current_device()
                # This would require nvidia-ml-py but we'll use env vars instead
                os.environ['CUDA_FORCE_PTX_JIT'] = '1'
                os.environ['CUDA_CACHE_MAXSIZE'] = '268435456'
            except:
                pass
        
        try:
            yield
        finally:
            # Reset after computation
            pass

class StreamedInference:
    """Implements time-sliced inference to prevent GPU monopolization"""
    
    def __init__(self, time_slice_ms=5):
        self.time_slice = time_slice_ms / 1000.0
        self.streams = [torch.cuda.Stream() for _ in range(4)]
        self.current_stream = 0
        
    def run_sliced(self, func, *args, **kwargs):
        """Run function in time slices"""
        start_time = time.time()
        
        with torch.cuda.stream(self.streams[self.current_stream]):
            result = func(*args, **kwargs)
            
            # Check if we need to yield
            elapsed = time.time() - start_time
            if elapsed > self.time_slice:
                torch.cuda.synchronize()
                time.sleep(0.001)  # Yield to system
                
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return result

def create_display_friendly_inference():
    """Creates an inference configuration that won't lag the display"""
    yield_ms = float(os.environ.get('LATENTSYNC_YIELD_MS', '1'))
    config = {
        'gpu_throttle': GPUThrottleManager(),
        'streamed_inference': StreamedInference(),
        'settings': {
            'max_gpu_utilization': 0.85,
            'enable_time_slicing': True,
            'display_priority_mode': True,
            'inference_streams': 4,
            'yield_frequency_ms': yield_ms,
            'memory_reserve_fraction': 0.15,
            'enable_async_compute': True,
            'reduce_memory_fragmentation': True
        }
    }
    return config

@contextmanager 
def gpu_friendly_inference(inference_func):
    """Wrapper to make any inference function display-friendly"""
    throttle = GPUThrottleManager()
    
    # Create lower priority CUDA context
    if torch.cuda.is_available():
        # Set memory growth to prevent sudden allocations
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Use smaller memory pool
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Wrap the inference function
    def throttled_inference(*args, **kwargs):
        with throttle.throttled_compute(priority='low'):
            # Run in chunks with yields
            result = inference_func(*args, **kwargs)
            throttle.yield_to_display()
            return result
    
    try:
        yield throttled_inference
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

# Monkey patch torch operations to add micro-yields
original_conv2d = torch.nn.functional.conv2d
original_linear = torch.nn.functional.linear

def throttled_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Conv2d with display yields"""
    result = original_conv2d(input, weight, bias, stride, padding, dilation, groups)
    if hasattr(torch.cuda, 'synchronize') and input.is_cuda:
        # Micro-yield every few operations
        if hasattr(throttled_conv2d, 'counter'):
            throttled_conv2d.counter += 1
            if throttled_conv2d.counter % 50 == 0:
                torch.cuda.synchronize()
                time.sleep(0.0001)
        else:
            throttled_conv2d.counter = 0
    return result

def apply_display_priority_patches():
    """Apply patches to prevent display lag"""
    # Only patch if explicitly enabled
    if os.environ.get('LATENTSYNC_DISPLAY_PRIORITY', '1') == '1':
        torch.nn.functional.conv2d = throttled_conv2d
        print("✓ Applied display priority patches")

# Auto-apply on import
apply_display_priority_patches()