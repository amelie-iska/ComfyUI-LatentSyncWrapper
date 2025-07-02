"""
GPU Configuration Node for ComfyUI
Allows users to easily configure and test GPU settings
"""
import torch
from .adaptive_gpu_config import AdaptiveGPUConfig

class GPUConfigNode:
    """Node to configure and display GPU settings"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vram_fraction": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.05,
                    "display": "slider"
                }),
                "batch_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "display": "number"
                }),
                "save_settings": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trigger": ("*",),  # Any input to trigger reconfiguration
            }
        }
    
    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("GPU_CONFIG", "STRING")
    RETURN_NAMES = ("config", "info")
    FUNCTION = "configure"
    
    def configure(self, vram_fraction, batch_size_override, save_settings, trigger=None):
        """Configure GPU settings and return info"""
        
        # Create adaptive config
        custom_vram = vram_fraction if vram_fraction > 0 else None
        gpu_config = AdaptiveGPUConfig(custom_vram_fraction=custom_vram)
        
        # Apply overrides
        if batch_size_override > 0:
            gpu_config.profile.batch_size = batch_size_override
        
        # Save settings if requested
        if save_settings:
            gpu_config.save_preferences(
                batch_size=batch_size_override if batch_size_override > 0 else None,
                vram_fraction=vram_fraction if vram_fraction > 0 else None
            )
        
        # Generate info string
        config_dict = gpu_config.get_config()
        info_lines = [
            f"GPU: {config_dict['gpu_name']} ({config_dict['vram_gb']:.1f}GB)",
            f"Profile: {config_dict['profile_name']}",
            f"VRAM Usage: {config_dict['vram_fraction']:.0%}",
            f"Batch Size: {config_dict['batch_size']}",
            f"Num Frames: {config_dict['num_frames']}",
            f"Mixed Precision: {'Yes' if config_dict['use_mixed_precision'] else 'No'}",
            "",
            "Optimizations:",
        ]
        
        for opt, enabled in config_dict['optimizations'].items():
            info_lines.append(f"  {opt}: {'✓' if enabled else '✗'}")
        
        info_string = "\n".join(info_lines)
        
        return (gpu_config, info_string)


class GPUBenchmarkNode:
    """Node to benchmark GPU performance"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 128,
                    "display": "number"
                }),
                "test_iterations": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("benchmark_results",)
    FUNCTION = "benchmark"
    
    def benchmark(self, test_size, test_iterations):
        """Run a simple GPU benchmark"""
        import time
        
        if not torch.cuda.is_available():
            return ("GPU not available for benchmarking",)
        
        results = []
        results.append(f"GPU Benchmark Results")
        results.append(f"Test Size: {test_size}x{test_size}")
        results.append(f"Iterations: {test_iterations}")
        results.append("")
        
        # Test different data types
        for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            # Create test tensors
            a = torch.randn(test_size, test_size, dtype=dtype, device='cuda')
            b = torch.randn(test_size, test_size, dtype=dtype, device='cuda')
            
            # Warmup
            for _ in range(3):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(test_iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            tflops = (2 * test_size ** 3 * test_iterations) / (elapsed * 1e12)
            
            results.append(f"{dtype_name} Performance: {tflops:.2f} TFLOPS")
            
            # Clean up
            del a, b, c
            torch.cuda.empty_cache()
        
        # Memory bandwidth test
        test_gb = 1  # 1GB test
        test_bytes = test_gb * 1024 * 1024 * 1024
        elements = test_bytes // 4  # float32 elements
        
        a = torch.zeros(elements, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            a.fill_(1.0)
        torch.cuda.synchronize()
        end_time = time.time()
        
        bandwidth = (test_gb * 10) / (end_time - start_time)
        results.append(f"Memory Bandwidth: {bandwidth:.1f} GB/s")
        
        del a
        torch.cuda.empty_cache()
        
        return ("\n".join(results),)


# Node mappings
GPU_NODE_CLASS_MAPPINGS = {
    "GPUConfigNode": GPUConfigNode,
    "GPUBenchmarkNode": GPUBenchmarkNode,
}

GPU_NODE_DISPLAY_NAME_MAPPINGS = {
    "GPUConfigNode": "GPU Configuration",
    "GPUBenchmarkNode": "GPU Benchmark",
}