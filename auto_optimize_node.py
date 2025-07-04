"""
Auto-Optimization Node for LatentSync
Automatically configures optimal settings based on GPU benchmark
"""

import os
import json
from typing import Dict, Tuple, Optional
from .gpu_benchmark import GPUBenchmark, OptimalSettings, benchmark_gpu

class LatentSyncAutoOptimizeNode:
    """
    Node that automatically determines optimal settings based on GPU and content
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimization_goal": (["preview", "balanced", "quality", "auto"], {
                    "default": "auto",
                    "description": "What to optimize for"
                }),
                "video_length_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 300.0,
                    "step": 0.1,
                    "description": "Approximate video length for optimization"
                }),
                "run_benchmark": ("BOOLEAN", {
                    "default": False,
                    "description": "Run GPU benchmark (only needed once)"
                }),
                "use_cached": ("BOOLEAN", {
                    "default": True,
                    "description": "Use cached benchmark results if available"
                }),
            },
            "optional": {
                "force_settings": ("SPEED_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("SPEED_CONFIG", "STRING")
    RETURN_NAMES = ("optimized_settings", "info")
    FUNCTION = "optimize"
    CATEGORY = "LatentSync/Optimization"
    
    def __init__(self):
        self.benchmark = GPUBenchmark()
        self._cached_settings = None
        
    def optimize(self, optimization_goal="auto", video_length_seconds=5.0, 
                run_benchmark=False, use_cached=True, force_settings=None):
        """Determine optimal settings based on GPU and use case"""
        
        # If force_settings provided, use those
        if force_settings is not None:
            return (force_settings, "Using manually configured settings")
            
        # Run benchmark if requested or no cached results
        if run_benchmark or (not use_cached):
            print("🔄 Running GPU benchmark...")
            settings = self.benchmark.run_comprehensive_benchmark(quick_mode=True)
            self._cached_settings = settings
        else:
            # Try to load cached results
            if self._cached_settings is None:
                self._cached_settings = self.benchmark.load_cached_results()
                
            if self._cached_settings is None:
                print("⚠️ No cached benchmark results found. Running quick benchmark...")
                settings = self.benchmark.run_comprehensive_benchmark(quick_mode=True)
                self._cached_settings = settings
            else:
                settings = self._cached_settings
                
        # Determine video length category
        if video_length_seconds < 5:
            length_category = "short"
            length_settings = settings.short_video
        elif video_length_seconds < 20:
            length_category = "medium"
            length_settings = settings.medium_video
        else:
            length_category = "long"
            length_settings = settings.long_video
            
        # Select settings based on goal
        if optimization_goal == "preview":
            selected_settings = settings.preview_mode
        elif optimization_goal == "balanced":
            selected_settings = settings.balanced_mode
        elif optimization_goal == "quality":
            selected_settings = settings.quality_mode
        else:  # auto
            # Use length-based recommendation
            selected_settings = length_settings
            
        # Build config dict
        config = {
            "speed_mode": selected_settings["speed_mode"],
            "quality_preset": selected_settings["quality_preset"],
            "enable_deepcache": selected_settings["enable_deepcache"],
            "batch_size": settings.recommended_batch_size,
            "max_resolution": settings.max_resolution,
            "supports_ultra": settings.supports_ultra,
            "supports_deepcache": settings.supports_deepcache,
        }
        
        # Build info string
        info = f"""
GPU: {self.benchmark.gpu_info['name']} ({self.benchmark.gpu_info['vram_gb']:.1f}GB)
Goal: {optimization_goal.upper()}
Video Length: {video_length_seconds:.1f}s ({length_category})
Settings: {config['speed_mode']}/{config['quality_preset']}
DeepCache: {'ON' if config['enable_deepcache'] else 'OFF'}
Expected Speedup: {selected_settings.get('expected_speedup', 1.0):.1f}x
Batch Size: {config['batch_size']}
Max Resolution: {config['max_resolution'][0]}x{config['max_resolution'][1]}
"""
        
        return (config, info)


class LatentSyncBenchmarkNode:
    """
    Dedicated benchmarking node for detailed GPU testing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_mode": (["quick", "full", "stress_test"], {
                    "default": "quick",
                    "description": "Benchmark thoroughness"
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": False,
                    "description": "Clear previous benchmark results"
                }),
                "test_resolution": (["256x256", "512x512", "768x768", "auto"], {
                    "default": "auto",
                    "description": "Resolution to test"
                }),
                "save_report": ("BOOLEAN", {
                    "default": True,
                    "description": "Save detailed benchmark report"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "SPEED_CONFIG")
    RETURN_NAMES = ("benchmark_report", "optimal_settings")
    FUNCTION = "run_benchmark"
    CATEGORY = "LatentSync/Optimization"
    OUTPUT_NODE = True
    
    def run_benchmark(self, benchmark_mode="quick", clear_cache=False, 
                     test_resolution="auto", save_report=True):
        """Run comprehensive GPU benchmark"""
        
        benchmark = GPUBenchmark()
        
        # Clear cache if requested
        if clear_cache and os.path.exists(benchmark.cache_file):
            os.remove(benchmark.cache_file)
            print("✅ Cleared benchmark cache")
            
        # Run benchmark
        print(f"\n🚀 Starting {benchmark_mode.upper()} benchmark...")
        
        if benchmark_mode == "stress_test":
            # Run multiple iterations to test stability
            results = []
            for i in range(3):
                print(f"\nIteration {i+1}/3")
                settings = benchmark.run_comprehensive_benchmark(quick_mode=False)
                results.append(settings)
            # Use the last result
            optimal_settings = results[-1]
        else:
            optimal_settings = benchmark.run_comprehensive_benchmark(
                quick_mode=(benchmark_mode == "quick")
            )
            
        # Generate detailed report
        report = self._generate_report(benchmark, optimal_settings)
        
        # Save report if requested
        if save_report:
            report_path = os.path.expanduser("~/latentsync_benchmark_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\n📄 Detailed report saved to: {report_path}")
            
        # Convert to config dict
        config = {
            "speed_mode": optimal_settings.balanced_mode["speed_mode"],
            "quality_preset": optimal_settings.balanced_mode["quality_preset"],
            "enable_deepcache": optimal_settings.balanced_mode["enable_deepcache"],
            "batch_size": optimal_settings.recommended_batch_size,
            "max_resolution": optimal_settings.max_resolution,
        }
        
        return (report, config)
        
    def _generate_report(self, benchmark, settings):
        """Generate detailed benchmark report"""
        
        report = []
        report.append("="*60)
        report.append("LATENTSYNC GPU BENCHMARK REPORT")
        report.append("="*60)
        report.append("")
        
        # GPU Info
        report.append("GPU INFORMATION:")
        report.append(f"  Name: {benchmark.gpu_info['name']}")
        report.append(f"  VRAM: {benchmark.gpu_info['vram_gb']:.1f} GB")
        report.append(f"  Compute Capability: {benchmark.gpu_info['compute_capability']}")
        report.append("")
        
        # Benchmark Results
        report.append("BENCHMARK RESULTS:")
        report.append("")
        
        if benchmark.results:
            report.append("Configuration Performance:")
            report.append("-"*60)
            report.append(f"{'Config':<30} {'Time/Frame':<12} {'Speedup':<10} {'Quality':<10}")
            report.append("-"*60)
            
            for result in sorted(benchmark.results, key=lambda x: x.speedup, reverse=True):
                config_str = f"{result.speed_mode}/{result.quality_preset}"
                if result.enable_deepcache:
                    config_str += "+DC"
                    
                report.append(
                    f"{config_str:<30} "
                    f"{result.avg_time_per_frame:.3f}s{'':<7} "
                    f"{result.speedup:.2f}x{'':<6} "
                    f"{result.quality_score*100:.0f}%"
                )
                
        report.append("")
        report.append("OPTIMAL SETTINGS:")
        report.append("")
        
        # Preview Mode
        report.append("Preview/Draft Mode:")
        report.append(f"  Speed Mode: {settings.preview_mode['speed_mode']}")
        report.append(f"  Quality Preset: {settings.preview_mode['quality_preset']}")
        report.append(f"  DeepCache: {settings.preview_mode['enable_deepcache']}")
        report.append(f"  Expected Speedup: {settings.preview_mode['expected_speedup']:.1f}x")
        report.append("")
        
        # Balanced Mode
        report.append("Balanced Performance:")
        report.append(f"  Speed Mode: {settings.balanced_mode['speed_mode']}")
        report.append(f"  Quality Preset: {settings.balanced_mode['quality_preset']}")
        report.append(f"  DeepCache: {settings.balanced_mode['enable_deepcache']}")
        report.append(f"  Expected Speedup: {settings.balanced_mode['expected_speedup']:.1f}x")
        report.append("")
        
        # Quality Mode
        report.append("High Quality:")
        report.append(f"  Speed Mode: {settings.quality_mode['speed_mode']}")
        report.append(f"  Quality Preset: {settings.quality_mode['quality_preset']}")
        report.append(f"  DeepCache: {settings.quality_mode['enable_deepcache']}")
        report.append(f"  Expected Speedup: {settings.quality_mode['expected_speedup']:.1f}x")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append(f"  Supports Ultra Mode: {'Yes' if settings.supports_ultra else 'No'}")
        report.append(f"  Supports DeepCache: {'Yes' if settings.supports_deepcache else 'No'}")
        report.append(f"  Recommended Batch Size: {settings.recommended_batch_size}")
        report.append(f"  Maximum Resolution: {settings.max_resolution[0]}x{settings.max_resolution[1]}")
        
        return "\n".join(report)


# Node mappings
NODE_CLASS_MAPPINGS_AUTO = {
    "LatentSyncAutoOptimize": LatentSyncAutoOptimizeNode,
    "LatentSyncBenchmark": LatentSyncBenchmarkNode,
}

NODE_DISPLAY_NAME_MAPPINGS_AUTO = {
    "LatentSyncAutoOptimize": "⚡ LatentSync Auto-Optimize",
    "LatentSyncBenchmark": "📊 LatentSync GPU Benchmark",
}