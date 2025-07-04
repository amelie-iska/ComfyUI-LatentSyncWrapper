"""
GPU Benchmarking and Auto-Configuration for LatentSync
Automatically determines optimal speed settings based on hardware
"""

import torch
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
import psutil
import GPUtil

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    speed_mode: str
    quality_preset: str
    enable_deepcache: bool
    inference_steps: int
    avg_time_per_frame: float
    total_time: float
    memory_used_gb: float
    quality_score: float
    speedup: float
    
@dataclass
class OptimalSettings:
    """Recommended settings for different use cases"""
    # For maximum speed (previews)
    preview_mode: Dict[str, any]
    # For balanced performance
    balanced_mode: Dict[str, any]
    # For high quality
    quality_mode: Dict[str, any]
    # For specific video lengths
    short_video: Dict[str, any]  # < 5 seconds
    medium_video: Dict[str, any]  # 5-20 seconds
    long_video: Dict[str, any]   # > 20 seconds
    
    # GPU capabilities
    supports_ultra: bool
    supports_deepcache: bool
    recommended_batch_size: int
    max_resolution: Tuple[int, int]

class GPUBenchmark:
    """Comprehensive GPU benchmarking for LatentSync"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_info = self._get_gpu_info()
        self.cache_file = os.path.expanduser("~/.latentsync_benchmark_cache.json")
        self.results = []
        
    def _get_gpu_info(self) -> Dict:
        """Get detailed GPU information"""
        if not torch.cuda.is_available():
            return {"name": "CPU", "vram_gb": 0, "compute_capability": (0, 0)}
            
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "compute_capability": torch.cuda.get_device_capability(0),
            "driver_version": torch.cuda.get_device_properties(0).major,
            "memory_bandwidth_gb": gpu.memoryTotal / 1024 if gpu else 0,
            "temperature": gpu.temperature if gpu else 0,
            "utilization": gpu.load * 100 if gpu else 0
        }
        
    def _get_cache_key(self) -> str:
        """Generate unique cache key for this GPU"""
        gpu_str = f"{self.gpu_info['name']}_{self.gpu_info['vram_gb']:.1f}GB"
        return hashlib.md5(gpu_str.encode()).hexdigest()
        
    def load_cached_results(self) -> Optional[OptimalSettings]:
        """Load cached benchmark results if available"""
        if not os.path.exists(self.cache_file):
            return None
            
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                
            cache_key = self._get_cache_key()
            if cache_key in cache:
                cached_data = cache[cache_key]
                # Check if cache is recent (within 30 days)
                if time.time() - cached_data['timestamp'] < 30 * 24 * 3600:
                    print(f"📊 Loading cached benchmark results for {self.gpu_info['name']}")
                    return OptimalSettings(**cached_data['settings'])
        except:
            pass
            
        return None
        
    def save_cached_results(self, settings: OptimalSettings):
        """Save benchmark results to cache"""
        cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                pass
                
        cache_key = self._get_cache_key()
        cache[cache_key] = {
            'timestamp': time.time(),
            'gpu_info': self.gpu_info,
            'settings': asdict(settings)
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
            
    def create_test_pipeline(self):
        """Create a minimal test pipeline for benchmarking"""
        # Import here to avoid circular imports
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from transformers import Wav2Vec2Model
        
        # Create minimal pipeline with dummy models
        class DummyUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
                
            def forward(self, x, t, encoder_hidden_states=None):
                return self.conv(x)
                
        class DummyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.decoder = torch.nn.Conv2d(4, 3, 3, padding=1)
                
            def encode(self, x):
                class LatentDist:
                    def __init__(self, latent):
                        self.latent = latent
                    def sample(self):
                        return self.latent
                return LatentDist(self.encoder(x))
                
            def decode(self, x):
                return self.decoder(x)
                
        # Create minimal pipeline
        pipeline = LipsyncPipeline(
            unet=DummyUNet().to(self.device),
            vae=DummyVAE().to(self.device),
            audio_encoder=torch.nn.Linear(256, 768).to(self.device),
            scheduler=None
        )
        
        return pipeline
        
    def benchmark_configuration(self, 
                              pipeline,
                              speed_mode: str,
                              quality_preset: str,
                              enable_deepcache: bool,
                              test_frames: int = 25) -> BenchmarkResult:
        """Benchmark a specific configuration"""
        
        # Apply optimizations
        from .nodes import LatentSyncNode
        node = LatentSyncNode()
        
        # Calculate inference steps
        base_steps = 20
        quality_multipliers = {
            "draft": 0.5, "fast": 0.7, "balanced": 1.0, 
            "quality": 1.5, "ultra": 2.0
        }
        speed_multipliers = {
            "normal": 1.0, "fast": 0.7, "turbo": 0.5, "ultra": 0.3
        }
        
        inference_steps = base_steps
        if quality_preset in quality_multipliers:
            inference_steps = int(inference_steps * quality_multipliers[quality_preset])
        if speed_mode in speed_multipliers:
            inference_steps = int(inference_steps * speed_multipliers[speed_mode])
        inference_steps = max(3, inference_steps)
        
        # Apply optimizations to pipeline
        optimized_pipeline = node._apply_speed_optimizations(
            pipeline, 
            speed_mode=speed_mode,
            quality_preset=quality_preset,
            enable_deepcache=enable_deepcache
        )
        
        # Create test data
        test_resolution = (512, 512) if self.gpu_info['vram_gb'] >= 8 else (256, 256)
        test_latents = torch.randn(1, 4, test_resolution[0]//8, test_resolution[1]//8).to(self.device)
        test_audio = torch.randn(1, test_frames, 768).to(self.device)
        
        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = optimized_pipeline.unet(test_latents, 0, test_audio[0, 0:1])
        
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1024**3
        
        # Benchmark
        times = []
        for frame_idx in range(min(test_frames, 10)):  # Test 10 frames max
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Simulate inference step
                for step in range(min(inference_steps, 5)):  # Cap at 5 for benchmark
                    _ = optimized_pipeline.unet(
                        test_latents, 
                        step, 
                        test_audio[0, frame_idx:frame_idx+1]
                    )
                    
            torch.cuda.synchronize()
            frame_time = time.time() - start_time
            times.append(frame_time)
            
        # Calculate metrics
        avg_time = np.mean(times)
        total_time = avg_time * test_frames
        memory_used = (torch.cuda.max_memory_allocated() / 1024**3) - start_mem
        
        # Estimate quality score (simplified)
        quality_score = min(1.0, inference_steps / 20.0)
        if speed_mode == "ultra":
            quality_score *= 0.7
        elif speed_mode == "turbo":
            quality_score *= 0.85
        elif speed_mode == "fast":
            quality_score *= 0.95
            
        # Calculate speedup vs baseline
        baseline_time = 0.8  # Estimated baseline time per frame
        speedup = baseline_time / avg_time
        
        # Cleanup
        del optimized_pipeline
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            speed_mode=speed_mode,
            quality_preset=quality_preset,
            enable_deepcache=enable_deepcache,
            inference_steps=inference_steps,
            avg_time_per_frame=avg_time,
            total_time=total_time,
            memory_used_gb=memory_used,
            quality_score=quality_score,
            speedup=speedup
        )
        
    def run_comprehensive_benchmark(self, quick_mode: bool = False) -> OptimalSettings:
        """Run full benchmark suite"""
        
        print(f"\n{'='*60}")
        print(f"🚀 GPU BENCHMARK FOR {self.gpu_info['name']}")
        print(f"{'='*60}")
        print(f"VRAM: {self.gpu_info['vram_gb']:.1f} GB")
        print(f"Compute Capability: {self.gpu_info['compute_capability']}")
        print(f"{'='*60}\n")
        
        # Check cache first
        cached_settings = self.load_cached_results()
        if cached_settings and quick_mode:
            print("✅ Using cached benchmark results")
            return cached_settings
            
        # Create test pipeline
        try:
            pipeline = self.create_test_pipeline()
        except Exception as e:
            print(f"⚠️ Could not create test pipeline: {e}")
            return self._get_default_settings()
            
        # Define test configurations
        if quick_mode:
            # Quick benchmark - test fewer configurations
            configs = [
                ("normal", "balanced", False),
                ("fast", "balanced", True),
                ("turbo", "fast", True),
            ]
        else:
            # Full benchmark
            configs = [
                # Baseline
                ("normal", "balanced", False),
                # Fast configs
                ("fast", "balanced", False),
                ("fast", "balanced", True),
                ("fast", "fast", True),
                # Turbo configs
                ("turbo", "fast", False),
                ("turbo", "fast", True),
                ("turbo", "draft", True),
                # Ultra configs (only for high-end GPUs)
                ("ultra", "draft", True),
            ]
            
        # Filter configs based on GPU capability
        if self.gpu_info['vram_gb'] < 8:
            configs = [c for c in configs if c[0] != "ultra"]
            
        # Run benchmarks
        print("Running benchmark configurations...")
        for i, (speed, quality, deepcache) in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Testing: {speed}/{quality}, DeepCache={'ON' if deepcache else 'OFF'}")
            
            try:
                result = self.benchmark_configuration(
                    pipeline, speed, quality, deepcache
                )
                self.results.append(result)
                
                print(f"  ⏱️  Time per frame: {result.avg_time_per_frame:.3f}s")
                print(f"  🚀 Speedup: {result.speedup:.2f}x")
                print(f"  💾 Memory: {result.memory_used_gb:.2f} GB")
                print(f"  ✨ Quality: {result.quality_score*100:.0f}%")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                
        # Analyze results
        optimal_settings = self._analyze_results()
        
        # Save to cache
        self.save_cached_results(optimal_settings)
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        
        return optimal_settings
        
    def _analyze_results(self) -> OptimalSettings:
        """Analyze benchmark results and determine optimal settings"""
        
        if not self.results:
            return self._get_default_settings()
            
        # Sort by different criteria
        by_speed = sorted(self.results, key=lambda x: x.avg_time_per_frame)
        by_quality = sorted(self.results, key=lambda x: x.quality_score, reverse=True)
        by_balanced = sorted(self.results, key=lambda x: x.speedup * x.quality_score, reverse=True)
        
        # Find best configurations
        fastest = by_speed[0]
        highest_quality = by_quality[0]
        best_balanced = by_balanced[0]
        
        # Determine GPU capabilities
        supports_ultra = self.gpu_info['vram_gb'] >= 8 and any(r.speed_mode == "ultra" for r in self.results)
        supports_deepcache = any(r.enable_deepcache and r.speedup > 1.2 for r in self.results)
        
        # Determine recommended batch size based on VRAM
        if self.gpu_info['vram_gb'] >= 24:  # RTX 4090, A6000
            batch_size = 16
            max_res = (768, 768)
        elif self.gpu_info['vram_gb'] >= 16:  # RTX 4080, 3090
            batch_size = 12
            max_res = (640, 640)
        elif self.gpu_info['vram_gb'] >= 12:  # RTX 4070, 3080
            batch_size = 8
            max_res = (512, 512)
        elif self.gpu_info['vram_gb'] >= 8:   # RTX 4060, 3070
            batch_size = 4
            max_res = (512, 512)
        else:  # Lower-end GPUs
            batch_size = 2
            max_res = (256, 256)
            
        # Build optimal settings
        settings = OptimalSettings(
            preview_mode={
                "speed_mode": fastest.speed_mode,
                "quality_preset": fastest.quality_preset,
                "enable_deepcache": fastest.enable_deepcache,
                "expected_speedup": fastest.speedup
            },
            balanced_mode={
                "speed_mode": best_balanced.speed_mode,
                "quality_preset": best_balanced.quality_preset,
                "enable_deepcache": best_balanced.enable_deepcache,
                "expected_speedup": best_balanced.speedup
            },
            quality_mode={
                "speed_mode": highest_quality.speed_mode,
                "quality_preset": highest_quality.quality_preset,
                "enable_deepcache": highest_quality.enable_deepcache,
                "expected_speedup": highest_quality.speedup
            },
            short_video={
                "speed_mode": "fast" if supports_deepcache else "normal",
                "quality_preset": "balanced",
                "enable_deepcache": False,  # Not worth it for short videos
            },
            medium_video={
                "speed_mode": best_balanced.speed_mode,
                "quality_preset": best_balanced.quality_preset,
                "enable_deepcache": supports_deepcache,
            },
            long_video={
                "speed_mode": "turbo" if self.gpu_info['vram_gb'] >= 8 else "fast",
                "quality_preset": "fast",
                "enable_deepcache": True,
            },
            supports_ultra=supports_ultra,
            supports_deepcache=supports_deepcache,
            recommended_batch_size=batch_size,
            max_resolution=max_res
        )
        
        return settings
        
    def _get_default_settings(self) -> OptimalSettings:
        """Get conservative default settings"""
        
        # Base on VRAM
        if self.gpu_info['vram_gb'] >= 16:
            default_speed = "fast"
            default_quality = "balanced"
            deepcache = True
        elif self.gpu_info['vram_gb'] >= 8:
            default_speed = "fast"
            default_quality = "fast"
            deepcache = True
        else:
            default_speed = "normal"
            default_quality = "fast"
            deepcache = False
            
        return OptimalSettings(
            preview_mode={
                "speed_mode": "turbo" if self.gpu_info['vram_gb'] >= 8 else "fast",
                "quality_preset": "draft",
                "enable_deepcache": deepcache,
                "expected_speedup": 2.0
            },
            balanced_mode={
                "speed_mode": default_speed,
                "quality_preset": default_quality,
                "enable_deepcache": deepcache,
                "expected_speedup": 1.5
            },
            quality_mode={
                "speed_mode": "normal",
                "quality_preset": "quality",
                "enable_deepcache": False,
                "expected_speedup": 1.0
            },
            short_video={
                "speed_mode": default_speed,
                "quality_preset": default_quality,
                "enable_deepcache": False,
            },
            medium_video={
                "speed_mode": default_speed,
                "quality_preset": default_quality,
                "enable_deepcache": deepcache,
            },
            long_video={
                "speed_mode": "turbo" if self.gpu_info['vram_gb'] >= 8 else "fast",
                "quality_preset": "fast",
                "enable_deepcache": True,
            },
            supports_ultra=self.gpu_info['vram_gb'] >= 8,
            supports_deepcache=self.gpu_info['vram_gb'] >= 6,
            recommended_batch_size=min(8, int(self.gpu_info['vram_gb'] / 2)),
            max_resolution=(512, 512) if self.gpu_info['vram_gb'] >= 8 else (256, 256)
        )
        
    def print_recommendations(self, settings: OptimalSettings):
        """Print benchmark results and recommendations"""
        
        print(f"\n{'='*60}")
        print("📊 BENCHMARK RESULTS & RECOMMENDATIONS")
        print(f"{'='*60}\n")
        
        print("🎯 OPTIMAL SETTINGS BY USE CASE:\n")
        
        print("1️⃣ PREVIEW/DRAFT MODE:")
        print(f"   Speed Mode: {settings.preview_mode['speed_mode']}")
        print(f"   Quality Preset: {settings.preview_mode['quality_preset']}")
        print(f"   DeepCache: {'ON' if settings.preview_mode['enable_deepcache'] else 'OFF'}")
        print(f"   Expected Speedup: {settings.preview_mode['expected_speedup']:.1f}x\n")
        
        print("2️⃣ BALANCED PERFORMANCE:")
        print(f"   Speed Mode: {settings.balanced_mode['speed_mode']}")
        print(f"   Quality Preset: {settings.balanced_mode['quality_preset']}")
        print(f"   DeepCache: {'ON' if settings.balanced_mode['enable_deepcache'] else 'OFF'}")
        print(f"   Expected Speedup: {settings.balanced_mode['expected_speedup']:.1f}x\n")
        
        print("3️⃣ HIGH QUALITY:")
        print(f"   Speed Mode: {settings.quality_mode['speed_mode']}")
        print(f"   Quality Preset: {settings.quality_mode['quality_preset']}")
        print(f"   DeepCache: {'ON' if settings.quality_mode['enable_deepcache'] else 'OFF'}")
        print(f"   Expected Speedup: {settings.quality_mode['expected_speedup']:.1f}x\n")
        
        print("📹 VIDEO LENGTH RECOMMENDATIONS:\n")
        print(f"   Short (<5s): {settings.short_video['speed_mode']}/{settings.short_video['quality_preset']}")
        print(f"   Medium (5-20s): {settings.medium_video['speed_mode']}/{settings.medium_video['quality_preset']}")
        print(f"   Long (>20s): {settings.long_video['speed_mode']}/{settings.long_video['quality_preset']}\n")
        
        print("🔧 GPU CAPABILITIES:")
        print(f"   Ultra Mode: {'✅ Supported' if settings.supports_ultra else '❌ Not Recommended'}")
        print(f"   DeepCache: {'✅ Supported' if settings.supports_deepcache else '❌ Not Recommended'}")
        print(f"   Recommended Batch Size: {settings.recommended_batch_size}")
        print(f"   Max Resolution: {settings.max_resolution[0]}x{settings.max_resolution[1]}")
        
        print(f"\n{'='*60}\n")

# Convenience function for quick benchmarking
def benchmark_gpu(quick_mode: bool = True) -> OptimalSettings:
    """Run GPU benchmark and return optimal settings"""
    
    benchmark = GPUBenchmark()
    settings = benchmark.run_comprehensive_benchmark(quick_mode=quick_mode)
    benchmark.print_recommendations(settings)
    
    return settings

# Auto-benchmark on import if running as main
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark GPU for LatentSync")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (slower but more accurate)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached results and re-run")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        cache_file = os.path.expanduser("~/.latentsync_benchmark_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("✅ Cleared benchmark cache")
    
    benchmark_gpu(quick_mode=not args.full)