"""
Unified Speed Optimization System for LatentSync
Consolidates all speed optimization logic in one place
"""

import torch
from typing import Dict, Any, Optional, Tuple


class UnifiedSpeedOptimizer:
    """Centralized speed optimization for LatentSync pipeline"""
    
    # Speed mode configurations
    SPEED_MODES = {
        "normal": {
            "steps_multiplier": 1.0,
            "batch_multiplier": 1.0,
            "optimization_level": "balanced",
            "enable_optimizations": False
        },
        "fast": {
            "steps_multiplier": 0.75,
            "batch_multiplier": 1.5,
            "optimization_level": "balanced",
            "enable_optimizations": True
        },
        "turbo": {
            "steps_multiplier": 0.5,
            "batch_multiplier": 2.0,
            "optimization_level": "aggressive",
            "enable_optimizations": True
        },
        "ultra": {
            "steps_multiplier": 0.25,
            "batch_multiplier": 3.0,
            "optimization_level": "aggressive",
            "enable_optimizations": True
        }
    }
    
    # Import quality presets from the existing module
    @property
    def quality_presets(self):
        """Get quality presets from the existing quality_presets module"""
        try:
            from .quality_presets import LatentSyncQualityPresets
            preset_manager = LatentSyncQualityPresets()
            return {
                name: {
                    "base_steps": preset.inference_steps,
                    "quality_score": preset.quality_score,
                    "speed_score": preset.speed_score,
                    "optimization_level": "aggressive" if preset.optimize_speed else "balanced"
                }
                for name, preset in preset_manager.PRESETS.items()
            }
        except ImportError:
            # Fallback if quality_presets module not available
            return {
                "balanced": {
                    "base_steps": 20,
                    "quality_score": 0.7,
                    "speed_score": 0.5,
                    "optimization_level": "balanced"
                }
            }
    
    def __init__(self):
        self.turbo_enabled = False
        self.deepcache_enabled = False
        self.adaptive_steps_enabled = False
        
    def get_optimized_settings(
        self,
        base_inference_steps: int,
        base_batch_size: int,
        speed_mode: str = "normal",
        quality_preset: str = "auto",
        enable_deepcache: bool = False,
        gpu_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimized settings based on speed mode and quality preset
        
        Returns:
            Dict with optimized inference_steps, batch_size, and optimization_level
        """
        settings = {
            "inference_steps": base_inference_steps,
            "batch_size": base_batch_size,
            "optimization_level": "balanced",
            "enable_deepcache": enable_deepcache,
            "cache_interval": 3
        }
        
        # Apply quality preset first (if not auto)
        if quality_preset != "auto" and quality_preset in self.quality_presets:
            preset = self.quality_presets[quality_preset]
            settings["inference_steps"] = preset["base_steps"]
            settings["optimization_level"] = preset["optimization_level"]
            
            # Adjust cache interval based on quality
            if preset["speed_score"] > 0.7:
                settings["cache_interval"] = 2
            elif preset["speed_score"] < 0.3:
                settings["cache_interval"] = 5
                
        # Apply speed mode adjustments
        if speed_mode in self.SPEED_MODES:
            mode_config = self.SPEED_MODES[speed_mode]
            
            # Adjust steps
            settings["inference_steps"] = max(
                3,  # Minimum steps
                int(settings["inference_steps"] * mode_config["steps_multiplier"])
            )
            
            # Adjust batch size
            settings["batch_size"] = max(
                1,
                int(settings["batch_size"] * mode_config["batch_multiplier"])
            )
            
            # Override optimization level if more aggressive
            if mode_config["optimization_level"] == "aggressive":
                settings["optimization_level"] = "aggressive"
                
            # Enable optimizations for speed modes
            if mode_config["enable_optimizations"]:
                settings["enable_deepcache"] = True
                
        # GPU-specific adjustments
        if gpu_info:
            vram_gb = gpu_info.get("vram_gb", 8)
            
            # High-end GPU optimizations
            if vram_gb >= 16:
                if speed_mode in ["turbo", "ultra"]:
                    settings["batch_size"] = min(20, settings["batch_size"] * 1.5)
                    
            # Low-end GPU limitations
            elif vram_gb < 8:
                settings["batch_size"] = min(4, settings["batch_size"])
                if settings["optimization_level"] == "aggressive":
                    settings["optimization_level"] = "balanced"
                    
        return settings
        
    def apply_to_pipeline(
        self,
        pipeline: Any,
        settings: Dict[str, Any]
    ) -> Any:
        """Apply optimizations to the pipeline"""
        
        # Apply DeepCache if enabled
        if settings.get("enable_deepcache", False):
            try:
                from .deepcache_integration import accelerate_with_deepcache
                pipeline = accelerate_with_deepcache(
                    pipeline,
                    cache_interval=settings.get("cache_interval", 3)
                )
                self.deepcache_enabled = True
                print(f"✓ DeepCache enabled (interval={settings['cache_interval']})")
            except ImportError:
                print("DeepCache module not available")
                
        # Apply Turbo Mode for fast/turbo/ultra modes
        if settings.get("optimization_level") == "aggressive":
            try:
                from .turbo_mode import enable_turbo_mode
                pipeline = enable_turbo_mode(pipeline)
                self.turbo_enabled = True
                print("✓ Turbo Mode enabled")
            except ImportError:
                print("Turbo Mode module not available")
                
        # Configure pipeline settings
        if hasattr(pipeline, 'set_optimization_settings'):
            pipeline.set_optimization_settings(settings)
            
        return pipeline
        
    def get_speedup_estimate(self, settings: Dict[str, Any]) -> float:
        """Estimate speedup based on settings"""
        speedup = 1.0
        
        # Base speedup from reduced steps
        base_steps = 20  # Default
        current_steps = settings.get("inference_steps", base_steps)
        speedup *= (base_steps / current_steps)
        
        # DeepCache speedup
        if settings.get("enable_deepcache", False):
            cache_interval = settings.get("cache_interval", 3)
            speedup *= (1 + (cache_interval - 1) * 0.3)  # ~30% speedup per cached step
            
        # Batch size speedup (diminishing returns)
        base_batch = 1
        current_batch = settings.get("batch_size", base_batch)
        if current_batch > base_batch:
            speedup *= (1 + (current_batch / base_batch - 1) * 0.2)
            
        return speedup
        
    def print_optimization_summary(self, settings: Dict[str, Any], original_settings: Dict[str, Any]):
        """Print a summary of applied optimizations"""
        print("\n" + "="*60)
        print("⚡ SPEED OPTIMIZATIONS APPLIED ⚡")
        print("="*60)
        
        # Steps reduction
        original_steps = original_settings.get("inference_steps", 20)
        optimized_steps = settings.get("inference_steps", 20)
        if optimized_steps != original_steps:
            print(f"Inference Steps: {original_steps} → {optimized_steps} ({optimized_steps/original_steps:.0%})")
            
        # Batch size change
        original_batch = original_settings.get("batch_size", 1)
        optimized_batch = settings.get("batch_size", 1)
        if optimized_batch != original_batch:
            print(f"Batch Size: {original_batch} → {optimized_batch} ({optimized_batch/original_batch:.0%})")
            
        # Features enabled
        if settings.get("enable_deepcache", False):
            print(f"DeepCache: ✓ Enabled (interval={settings.get('cache_interval', 3)})")
        if self.turbo_enabled:
            print("Turbo Mode: ✓ Enabled")
            
        # Estimated speedup
        speedup = self.get_speedup_estimate(settings)
        print(f"\nEstimated Speedup: ~{speedup:.1f}x")
        print("="*60 + "\n")


# Singleton instance
unified_optimizer = UnifiedSpeedOptimizer()


def optimize_inference_settings(
    inference_steps: int = 20,
    batch_size: int = 1,
    speed_mode: str = "normal",
    quality_preset: str = "auto",
    enable_deepcache: bool = False,
    gpu_info: Optional[Dict[str, Any]] = None,
    print_summary: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for getting optimized inference settings
    
    Args:
        inference_steps: Base number of inference steps
        batch_size: Base batch size
        speed_mode: One of "normal", "fast", "turbo", "ultra"
        quality_preset: One of "auto", "draft", "fast", "balanced", "quality", "ultra"
        enable_deepcache: Whether to enable DeepCache optimization
        gpu_info: Optional GPU information for hardware-specific optimizations
        print_summary: Whether to print optimization summary
        
    Returns:
        Dict with optimized settings
    """
    original_settings = {
        "inference_steps": inference_steps,
        "batch_size": batch_size,
        "optimization_level": "balanced"
    }
    
    # Get optimized settings
    settings = unified_optimizer.get_optimized_settings(
        inference_steps,
        batch_size,
        speed_mode,
        quality_preset,
        enable_deepcache,
        gpu_info
    )
    
    # Print summary if requested
    if print_summary and (speed_mode != "normal" or quality_preset != "auto"):
        unified_optimizer.print_optimization_summary(settings, original_settings)
        
    return settings


def apply_pipeline_optimizations(
    pipeline: Any,
    settings: Dict[str, Any]
) -> Any:
    """
    Apply optimizations to a pipeline based on settings
    
    Args:
        pipeline: The pipeline to optimize
        settings: Optimization settings from optimize_inference_settings
        
    Returns:
        Optimized pipeline
    """
    return unified_optimizer.apply_to_pipeline(pipeline, settings)