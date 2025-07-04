"""
Quality vs Speed presets for LatentSync
Provides easy-to-use presets for different use cases
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class QualityPreset:
    """Quality preset configuration"""
    name: str
    description: str
    inference_steps: int
    batch_size: int
    attention_mode: str
    vae_slicing: bool
    vae_tiling: bool
    mixed_precision: bool
    memory_efficient: bool
    optimize_speed: bool
    quality_score: float  # 0-1, higher is better quality
    speed_score: float    # 0-1, higher is faster
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline kwargs"""
        return {
            'num_inference_steps': self.inference_steps,
            'batch_size': self.batch_size,
            'attention_mode': self.attention_mode,
            'enable_vae_slicing': self.vae_slicing,
            'enable_vae_tiling': self.vae_tiling,
            'mixed_precision': self.mixed_precision,
            'memory_efficient': self.memory_efficient,
            'optimize_for_speed': self.optimize_speed
        }


class LatentSyncQualityPresets:
    """Preset manager for LatentSync quality settings"""
    
    # Define built-in presets
    PRESETS = {
        'draft': QualityPreset(
            name='Draft',
            description='Fastest processing, lower quality - good for previews',
            inference_steps=10,
            batch_size=16,
            attention_mode='flash',
            vae_slicing=True,
            vae_tiling=True,
            mixed_precision=True,
            memory_efficient=True,
            optimize_speed=True,
            quality_score=0.3,
            speed_score=1.0
        ),
        'fast': QualityPreset(
            name='Fast',
            description='Good balance for quick results',
            inference_steps=15,
            batch_size=12,
            attention_mode='flash',
            vae_slicing=True,
            vae_tiling=False,
            mixed_precision=True,
            memory_efficient=True,
            optimize_speed=True,
            quality_score=0.5,
            speed_score=0.8
        ),
        'balanced': QualityPreset(
            name='Balanced',
            description='Default settings - good quality and reasonable speed',
            inference_steps=20,
            batch_size=8,
            attention_mode='flash',
            vae_slicing=False,
            vae_tiling=False,
            mixed_precision=True,
            memory_efficient=True,
            optimize_speed=False,
            quality_score=0.7,
            speed_score=0.6
        ),
        'quality': QualityPreset(
            name='Quality',
            description='Higher quality, slower processing',
            inference_steps=30,
            batch_size=6,
            attention_mode='flex',
            vae_slicing=False,
            vae_tiling=False,
            mixed_precision=True,
            memory_efficient=False,
            optimize_speed=False,
            quality_score=0.85,
            speed_score=0.4
        ),
        'ultra': QualityPreset(
            name='Ultra',
            description='Maximum quality, slowest processing',
            inference_steps=40,
            batch_size=4,
            attention_mode='flex',
            vae_slicing=False,
            vae_tiling=False,
            mixed_precision=False,
            memory_efficient=False,
            optimize_speed=False,
            quality_score=1.0,
            speed_score=0.2
        )
    }
    
    def __init__(self):
        self.custom_presets: Dict[str, QualityPreset] = {}
        self.load_custom_presets()
        
    def get_preset(self, name: str) -> Optional[QualityPreset]:
        """Get a preset by name"""
        # Check built-in presets first
        if name in self.PRESETS:
            return self.PRESETS[name]
        # Then check custom presets
        return self.custom_presets.get(name)
        
    def get_preset_from_slider(self, value: float) -> QualityPreset:
        """
        Get preset from slider value (0-1)
        
        Args:
            value: Slider value where 0 = fastest, 1 = highest quality
            
        Returns:
            Interpolated preset
        """
        # Map slider to presets
        if value <= 0.2:
            return self.PRESETS['draft']
        elif value <= 0.4:
            return self.interpolate_presets('draft', 'fast', (value - 0.2) / 0.2)
        elif value <= 0.6:
            return self.interpolate_presets('fast', 'balanced', (value - 0.4) / 0.2)
        elif value <= 0.8:
            return self.interpolate_presets('balanced', 'quality', (value - 0.6) / 0.2)
        else:
            return self.interpolate_presets('quality', 'ultra', (value - 0.8) / 0.2)
            
    def interpolate_presets(self, preset1_name: str, preset2_name: str, weight: float) -> QualityPreset:
        """
        Interpolate between two presets
        
        Args:
            preset1_name: First preset name
            preset2_name: Second preset name
            weight: Weight for second preset (0-1)
            
        Returns:
            Interpolated preset
        """
        p1 = self.PRESETS[preset1_name]
        p2 = self.PRESETS[preset2_name]
        
        # Interpolate numeric values
        inference_steps = int(p1.inference_steps * (1 - weight) + p2.inference_steps * weight)
        batch_size = int(p1.batch_size * (1 - weight) + p2.batch_size * weight)
        quality_score = p1.quality_score * (1 - weight) + p2.quality_score * weight
        speed_score = p1.speed_score * (1 - weight) + p2.speed_score * weight
        
        # Choose discrete values based on weight
        if weight < 0.5:
            base_preset = p1
        else:
            base_preset = p2
            
        return QualityPreset(
            name=f'Custom ({p1.name}-{p2.name})',
            description=f'Interpolated between {p1.name} and {p2.name}',
            inference_steps=inference_steps,
            batch_size=batch_size,
            attention_mode=base_preset.attention_mode,
            vae_slicing=base_preset.vae_slicing,
            vae_tiling=base_preset.vae_tiling,
            mixed_precision=base_preset.mixed_precision,
            memory_efficient=base_preset.memory_efficient,
            optimize_speed=base_preset.optimize_speed,
            quality_score=quality_score,
            speed_score=speed_score
        )
        
    def create_custom_preset(self, name: str, base_preset: str, **kwargs) -> QualityPreset:
        """
        Create a custom preset based on an existing one
        
        Args:
            name: Name for the custom preset
            base_preset: Base preset to modify
            **kwargs: Values to override
            
        Returns:
            New custom preset
        """
        base = self.get_preset(base_preset)
        if not base:
            raise ValueError(f"Base preset '{base_preset}' not found")
            
        # Create new preset with overrides
        preset_dict = base.__dict__.copy()
        preset_dict['name'] = name
        preset_dict.update(kwargs)
        
        custom = QualityPreset(**preset_dict)
        self.custom_presets[name] = custom
        self.save_custom_presets()
        
        return custom
        
    def estimate_processing_time(self, preset: QualityPreset, num_frames: int, gpu_name: str = "RTX 4090") -> float:
        """
        Estimate processing time for given preset and video length
        
        Args:
            preset: Quality preset
            num_frames: Number of frames to process
            gpu_name: GPU model name
            
        Returns:
            Estimated time in seconds
        """
        # Base time per frame for RTX 4090 at balanced settings
        base_time_per_frame = 0.8  # seconds
        
        # Adjust for GPU (rough estimates)
        gpu_multipliers = {
            "RTX 4090": 1.0,
            "RTX 4080": 1.3,
            "RTX 4070": 1.8,
            "RTX 3090": 1.5,
            "RTX 3080": 2.0,
            "RTX 3070": 2.5,
        }
        gpu_mult = gpu_multipliers.get(gpu_name, 2.0)
        
        # Adjust for quality settings
        steps_mult = preset.inference_steps / 20  # Relative to balanced (20 steps)
        batch_mult = 8 / preset.batch_size  # Relative to balanced (8 batch)
        
        # Attention mode multiplier
        attention_mult = {
            'flash': 1.0,
            'flex': 1.1,
            'xformers': 1.0,
            'standard': 1.5
        }.get(preset.attention_mode, 1.2)
        
        # Calculate total time
        time_per_frame = base_time_per_frame * gpu_mult * steps_mult * batch_mult * attention_mult
        
        # Account for optimizations
        if preset.optimize_speed:
            time_per_frame *= 0.8
        if preset.mixed_precision:
            time_per_frame *= 0.9
            
        total_time = time_per_frame * num_frames
        
        return total_time
        
    def get_recommended_preset(self, 
                             video_length: int,
                             target_time: Optional[float] = None,
                             min_quality: float = 0.5) -> QualityPreset:
        """
        Get recommended preset based on video length and constraints
        
        Args:
            video_length: Number of frames
            target_time: Target processing time in seconds (optional)
            min_quality: Minimum acceptable quality score (0-1)
            
        Returns:
            Recommended preset
        """
        candidates = []
        
        for preset_name, preset in self.PRESETS.items():
            if preset.quality_score >= min_quality:
                est_time = self.estimate_processing_time(preset, video_length)
                
                if target_time is None or est_time <= target_time:
                    candidates.append((preset, est_time))
                    
        if not candidates:
            # Return fastest preset that meets quality requirement
            return self.PRESETS['fast']
            
        # Sort by quality score (descending)
        candidates.sort(key=lambda x: x[0].quality_score, reverse=True)
        
        return candidates[0][0]
        
    def save_custom_presets(self):
        """Save custom presets to file"""
        import os
        
        preset_file = os.path.join(os.path.dirname(__file__), 'custom_presets.json')
        
        data = {}
        for name, preset in self.custom_presets.items():
            data[name] = preset.__dict__
            
        with open(preset_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_custom_presets(self):
        """Load custom presets from file"""
        import os
        
        preset_file = os.path.join(os.path.dirname(__file__), 'custom_presets.json')
        
        if os.path.exists(preset_file):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                    
                for name, preset_dict in data.items():
                    self.custom_presets[name] = QualityPreset(**preset_dict)
            except Exception as e:
                print(f"Error loading custom presets: {e}")


# ComfyUI integration
class QualitySliderNode:
    """ComfyUI node for quality/speed control"""
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_names = list(LatentSyncQualityPresets.PRESETS.keys())
        
        return {
            "required": {
                "quality_slider": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "preset_override": (["auto"] + preset_names, {"default": "auto"}),
                "show_estimate": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_frames": ("INT", {"default": 100, "min": 1}),
            }
        }
        
    RETURN_TYPES = ("LATENTSYNC_SETTINGS", "STRING")
    RETURN_NAMES = ("settings", "info")
    FUNCTION = "get_quality_settings"
    CATEGORY = "LatentSync"
    
    def __init__(self):
        self.preset_manager = LatentSyncQualityPresets()
        
    def get_quality_settings(self, quality_slider, preset_override, show_estimate, video_frames=100):
        """Get quality settings from slider or preset"""
        
        # Get preset
        if preset_override == "auto":
            preset = self.preset_manager.get_preset_from_slider(quality_slider)
        else:
            preset = self.preset_manager.get_preset(preset_override)
            
        # Get settings dict
        settings = preset.to_dict()
        
        # Create info string
        info_parts = [
            f"Preset: {preset.name}",
            f"Quality: {preset.quality_score*100:.0f}%",
            f"Speed: {preset.speed_score*100:.0f}%",
            f"Steps: {preset.inference_steps}",
            f"Batch: {preset.batch_size}",
        ]
        
        if show_estimate:
            est_time = self.preset_manager.estimate_processing_time(preset, video_frames)
            info_parts.append(f"Est. Time: {est_time/60:.1f} min")
            
        info = " | ".join(info_parts)
        
        return (settings, info)


# Helper function for easy integration
def apply_quality_preset(pipeline, preset_name: str = 'balanced'):
    """
    Apply a quality preset to a LatentSync pipeline
    
    Args:
        pipeline: LatentSync pipeline instance
        preset_name: Name of the preset to apply
        
    Returns:
        Dictionary of settings applied
    """
    manager = LatentSyncQualityPresets()
    preset = manager.get_preset(preset_name)
    
    if not preset:
        print(f"Warning: Preset '{preset_name}' not found, using 'balanced'")
        preset = manager.get_preset('balanced')
        
    settings = preset.to_dict()
    
    # Apply settings to pipeline
    if hasattr(pipeline, 'set_attention_mode'):
        pipeline.set_attention_mode(settings['attention_mode'])
        
    # Store settings for later use
    pipeline._quality_preset = preset
    pipeline._quality_settings = settings
    
    print(f"✅ Applied preset: {preset.name}")
    print(f"   Quality: {preset.quality_score*100:.0f}% | Speed: {preset.speed_score*100:.0f}%")
    
    return settings