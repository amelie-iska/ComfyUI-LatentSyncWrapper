"""
Adaptive GPU Configuration for ComfyUI-LatentSyncWrapper
Automatically detects and optimizes settings based on GPU capabilities
"""
import torch
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class GPUProfile:
    """GPU performance profile with optimized settings"""
    name: str
    vram_gb: float
    compute_capability: Tuple[int, int]
    batch_size: int
    num_frames: int
    inference_steps: int
    use_mixed_precision: bool
    enable_tf32: bool
    vram_fraction: float
    enable_optimizations: Dict[str, bool]

class AdaptiveGPUConfig:
    """Adaptive GPU configuration that detects hardware and optimizes settings"""
    
    # GPU profiles based on common cards and their capabilities
    GPU_PROFILES = {
        "RTX 4090": GPUProfile(
            name="RTX 4090",
            vram_gb=24,
            compute_capability=(8, 9),
            batch_size=8,  # Reduced from 12 to prevent display lag
            num_frames=16,  # Can handle full 16 frames
            inference_steps=15,  # Reduced from 20 for faster inference
            use_mixed_precision=True,
            enable_tf32=True,
            vram_fraction=0.85,  # Can use more VRAM safely
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": False,  # Don't need CPU offload
                "gradient_checkpointing": False,
                "attention_slicing": False,  # Don't need slicing
                "enable_time_slicing": True,  # New: time-sliced execution
                "display_priority_mode": True,  # New: prioritize display
                "yield_frequency_ms": 1,  # New: yield every 1ms
                "async_execution": True  # New: async GPU operations
            }
        ),
        "RTX 4080": GPUProfile(
            name="RTX 4080",
            vram_gb=16,
            compute_capability=(8, 9),
            batch_size=8,
            num_frames=12,
            inference_steps=20,
            use_mixed_precision=True,
            enable_tf32=True,
            vram_fraction=0.80,
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": False,
                "gradient_checkpointing": False,
                "attention_slicing": False,
            }
        ),
        "RTX 3090": GPUProfile(
            name="RTX 3090",
            vram_gb=24,
            compute_capability=(8, 6),
            batch_size=10,
            num_frames=16,
            inference_steps=20,
            use_mixed_precision=True,
            enable_tf32=False,  # No TF32 on Ampere
            vram_fraction=0.80,
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": False,
                "gradient_checkpointing": False,
                "attention_slicing": False,
            }
        ),
        "RTX 3080": GPUProfile(
            name="RTX 3080",
            vram_gb=10,
            compute_capability=(8, 6),
            batch_size=4,
            num_frames=8,
            inference_steps=20,
            use_mixed_precision=True,
            enable_tf32=False,
            vram_fraction=0.75,
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": True,
                "gradient_checkpointing": True,
                "attention_slicing": True,
            }
        ),
        "RTX 3070": GPUProfile(
            name="RTX 3070",
            vram_gb=8,
            compute_capability=(8, 6),
            batch_size=2,
            num_frames=8,
            inference_steps=15,
            use_mixed_precision=True,
            enable_tf32=False,
            vram_fraction=0.70,
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": True,
                "gradient_checkpointing": True,
                "attention_slicing": True,
            }
        ),
        "DEFAULT_HIGH": GPUProfile(
            name="High-End GPU",
            vram_gb=20,
            compute_capability=(8, 0),
            batch_size=8,
            num_frames=12,
            inference_steps=20,
            use_mixed_precision=True,
            enable_tf32=True,
            vram_fraction=0.80,
            enable_optimizations={
                "flash_attention": True,
                "vae_slicing": True,
                "vae_tiling": False,
                "cpu_offload": False,
                "gradient_checkpointing": False,
                "attention_slicing": False,
            }
        ),
        "DEFAULT_MID": GPUProfile(
            name="Mid-Range GPU",
            vram_gb=12,
            compute_capability=(7, 5),
            batch_size=4,
            num_frames=8,
            inference_steps=20,
            use_mixed_precision=True,
            enable_tf32=False,
            vram_fraction=0.75,
            enable_optimizations={
                "flash_attention": False,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": True,
                "gradient_checkpointing": True,
                "attention_slicing": True,
            }
        ),
        "DEFAULT_LOW": GPUProfile(
            name="Entry-Level GPU",
            vram_gb=6,
            compute_capability=(6, 0),
            batch_size=1,
            num_frames=4,
            inference_steps=10,
            use_mixed_precision=False,
            enable_tf32=False,
            vram_fraction=0.65,
            enable_optimizations={
                "flash_attention": False,
                "vae_slicing": True,
                "vae_tiling": True,
                "cpu_offload": True,
                "gradient_checkpointing": True,
                "attention_slicing": True,
            }
        ),
    }
    
    def __init__(self, custom_vram_fraction: Optional[float] = None):
        """Initialize with optional custom VRAM fraction override"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.custom_vram_fraction = custom_vram_fraction
        self.gpu_info = self._detect_gpu()
        self.profile = self._select_profile()
        self._load_user_preferences()
        
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities and properties"""
        if not torch.cuda.is_available():
            return {"name": "CPU", "vram_gb": 0, "compute_capability": (0, 0)}
        
        properties = torch.cuda.get_device_properties(0)
        vram_gb = properties.total_memory / (1024 ** 3)
        compute_capability = (properties.major, properties.minor)
        
        # Try to get the marketing name
        gpu_name = properties.name
        
        # Clean up GPU name for matching
        gpu_name_clean = gpu_name.upper().replace("GEFORCE ", "").replace("NVIDIA ", "")
        
        return {
            "name": gpu_name,
            "name_clean": gpu_name_clean,
            "vram_gb": vram_gb,
            "compute_capability": compute_capability,
            "cuda_cores": properties.multi_processor_count,
            "memory_bandwidth": getattr(properties, 'memory_bandwidth', 0),
        }
    
    def _select_profile(self) -> GPUProfile:
        """Select the best matching GPU profile"""
        gpu_name = self.gpu_info["name_clean"]
        vram_gb = self.gpu_info["vram_gb"]
        
        # Check for exact GPU model match
        for profile_name, profile in self.GPU_PROFILES.items():
            if profile_name.upper() in gpu_name:
                print(f"Detected {profile_name} - using optimized profile")
                return self._adjust_profile_for_vram(profile)
        
        # Fall back to VRAM-based selection
        if vram_gb >= 20:
            profile = self.GPU_PROFILES["DEFAULT_HIGH"]
        elif vram_gb >= 10:
            profile = self.GPU_PROFILES["DEFAULT_MID"]
        else:
            profile = self.GPU_PROFILES["DEFAULT_LOW"]
        
        print(f"Using {profile.name} profile for {vram_gb:.1f}GB GPU")
        return self._adjust_profile_for_vram(profile)
    
    def _adjust_profile_for_vram(self, profile: GPUProfile) -> GPUProfile:
        """Adjust profile if custom VRAM fraction is set"""
        if self.custom_vram_fraction is not None:
            profile.vram_fraction = self.custom_vram_fraction
            
            # Adjust batch size based on VRAM fraction
            if self.custom_vram_fraction < 0.6:
                profile.batch_size = max(1, profile.batch_size // 2)
                profile.num_frames = max(4, profile.num_frames // 2)
            elif self.custom_vram_fraction > 0.85:
                profile.batch_size = int(profile.batch_size * 1.2)
                
        return profile
    
    def _load_user_preferences(self):
        """Load user preferences from config file if exists"""
        config_path = os.path.join(os.path.dirname(__file__), "gpu_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    prefs = json.load(f)
                    
                # Override profile settings with user preferences
                if "batch_size" in prefs:
                    self.profile.batch_size = prefs["batch_size"]
                if "vram_fraction" in prefs:
                    self.profile.vram_fraction = prefs["vram_fraction"]
                if "inference_steps" in prefs:
                    self.profile.inference_steps = prefs["inference_steps"]
                    
                print(f"Loaded user preferences from {config_path}")
            except Exception as e:
                print(f"Could not load user preferences: {e}")
    
    def save_preferences(self, batch_size: Optional[int] = None, 
                        vram_fraction: Optional[float] = None,
                        inference_steps: Optional[int] = None):
        """Save user preferences to config file"""
        config_path = os.path.join(os.path.dirname(__file__), "gpu_config.json")
        
        prefs = {}
        if batch_size is not None:
            prefs["batch_size"] = batch_size
        if vram_fraction is not None:
            prefs["vram_fraction"] = vram_fraction
        if inference_steps is not None:
            prefs["inference_steps"] = inference_steps
            
        try:
            with open(config_path, 'w') as f:
                json.dump(prefs, f, indent=2)
            print(f"Saved preferences to {config_path}")
        except Exception as e:
            print(f"Could not save preferences: {e}")
    
    def get_config(self) -> Dict:
        """Get the current configuration as a dictionary"""
        return {
            "gpu_name": self.gpu_info["name"],
            "vram_gb": self.gpu_info["vram_gb"],
            "compute_capability": self.gpu_info["compute_capability"],
            "profile_name": self.profile.name,
            "batch_size": self.profile.batch_size,
            "num_frames": self.profile.num_frames,
            "inference_steps": self.profile.inference_steps,
            "vram_fraction": self.profile.vram_fraction,
            "use_mixed_precision": self.profile.use_mixed_precision,
            "enable_tf32": self.profile.enable_tf32,
            "optimizations": self.profile.enable_optimizations,
        }
    
    def apply_memory_limit(self):
        """Apply the memory limit based on profile"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.profile.vram_fraction)
            torch.cuda.empty_cache()
            
            # Set performance flags
            torch.backends.cudnn.benchmark = True
            if self.profile.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            print(f"Applied memory limit: {self.profile.vram_fraction:.1%} of {self.gpu_info['vram_gb']:.1f}GB")
    
    def get_optimal_settings(self, task: str = "inference") -> Dict:
        """Get optimal settings for a specific task"""
        settings = {
            "batch_size": self.profile.batch_size,
            "num_frames": self.profile.num_frames,
            "inference_steps": self.profile.inference_steps,
            "device": self.device,
            "dtype": torch.float16 if self.profile.use_mixed_precision else torch.float32,
            "enable_optimizations": self.profile.enable_optimizations,
        }
        
        # Adjust for specific tasks
        if task == "training":
            settings["batch_size"] = max(1, settings["batch_size"] // 2)
        elif task == "preview":
            settings["inference_steps"] = max(5, settings["inference_steps"] // 2)
            
        return settings
    
    def log_config(self):
        """Log the current configuration"""
        config = self.get_config()
        print("\n" + "="*60)
        print("GPU CONFIGURATION")
        print("="*60)
        print(f"GPU: {config['gpu_name']} ({config['vram_gb']:.1f}GB)")
        print(f"Compute Capability: {config['compute_capability']}")
        print(f"Profile: {config['profile_name']}")
        print(f"VRAM Usage: {config['vram_fraction']:.1%}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Frames per Batch: {config['num_frames']}")
        print(f"Inference Steps: {config['inference_steps']}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        print(f"TF32: {config['enable_tf32']}")
        print("Optimizations:")
        for opt, enabled in config['optimizations'].items():
            print(f"  - {opt}: {'✓' if enabled else '✗'}")
        print("="*60 + "\n")


# Convenience function for quick setup
def auto_configure_gpu(custom_vram_fraction: Optional[float] = None) -> AdaptiveGPUConfig:
    """Automatically configure GPU settings"""
    config = AdaptiveGPUConfig(custom_vram_fraction)
    config.apply_memory_limit()
    config.log_config()
    return config