import os
import tempfile
import torchaudio
import uuid
import sys
import shutil
import time
from collections.abc import Mapping
from datetime import datetime
import gc
from .memory_limiter import limit_gpu_memory, clear_cache_periodically, log_memory_usage
from .inference_optimizer import optimized_inference_context, optimize_inference_pipeline, reduce_inference_lag
from .gpu_monitor import gpu_monitor
from .adaptive_gpu_config import AdaptiveGPUConfig, auto_configure_gpu
from .speed_optimizer import apply_speed_optimizations, dynamic_inference_steps, cuda_graphs_context, DeepCacheOptimizer
from .long_video_handler import LongVideoHandler, VideoLengthAdjuster, ProgressiveVideoProcessor
from .memory_optimizer import InferenceMemoryOptimizer, FrameBufferManager, optimize_end_stage_inference
try:
    from .adaptive_memory_optimizer import create_adaptive_optimizer, integrate_adaptive_optimizer
    ADAPTIVE_MEMORY_AVAILABLE = True
except ImportError:
    ADAPTIVE_MEMORY_AVAILABLE = False

# Apply GPU memory limit immediately after importing
limit_gpu_memory()


# Function to find ComfyUI directories
def get_comfyui_temp_dir():
    """Dynamically find the ComfyUI temp directory"""
    # First check using folder_paths if available
    try:
        import folder_paths
        comfy_dir = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        temp_dir = os.path.join(comfy_dir, "temp")
        return temp_dir
    except:
        pass
    
    # Try to locate based on current script location
    try:
        # This script is likely in a ComfyUI custom nodes directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up until we find the ComfyUI directory
        potential_dir = current_dir
        for _ in range(5):  # Limit to 5 levels up
            if os.path.exists(os.path.join(potential_dir, "comfy.py")):
                return os.path.join(potential_dir, "temp")
            potential_dir = os.path.dirname(potential_dir)
    except:
        pass
    
    # Return None if we can't find it
    return None

# Function to clean up any ComfyUI temp directories
def cleanup_comfyui_temp_directories():
    """Clean up only our LatentSync temp directories, not ComfyUI's main temp"""
    # Only clean up our module's temp directory, not ComfyUI's
    global MODULE_TEMP_DIR
    if 'MODULE_TEMP_DIR' in globals() and MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            # Only clean old run directories inside our temp
            for item in os.listdir(MODULE_TEMP_DIR):
                item_path = os.path.join(MODULE_TEMP_DIR, item)
                if os.path.isdir(item_path) and item.startswith("run_"):
                    try:
                        # Only remove if older than 1 hour
                        stat = os.stat(item_path)
                        age_hours = (time.time() - stat.st_mtime) / 3600
                        if age_hours > 1:
                            shutil.rmtree(item_path)
                            print(f"Cleaned old LatentSync temp: {item}")
                    except Exception as e:
                        # Silently skip directories in use
                        pass
        except Exception as e:
            # Don't fail if cleanup doesn't work
            pass

# Create a module-level function to set up system-wide temp directory
def init_temp_directories():
    """Initialize global temporary directory settings"""
    # Don't call cleanup here as MODULE_TEMP_DIR doesn't exist yet
    
    # Generate a unique base directory for this module
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    temp_base_path = os.path.join(system_temp, f"latentsync_{unique_id}")
    os.makedirs(temp_base_path, exist_ok=True)
    
    # Override environment variables that control temp directories
    os.environ['TMPDIR'] = temp_base_path
    os.environ['TEMP'] = temp_base_path
    os.environ['TMP'] = temp_base_path
    
    # Force Python's tempfile module to use our directory
    tempfile.tempdir = temp_base_path
    
    # Note: We no longer interfere with ComfyUI's temp directory
    # This was causing conflicts with other nodes
    
    print(f"Set up system temp directory: {temp_base_path}")
    return temp_base_path

# Function to clean up everything when the module exits
def module_cleanup():
    """Clean up all resources when the module is unloaded"""
    global MODULE_TEMP_DIR
    
    # Clean up our module temp directory
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up module temp directory: {MODULE_TEMP_DIR}")
        except:
            pass
    
    # Do a final sweep for any ComfyUI temp directories
    cleanup_comfyui_temp_directories()

# Call this before anything else
MODULE_TEMP_DIR = init_temp_directories()

# Now we can safely clean up old temp directories
cleanup_comfyui_temp_directories()

# Register the cleanup handler to run when Python exits
import atexit
atexit.register(module_cleanup)

# Now import regular dependencies
import math
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
from decimal import Decimal, ROUND_UP
import requests

# Modify folder_paths module to use our temp directory
if hasattr(folder_paths, "get_temp_directory"):
    original_get_temp = folder_paths.get_temp_directory
    folder_paths.get_temp_directory = lambda: MODULE_TEMP_DIR
else:
    # Add the function if it doesn't exist
    setattr(folder_paths, 'get_temp_directory', lambda: MODULE_TEMP_DIR)

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    # Check if auto-install is disabled via environment variable
    if os.environ.get("LATENTSYNC_NO_AUTO_INSTALL", "").lower() in ["true", "1", "yes"]:
        print("Auto-installation disabled. Please install dependencies manually:")
        print("pip install omegaconf pytorch_lightning transformers accelerate huggingface_hub einops diffusers ffmpeg-python")
        return

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers',
        'ffmpeg-python' 
    ]

    # Create a flag file to remember we've already installed packages
    user_home = os.path.expanduser("~")
    dependencies_installed_flag = os.path.join(user_home, ".latentsync16_dependencies_installed")
    
    # Skip installation if we've already done it before
    if os.path.exists(dependencies_installed_flag):
        print("Dependencies already installed from previous run")
        return
        
    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None

    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")

    missing_packages = []
    for package in required_packages:
        if not is_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing dependencies detected: {', '.join(missing_packages)}")
        print("Attempting automatic installation...")
        print("To disable auto-install, set environment variable: LATENTSYNC_NO_AUTO_INSTALL=true")
        
        for package in missing_packages:
            try:
                install_package(package)
            except Exception as e:
                print(f"Error: Failed to install {package}: {str(e)}")
                print(f"Please install manually: pip install {package}")
                raise RuntimeError(f"Required package '{package}' is not installed")
    else:
        print("All dependencies are already installed")
    
    # Create flag file to remember we've installed packages
    try:
        with open(dependencies_installed_flag, "w") as f:
            f.write("Dependencies installed on " + str(datetime.now()))
        print("Recorded dependencies installation status")
    except:
        print("Failed to create dependencies flag file - might reinstall next time")

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    """Get extension directory path, optionally with a subpath"""
    # Get the directory containing this script
    dir = os.path.dirname(os.path.abspath(__file__))
    
    # Special case for temp directories
    if subpath and ("temp" in subpath.lower() or "tmp" in subpath.lower()):
        # Use our global temp directory instead
        global MODULE_TEMP_DIR
        sub_temp = os.path.join(MODULE_TEMP_DIR, subpath)
        if mkdir and not os.path.exists(sub_temp):
            os.makedirs(sub_temp, exist_ok=True)
        return sub_temp
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

def download_model(url, save_path):
    """Download a model from a URL and save it to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def pre_download_models():
    """Pre-download all required models and cache them properly."""
    models = {
        "s3fd-e19a316812.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/hub/checkpoints/s3fd-619a316812.pth",
        # Add other models here
    }

    # Use a persistent location for model cache instead of temporary directory
    # This ensures models are downloaded only once across all runs
    user_home = os.path.expanduser("~")
    persistent_cache_dir = os.path.join(user_home, ".latentsync16_models")
    os.makedirs(persistent_cache_dir, exist_ok=True)
    
    for model_name, url in models.items():
        save_path = os.path.join(persistent_cache_dir, model_name)
        if not os.path.exists(save_path):
            print(f"Model {model_name} not found in cache. Downloading...")
            try:
                download_model(url, save_path)
                print(f"Successfully downloaded {model_name} to {save_path}")
            except Exception as e:
                print(f"Error downloading {model_name}: {str(e)}")
                print(f"You may need to download it manually from {url}")
        else:
            print(f"Model {model_name} already exists in cache at {save_path}")
    
    # Return the cache directory so we can use it later
    return persistent_cache_dir

def get_latentsync_config_path(cur_dir):
    """Automatically detect the best config file for LatentSync version"""
    # Try 1.6 config first (512x512)
    config_512 = os.path.join(cur_dir, "configs", "unet", "stage2_512.yaml")
    if os.path.exists(config_512):
        print("Using LatentSync 1.6 config (512x512)")
        return config_512
    
    # Fallback to 1.5 config (256x256)
    config_256 = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
    if os.path.exists(config_256):
        print("Using LatentSync 1.5 config (256x256)")
        return config_256
    
    # If neither exists, default to 1.6
    print("Config files not found, defaulting to LatentSync 1.6 config")
    return config_512

def setup_models():
    """Setup and pre-download all required models."""
    # Pre-download additional models to a persistent location
    persistent_cache_dir = pre_download_models()

    # Existing setup logic for LatentSync models
    cur_dir = get_ext_dir()
    ckpt_dir = os.path.join(cur_dir, "checkpoints")
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    # Create a temp_downloads directory in our system temp
    temp_downloads = os.path.join(MODULE_TEMP_DIR, "downloads")
    os.makedirs(temp_downloads, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    # Check if models exist in the persistent cache first and copy them if needed
    cache_unet_path = os.path.join(persistent_cache_dir, "latentsync_unet.pt")
    cache_whisper_path = os.path.join(persistent_cache_dir, "whisper/tiny.pt")
    
    if os.path.exists(cache_unet_path) and not os.path.exists(unet_path):
        print(f"Copying unet model from cache {cache_unet_path} to {unet_path}")
        shutil.copy2(cache_unet_path, unet_path)
    
    if os.path.exists(cache_whisper_path) and not os.path.exists(whisper_path):
        print(f"Copying whisper model from cache {cache_whisper_path} to {whisper_path}")
        os.makedirs(os.path.dirname(whisper_path), exist_ok=True)
        shutil.copy2(cache_whisper_path, whisper_path)

    # Only download if models aren't in the working directory and weren't in the cache
    if not (os.path.exists(unet_path) and os.path.exists(whisper_path)):
        print("Downloading required LatentSync 1.6 model checkpoints... This may take a while.")
        try:
            from huggingface_hub import snapshot_download
            
            # Download to the persistent cache first
            snapshot_download(repo_id="ByteDance/LatentSync-1.6",
                            allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                            local_dir=persistent_cache_dir, 
                            local_dir_use_symlinks=False,
                            cache_dir=temp_downloads)
            
            # Then copy to the working directory if needed
            if not os.path.exists(unet_path) and os.path.exists(os.path.join(persistent_cache_dir, "latentsync_unet.pt")):
                shutil.copy2(os.path.join(persistent_cache_dir, "latentsync_unet.pt"), unet_path)
            
            cache_whisper_tiny = os.path.join(persistent_cache_dir, "whisper/tiny.pt")
            if not os.path.exists(whisper_path) and os.path.exists(cache_whisper_tiny):
                os.makedirs(os.path.dirname(whisper_path), exist_ok=True)
                shutil.copy2(cache_whisper_tiny, whisper_path)
                
            print("LatentSync 1.6 model checkpoints downloaded successfully!")
        except Exception as e:
            print(f"Error downloading models: {str(e)}")
            print("\nPlease download models manually:")
            print("1. Visit: https://huggingface.co/ByteDance/LatentSync-1.6")
            print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
            print(f"3. Place them in: {ckpt_dir}")
            print(f"   with whisper/tiny.pt in: {whisper_dir}")
            raise RuntimeError("Model download failed. See instructions above.")

class LatentSyncNode:
    def __init__(self):
        # Make sure our temp directory is the current one
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)
        
        # Note: We no longer interfere with ComfyUI's temp directory
        # This was causing conflicts with other nodes
        
        check_and_install_dependencies()
        setup_models()
        
        # Initialize adaptive GPU configuration
        self.gpu_config = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                    "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                    "vram_fraction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05, "display": "slider"}),
                    "optimization_level": (["conservative", "balanced", "aggressive"], {"default": "balanced"}),
                    "memory_mode": (["aggressive", "balanced", "conservative"], {"default": "balanced"}),
                    "enable_disk_cache": ("BOOLEAN", {"default": False}),
                    "return_frames": ("BOOLEAN", {"default": False, "description": "Return frames (True) or video path (False)"}),
                    "output_mode": (["auto", "video_file", "frames"], {"default": "auto", "description": "auto: choose based on video length"}),
                    "attention_mode": (["flash", "flex", "xformers", "standard"], {"default": "flash", "description": "Attention mechanism to use"}),
                 },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path") 
    FUNCTION = "inference"

    def process_batch(self, batch, use_mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def inference(self, images, audio, seed, lips_expression=1.5, inference_steps=20, vram_fraction=0.0, optimization_level="balanced", memory_mode="balanced", enable_disk_cache=False, return_frames=True, output_mode="auto", attention_mode="flash"):
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
        # Initialize variables that might be used in error handlers
        total_frames = 0
        width = 0
        height = 0
        estimated_memory = 0.0
        free_memory = 0.0
        should_return_frames = return_frames  # Initialize early for finally block
        
        # Initialize adaptive GPU configuration
        # Use custom vram_fraction if set (0.0 means auto)
        custom_vram = vram_fraction if vram_fraction > 0 else None
        
        # Create or update GPU configuration
        if self.gpu_config is None or custom_vram != getattr(self.gpu_config, 'custom_vram_fraction', None):
            self.gpu_config = AdaptiveGPUConfig(custom_vram_fraction=custom_vram)
            self.gpu_config.apply_memory_limit()
            self.gpu_config.log_config()
        
        # Get optimal settings from GPU config
        gpu_settings = self.gpu_config.get_optimal_settings(task="inference")
        
        # Extract settings
        device = gpu_settings["device"]
        BATCH_SIZE = gpu_settings["batch_size"]
        use_mixed_precision = self.gpu_config.profile.use_mixed_precision
        enable_tf32 = self.gpu_config.profile.enable_tf32
        
        # Override inference steps if user specified
        if inference_steps != 20:  # If not default
            gpu_settings["inference_steps"] = inference_steps
        else:
            inference_steps = gpu_settings["inference_steps"]
        
        # Log current settings
        print(f"\nUsing adaptive GPU settings:")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Inference Steps: {inference_steps}")
        print(f"  VRAM Fraction: {self.gpu_config.profile.vram_fraction:.1%}")
        print(f"  Mixed Precision: {use_mixed_precision}")
        
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_memory_usage("After adaptive configuration")


        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        # Ensure temp_dir is within our module directory (prevent path traversal)
        temp_dir = os.path.abspath(temp_dir)
        if not temp_dir.startswith(os.path.abspath(MODULE_TEMP_DIR)):
            raise ValueError("Invalid temp directory path")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Note: We no longer interfere with ComfyUI's temp directory
        
        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"latentsync_{run_id}_audio.wav")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check if we should use disk-based processing for long videos
            num_frames = len(images) if isinstance(images, list) else images.shape[0]
            use_disk_processing = False
            long_video_handler = None
            
            if enable_disk_cache and memory_mode != "aggressive":
                long_video_handler = LongVideoHandler(temp_dir, memory_mode)
                use_disk_processing = long_video_handler.should_use_disk_processing(num_frames)
                
                if use_disk_processing:
                    print(f"Using disk-based processing for {num_frames} frames (memory_mode: {memory_mode})")
            
            # Process input frames entirely on CPU to avoid unnecessary GPU
            # memory usage before inference
            if isinstance(images, list):
                frames_cpu = torch.stack(images).cpu()
            else:
                frames_cpu = images.cpu()

            frames_uint8 = (frames_cpu * 255).to(torch.uint8)
            del frames_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process audio with device awareness
            waveform = audio["waveform"].to(device)
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            if sample_rate != 16000:
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                ).to(device)
                waveform_16k = resampler(waveform)
                waveform, sample_rate = waveform_16k, new_sample_rate

            # Package resampled audio (ensure waveform is on CPU)
            resampled_audio = {
                "waveform": waveform.unsqueeze(0).cpu(),
                "sample_rate": sample_rate
            }
            
            # Move waveform to CPU for saving
            waveform_cpu = waveform.cpu()
            torchaudio.save(audio_path, waveform_cpu, sample_rate)
            del waveform_cpu
            gc.collect()

            # Write video frames
            try:
                import torchvision.io as io
                io.write_video(temp_video_path, frames_uint8, fps=25, video_codec='h264')
            except TypeError as e:
                # Check if the error is specifically about macro_block_size
                if "macro_block_size" in str(e):
                    import imageio
                    # Use imageio with macro_block_size parameter
                    imageio.mimsave(temp_video_path, frames_uint8.numpy(), fps=25, codec='h264', macro_block_size=1)
                else:
                    # Fall back to original PyAV code for other TypeError issues
                    import av
                    container = av.open(temp_video_path, mode='w')
                    stream = container.add_stream('h264', rate=25)
                    stream.width = frames_uint8.shape[2]
                    stream.height = frames_uint8.shape[1]

                    for frame in frames_uint8:
                        frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                        packet = stream.encode(frame)
                        container.mux(packet)

                    packet = stream.encode(None)
                    container.mux(packet)
                    container.close()

            del frames_uint8
            gc.collect()
            clear_cache_periodically()
            log_memory_usage("After video encoding")

            # Define paths to required files and configs
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = get_latentsync_config_path(cur_dir)
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(cur_dir, "checkpoints", "latentsync_unet.pt")
            whisper_ckpt_path = os.path.join(cur_dir, "checkpoints", "whisper", "tiny.pt")

            # Create config and args
            config = OmegaConf.load(config_path)
            
            # Override config with adaptive GPU settings
            if hasattr(config, 'data'):
                # Use adaptive num_frames setting
                optimal_num_frames = gpu_settings.get('num_frames', 8)
                if hasattr(config.data, 'num_frames'):
                    # For high-end GPUs with custom VRAM, don't reduce num_frames
                    if custom_vram and self.gpu_config.gpu_info["vram_gb"] >= 20:
                        optimal_num_frames = 16  # Use full 16 frames for RTX 4090/3090
                    if config.data.num_frames != optimal_num_frames:
                        print(f"Setting num_frames to {optimal_num_frames} based on GPU profile")
                        config.data.num_frames = optimal_num_frames
                
                # Use adaptive batch size
                if hasattr(config.data, 'batch_size'):
                    config.data.batch_size = 1  # Keep at 1 for pipeline compatibility

            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            # Make sure the mask image exists
            if not os.path.exists(mask_image_path):
                # Try to find it in the utils directory directly
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                else:
                    print(f"Warning: Could not find mask image at expected locations")

            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path

            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,  # Using lips_expression for the guidance_scale
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=BATCH_SIZE,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                mask_image_path=mask_image_path,
                enable_optimizations=self.gpu_config.profile.enable_optimizations,
                attention_mode=attention_mode  # Pass attention mode to pipeline
            )

            # Set PYTHONPATH to include our directories 
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Clean GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Note: We no longer interfere with ComfyUI's temp directory

            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
            
            # Monkey patch any temp directory functions in the inference module
            if hasattr(inference_module, 'get_temp_dir'):
                inference_module.get_temp_dir = lambda *args, **kwargs: temp_dir
                
            # Create subdirectories that the inference module might expect
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            # Apply speed optimizations based on user preference and GPU capability
            print(f"Using optimization level: {optimization_level}")
            
            if custom_vram is None or self.gpu_config.gpu_info["vram_gb"] < 12:
                config, args = reduce_inference_lag(config, args)
                # Low VRAM GPUs should use conservative optimization
                if optimization_level == "aggressive":
                    optimization_level = "balanced"
                    print("Downgrading to balanced optimization for GPU with <12GB VRAM")
            else:
                print("Using adaptive GPU configuration for high-end GPU - keeping optimized settings")
                # For RTX 4090, allow higher batch sizes
                if "4090" in self.gpu_config.gpu_info["name"]:
                    if optimization_level == "aggressive":
                        args.batch_size = min(20, args.batch_size)  # Allow up to 20 for 4090 aggressive
                    else:
                        args.batch_size = min(16, args.batch_size)  # Allow up to 16 for 4090 otherwise
                    
                # Apply dynamic inference step reduction based on optimization level
                if optimization_level != "conservative":
                    original_steps = args.inference_steps
                    args.inference_steps = dynamic_inference_steps(original_steps, optimization_level)
                    print(f"Optimized inference steps: {original_steps} -> {args.inference_steps}")
            
            # Log memory before inference
            log_memory_usage("Before inference")
            
            # Improvement: Add estimated processing time
            if num_frames > 100:
                estimated_time = (num_frames / 25.0) * 2  # Rough estimate: 2x realtime
                print(f"Estimated processing time: {estimated_time:.1f} seconds for {num_frames} frames")
            
            # Initialize memory optimizer for end-stage lag prevention
            memory_optimizer = InferenceMemoryOptimizer(
                memory_mode=memory_mode,
                enable_disk_cache=enable_disk_cache
            )
            
            # Apply adaptive memory optimization if available
            if ADAPTIVE_MEMORY_AVAILABLE and hasattr(inference_module, 'pipeline'):
                try:
                    inference_module.pipeline = integrate_adaptive_optimizer(
                        inference_module.pipeline,
                        video_length=num_frames,
                        resolution=(width, height)
                    )
                    print("✓ Adaptive memory optimization applied to pipeline")
                except Exception as e:
                    print(f"Could not apply adaptive optimization: {e}")
            
            # Calculate total iterations for optimization
            total_iterations = int(num_frames / 25.0 * inference_steps)  # Approximate
            
            # Start GPU monitoring to diagnose lag
            gpu_monitor.start()
            
            try:
                # Run inference with optimized context and gradient tracking disabled
                with optimized_inference_context(device=device):
                    with torch.inference_mode():
                        # Check if pipeline can be optimized with adaptive settings
                        if hasattr(inference_module, 'pipeline'):
                            optimizations = self.gpu_config.profile.enable_optimizations
                            inference_module.pipeline = optimize_inference_pipeline(
                                inference_module.pipeline, 
                                enable_optimizations=optimizations
                            )
                            
                            # Apply end-stage optimization to prevent lag
                            inference_module.pipeline = optimize_end_stage_inference(
                                inference_module.pipeline,
                                num_iterations=total_iterations,
                                memory_mode=memory_mode,
                                enable_disk_cache=enable_disk_cache
                            )
                            
                            # Apply additional speed optimizations for high-end GPUs
                            if optimization_level != "conservative":
                                inference_module.pipeline, speed_opts = apply_speed_optimizations(
                                    inference_module.pipeline,
                                    self.gpu_config.gpu_info,
                                    optimization_level
                                )
                                print(f"Applied speed optimizations: {speed_opts}")
                        
                        # Use CUDA graphs for RTX 4090 with aggressive optimization
                        if optimization_level == "aggressive" and "4090" in self.gpu_config.gpu_info["name"]:
                            with cuda_graphs_context():
                                inference_module.main(config, args)
                        else:
                            inference_module.main(config, args)
            finally:
                # Stop GPU monitoring
                gpu_monitor.stop()

            # Clean up models from inference module
            if hasattr(inference_module, 'pipeline'):
                if hasattr(inference_module.pipeline, 'unet'):
                    del inference_module.pipeline.unet
                if hasattr(inference_module.pipeline, 'vae'):
                    del inference_module.pipeline.vae
                if hasattr(inference_module.pipeline, 'audio_encoder'):
                    del inference_module.pipeline.audio_encoder
                del inference_module.pipeline
            # Fix: Delete the inference_module itself to prevent memory leak
            del inference_module

            # Force cleanup
            gc.collect()
            
            # Clean GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                clear_cache_periodically()
                log_memory_usage("After inference")

            # Verify output file exists
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # Determine output mode
            should_return_frames = return_frames  # Default to user setting
            
            # Auto mode logic
            if output_mode == "auto":
                # Get video info to determine frame count
                import ffmpeg
                probe = ffmpeg.probe(output_video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                total_frames = int(video_info['nb_frames'])
                
                # Auto threshold based on memory mode
                frame_thresholds = {
                    "aggressive": 300,
                    "balanced": 200, 
                    "conservative": 100
                }
                threshold = frame_thresholds.get(memory_mode, 200)
                
                # If video is too long, force file mode
                if total_frames > threshold:
                    print(f"Auto mode: Video has {total_frames} frames (>{threshold}), returning video file")
                    should_return_frames = False
                else:
                    print(f"Auto mode: Video has {total_frames} frames (<={threshold}), returning frames")
                    should_return_frames = True
            elif output_mode == "video_file":
                should_return_frames = False
            elif output_mode == "frames":
                should_return_frames = True
                
            # If not returning frames, create a permanent copy and return just the path
            if not should_return_frames:
                # Create a permanent output location
                output_dir = os.path.join(get_ext_dir(), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate a unique filename for the permanent output
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize filename to prevent path injection
                safe_timestamp = "".join(c for c in timestamp if c.isalnum() or c == "_")
                permanent_output_path = os.path.join(output_dir, f"latentsync_output_{safe_timestamp}.mp4")
                # Ensure output path is within output directory
                permanent_output_path = os.path.abspath(permanent_output_path)
                if not permanent_output_path.startswith(os.path.abspath(output_dir)):
                    raise ValueError("Invalid output path")
                
                # Copy the output file to the permanent location
                shutil.copy2(output_video_path, permanent_output_path)
                
                print(f"Video saved to: {permanent_output_path}")
                
                # Return empty tensor for frames and the path as a string
                empty_frames = torch.zeros(1, 1, 1, 3).cpu()  # Minimal tensor on CPU
                
                return (empty_frames, resampled_audio, permanent_output_path)
            
            # Read the processed video with memory-efficient batching
            print("Loading processed video frames...")
            
            # Use ffmpeg-python to get video info first
            import ffmpeg
            probe = ffmpeg.probe(output_video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            total_frames = int(video_info['nb_frames'])
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Improvement: Pre-check memory requirements
            if torch.cuda.is_available():
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
                estimated_memory = (total_frames * width * height * 3 * 4) / 1024**3  # GB (float32)
                if estimated_memory > free_memory * 0.8:
                    print(f"Warning: Video requires ~{estimated_memory:.1f}GB but only {free_memory:.1f}GB available")
                    print("Automatically switching to disk-based loading...")
            
            # Check if we should use disk-based loading for large videos
            use_disk_loading = False
            # Improvement: Auto-enable disk loading if memory is tight
            if torch.cuda.is_available() and 'estimated_memory' in locals() and 'free_memory' in locals() and estimated_memory > free_memory * 0.8:
                use_disk_loading = True
            elif total_frames > 200 or (enable_disk_cache and memory_mode == "conservative"):
                use_disk_loading = True
                print(f"Using disk-based frame loading for {total_frames} frames")
            
            # Initialize long_video_handler if needed for output loading
            if use_disk_loading and not long_video_handler:
                long_video_handler = LongVideoHandler(temp_dir, memory_mode)
            
            if use_disk_loading and long_video_handler:
                # Use LongVideoHandler for disk-based loading
                frame_dir, metadata = long_video_handler.extract_frames_to_disk(output_video_path)
                
                # Determine optimal batch size based on memory mode
                batch_sizes = {
                    "aggressive": 32,
                    "balanced": 16,
                    "conservative": 8
                }
                batch_size = batch_sizes.get(memory_mode, 16)
                
                processed_frames_list = []
                for start_idx, end_idx in long_video_handler.process_in_chunks(total_frames, batch_size):
                    batch_frames = long_video_handler.load_frame_batch(start_idx, end_idx - start_idx)
                    processed_frames_list.append(batch_frames)
                    
                    # Progress update
                    progress = (end_idx / total_frames) * 100
                    print(f"Loading frames: {progress:.1f}%")
                    
                    # Memory cleanup between batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Concatenate all frames
                processed_frames = torch.cat(processed_frames_list, dim=0)
                del processed_frames_list
                
                # Clean up extracted frames
                long_video_handler.cleanup()
                
            else:
                # Original batched loading for smaller videos
                # Improvement: Dynamic batch size based on available memory
                if ADAPTIVE_MEMORY_AVAILABLE:
                    try:
                        adaptive_opt = create_adaptive_optimizer()
                        batch_size = adaptive_opt.get_optimal_batch_size('frame_loading')
                        print(f"Using adaptive batch size: {batch_size} frames")
                    except:
                        # Fallback to original calculation
                        if torch.cuda.is_available():
                            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                            frame_size_mb = (width * height * 3 * 4) / (1024**2)
                            safe_batch_size = int((free_gb * 1024 * 0.5) / frame_size_mb)
                            batch_size = min(max(safe_batch_size, 10), 100)
                        else:
                            batch_size = 50
                else:
                    # Original calculation
                    if torch.cuda.is_available():
                        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                        frame_size_mb = (width * height * 3 * 4) / (1024**2)
                        safe_batch_size = int((free_gb * 1024 * 0.5) / frame_size_mb)
                        batch_size = min(max(safe_batch_size, 10), 100)
                    else:
                        batch_size = 50
                
                print(f"Using batch size: {batch_size} frames")
                processed_frames_list = []
                
                # Try to use decord for efficient video reading if available
                vr = None  # Initialize for cleanup
                try:
                    from decord import VideoReader
                    from decord import cpu
                    vr = VideoReader(output_video_path, ctx=cpu(0))
                    
                    for start_idx in range(0, total_frames, batch_size):
                        end_idx = min(start_idx + batch_size, total_frames)
                        batch_frames = vr.get_batch(list(range(start_idx, end_idx)))
                        batch_frames = torch.from_numpy(batch_frames.asnumpy()).float() / 255.0
                        processed_frames_list.append(batch_frames)
                        
                        # Clear memory periodically
                        if start_idx + batch_size < total_frames:
                            gc.collect()
                    
                    processed_frames = torch.cat(processed_frames_list, dim=0)
                    del processed_frames_list
                    del vr  # Cleanup VideoReader
                    
                except ImportError:
                    # Fallback to ffmpeg extraction
                    print("Using ffmpeg for video loading...")
                    
                    # Create a temporary frames directory
                    frames_temp_dir = os.path.join(temp_dir, "extracted_frames")
                    os.makedirs(frames_temp_dir, exist_ok=True)
                    
                    # Extract frames using ffmpeg
                    (
                        ffmpeg
                        .input(output_video_path)
                        .output(os.path.join(frames_temp_dir, "frame_%04d.png"), start_number=0)
                        .run(quiet=True, overwrite_output=True)
                    )
                    
                    # Load frames in batches
                    frame_files = sorted([f for f in os.listdir(frames_temp_dir) if f.endswith('.png')])
                    
                    for i in range(0, len(frame_files), batch_size):
                        batch_files = frame_files[i:i+batch_size]
                        batch_frames = []
                        
                        for frame_file in batch_files:
                            frame_path = os.path.join(frames_temp_dir, frame_file)
                            frame = Image.open(frame_path)
                            frame_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
                            frame.close()  # Fix: Close PIL Image to prevent memory leak
                            batch_frames.append(frame_tensor)
                        
                        batch_tensor = torch.stack(batch_frames)
                        processed_frames_list.append(batch_tensor)
                        
                        # Clear memory
                        del batch_frames
                        gc.collect()
                    
                    processed_frames = torch.cat(processed_frames_list, dim=0)
                    del processed_frames_list
                    
                    # Clean up extracted frames
                    shutil.rmtree(frames_temp_dir, ignore_errors=True)

            # Ensure all tensors are on CPU before returning
            if torch.cuda.is_available():
                # Move audio waveform to CPU
                if isinstance(resampled_audio.get("waveform"), torch.Tensor):
                    if resampled_audio["waveform"].is_cuda:
                        resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                
                # Move processed frames to CPU
                if isinstance(processed_frames, torch.Tensor) and processed_frames.is_cuda:
                    processed_frames = processed_frames.cpu()

            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Successfully loaded {processed_frames.shape[0]} frames")
            return (processed_frames, resampled_audio, "")

        except torch.cuda.OutOfMemoryError as e:
            # Improvement: Better error message for OOM
            print(f"GPU OUT OF MEMORY ERROR!")
            print(f"Video info: {total_frames} frames at {width}x{height}")
            print(f"Try these solutions:")
            print(f"1. Set output_mode='auto' or 'video_file'")
            print(f"2. Enable disk_cache=True")
            print(f"3. Use memory_mode='conservative'")
            print(f"4. Reduce vram_fraction to 0.6-0.7")
            raise RuntimeError("GPU memory exhausted. See suggestions above.") from e
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Clean up temporary files individually
            # Don't delete the permanent output if we're returning a video file
            paths_to_clean = [temp_video_path, audio_path]
            if should_return_frames:
                # Only clean output_video_path if we're returning frames
                paths_to_clean.append(output_video_path)
            
            for path in paths_to_clean:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed temporary file: {path}")
                    except Exception as e:
                        print(f"Failed to remove {path}: {str(e)}")

            # Remove temporary run directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Removed run temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Failed to remove temp run directory: {str(e)}")

            # Clean up any ComfyUI temp directories again (in case they were created during execution)
            cleanup_comfyui_temp_directories()

            # Final GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# class VideoLengthAdjuster:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "images": ("IMAGE",),
#                 "audio": ("AUDIO",),
#                 "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
#                 "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
#                 "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
#             }
#         }
# 
#     CATEGORY = "LatentSyncNode"
#     RETURN_TYPES = ("IMAGE", "AUDIO")
#     RETURN_NAMES = ("images", "audio")
#     FUNCTION = "adjust"
# 
#     def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
#         waveform = audio["waveform"].squeeze(0)
#         sample_rate = int(audio["sample_rate"])
#         original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()
# 
#         if mode == "normal":
#             # Add silent padding to the audio and then trim video to match
#             audio_duration = waveform.shape[1] / sample_rate
#             
#             # Add silent padding to the audio
#             silence_samples = math.ceil(silent_padding_sec * sample_rate)
#             silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
#             padded_audio = torch.cat([waveform, silence], dim=1)
#             
#             # Calculate required frames based on the padded audio
#             padded_audio_duration = (waveform.shape[1] + silence_samples) / sample_rate
#             required_frames = int(padded_audio_duration * fps)
#             
#             if len(original_frames) > required_frames:
#                 # Trim video frames to match padded audio duration
#                 adjusted_frames = original_frames[:required_frames]
#             else:
#                 # If video is shorter than padded audio, keep all video frames
#                 # and trim the audio accordingly
#                 adjusted_frames = original_frames
#                 required_samples = int(len(original_frames) / fps * sample_rate)
#                 padded_audio = padded_audio[:, :required_samples]
#             
#             return (
#                 torch.stack(adjusted_frames),
#                 {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
#             )
#             
#             # This return statement is no longer needed as it's handled in the updated code
# 
#         elif mode == "pingpong":
#             video_duration = len(original_frames) / fps
#             audio_duration = waveform.shape[1] / sample_rate
#             if audio_duration <= video_duration:
#                 required_samples = int(video_duration * sample_rate)
#                 silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
#                 adjusted_audio = torch.cat([waveform, silence], dim=1)
# 
#                 return (
#                     torch.stack(original_frames),
#                     {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
#                 )
# 
#             else:
#                 silence_samples = math.ceil(silent_padding_sec * sample_rate)
#                 silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
#                 padded_audio = torch.cat([waveform, silence], dim=1)
#                 total_duration = (waveform.shape[1] + silence_samples) / sample_rate
#                 target_frames = math.ceil(total_duration * fps)
#                 reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
#                 frames = original_frames + reversed_frames
#                 while len(frames) < target_frames:
#                     frames += frames[:target_frames - len(frames)]
#                 return (
#                     torch.stack(frames[:target_frames]),
#                     {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
#                 )
# 
#         elif mode == "loop_to_audio":
#             # Add silent padding then simple loop
#             silence_samples = math.ceil(silent_padding_sec * sample_rate)
#             silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
#             padded_audio = torch.cat([waveform, silence], dim=1)
#             total_duration = (waveform.shape[1] + silence_samples) / sample_rate
#             target_frames = math.ceil(total_duration * fps)
# 
#             frames = original_frames.copy()
#             while len(frames) < target_frames:
#                 frames += original_frames[:target_frames - len(frames)]
#             
#             return (
#                 torch.stack(frames[:target_frames]),
#                 {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
#             )
# 
# 
# 
class LatentSyncVideoPathNode(LatentSyncNode):
    """A version of LatentSyncNode that returns video path instead of frames to avoid memory issues"""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get parent inputs and modify for video path output
        inputs = super().INPUT_TYPES()
        # Remove return_frames and output_mode since we force them
        if "return_frames" in inputs["required"]:
            del inputs["required"]["return_frames"]
        if "output_mode" in inputs["required"]:
            del inputs["required"]["output_mode"]
        return inputs
    
    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("video_path", "audio")
    FUNCTION = "inference_path"
    
    def inference_path(self, images, audio, seed, lips_expression=1.5, inference_steps=20, vram_fraction=0.0, optimization_level="balanced", memory_mode="balanced", enable_disk_cache=False, output_mode="video_file"):
        # Call parent inference with output_mode="video_file" to ensure we get a path
        _, audio_output, video_path = super().inference(
            images, audio, seed, lips_expression, inference_steps, 
            vram_fraction, optimization_level, memory_mode, enable_disk_cache, 
            return_frames=False, output_mode="video_file"
        )
        
        return (video_path, audio_output)

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode,
    # Removed LatentSyncVideoPathNode - main node now outputs video_path directly
}

# Display Names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "🎭 LatentSync 1.6 (MEMSAFE)",
    # Removed redundant video path node
}