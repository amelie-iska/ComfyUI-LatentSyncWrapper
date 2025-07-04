"""Verify LatentSync MEMSAFE installation and optimization status"""

import torch
import sys
import os
from packaging import version

def check_pytorch():
    """Check PyTorch installation and version"""
    print("🔍 Checking PyTorch...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Check for FlexAttention support
    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        print("  ✅ FlexAttention supported (PyTorch 2.5+)")
    else:
        print("  ⚠️  FlexAttention not available (requires PyTorch 2.5+)")
    
    return torch.cuda.is_available()

def check_optimizations():
    """Check if optimizations are available"""
    print("\n🔍 Checking Optimizations...")
    
    # Check adaptive memory optimizer
    try:
        from adaptive_memory_optimizer import create_adaptive_optimizer
        optimizer = create_adaptive_optimizer()
        print(f"  ✅ Adaptive Memory Optimizer: {optimizer.profile.tier.value} profile")
    except Exception as e:
        print(f"  ❌ Adaptive Memory Optimizer: {e}")
    
    # Check FlexAttention
    try:
        from flex_attention import check_flex_attention, FLEX_ATTENTION_AVAILABLE
        if FLEX_ATTENTION_AVAILABLE:
            available, msg = check_flex_attention()
            if available:
                print(f"  ✅ FlexAttention: {msg}")
            else:
                print(f"  ⚠️  FlexAttention: {msg}")
        else:
            print("  ⚠️  FlexAttention: Not available")
    except Exception as e:
        print(f"  ❌ FlexAttention check failed: {e}")
    
    # Check xformers
    try:
        import xformers
        print(f"  ✅ xformers: {xformers.__version__}")
    except:
        print("  ⚠️  xformers: Not installed (optional)")
    
    # Check torch.compile
    if hasattr(torch, 'compile'):
        print("  ✅ torch.compile: Available")
    else:
        print("  ⚠️  torch.compile: Not available")

def check_dependencies():
    """Check required dependencies"""
    print("\n🔍 Checking Dependencies...")
    
    required = {
        'diffusers': None,
        'transformers': None,
        'accelerate': None,
        'einops': None,
        'omegaconf': None,
        'soundfile': None,
        'opencv-python': 'cv2',
        'ffmpeg-python': 'ffmpeg',
        'imageio': None,
        'psutil': None
    }
    
    missing = []
    for package, import_name in required.items():
        try:
            if import_name:
                __import__(import_name)
            else:
                __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    return missing

def check_models():
    """Check if required models are present"""
    print("\n🔍 Checking Models...")
    
    model_paths = {
        'UNet': 'checkpoints/latentsync_unet.pt',
        'Whisper': 'checkpoints/whisper/tiny.pt',
        'VAE': 'checkpoints/vae/sd-vae-ft-mse.safetensors'
    }
    
    missing_models = []
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"  ✅ {model_name}: {path} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {model_name}: {path} - NOT FOUND")
            missing_models.append(model_name)
    
    return missing_models

def check_memory_settings():
    """Check memory optimization settings"""
    print("\n🔍 Checking Memory Settings...")
    
    # Check PyTorch allocator config
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if alloc_conf:
        print(f"  ✅ PyTorch allocator config: {alloc_conf}")
    else:
        print("  ⚠️  PyTorch allocator config: Not set")
    
    # Check current memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  📊 Current GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

def main():
    """Run all checks"""
    print("=" * 60)
    print("🎬 LatentSync MEMSAFE Installation Verification")
    print("=" * 60)
    
    # Run checks
    cuda_available = check_pytorch()
    missing_deps = check_dependencies()
    missing_models = check_models()
    check_optimizations()
    check_memory_settings()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary:")
    
    issues = []
    if not cuda_available:
        issues.append("CUDA not available - GPU processing disabled")
    if missing_deps:
        issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
    if missing_models:
        issues.append(f"Missing models: {', '.join(missing_models)}")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n💡 To fix dependencies, run: pip install -r requirements.txt")
        print("💡 To download models, see the README.md for links")
    else:
        print("✅ All checks passed! LatentSync MEMSAFE is ready to use.")
        print("\n🚀 Optimizations available:")
        print("  - Adaptive memory management")
        print("  - FlexAttention (if PyTorch 2.5+)")
        print("  - Hardware video encoding")
        print("  - Smart batching")
    
    print("=" * 60)

if __name__ == "__main__":
    main()