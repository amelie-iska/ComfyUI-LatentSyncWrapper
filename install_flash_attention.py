#!/usr/bin/env python
"""Install Flash Attention (xformers) for LatentSync MEMSAFE"""

import subprocess
import sys
import os
import torch

def install_xformers():
    """Install xformers with proper CUDA version matching"""
    
    print("🚀 Installing Flash Attention (xformers) for LatentSync MEMSAFE")
    print("="*60)
    
    # Check current environment
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available!")
        return False
    
    # Determine CUDA version for xformers
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_major = int(cuda_version.split('.')[0])
        cuda_minor = int(cuda_version.split('.')[1])
        
        # Map to PyTorch index URLs
        if cuda_major == 11 and cuda_minor >= 8:
            index_url = "https://download.pytorch.org/whl/cu118"
            cuda_tag = "cu118"
        elif cuda_major == 12:
            index_url = "https://download.pytorch.org/whl/cu121"
            cuda_tag = "cu121"
        else:
            index_url = None
            cuda_tag = None
    
    print(f"\n📦 Installing xformers...")
    
    try:
        # First try to uninstall existing xformers
        print("Removing old xformers if exists...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "xformers", "-y"], 
                      capture_output=True)
        
        # Install xformers
        if index_url:
            print(f"Installing xformers for CUDA {cuda_tag}...")
            cmd = [sys.executable, "-m", "pip", "install", "xformers>=0.0.22", 
                   "--index-url", index_url, "--no-cache-dir"]
        else:
            print("Installing xformers (auto-detect CUDA)...")
            cmd = [sys.executable, "-m", "pip", "install", "xformers>=0.0.22", 
                   "--no-cache-dir"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ xformers installed successfully!")
        else:
            print("❌ Failed to install xformers")
            print("Error:", result.stderr)
            
            # Try alternative installation
            print("\n🔄 Trying alternative installation method...")
            cmd2 = [sys.executable, "-m", "pip", "install", "xformers", "--no-deps"]
            subprocess.run(cmd2)
            cmd3 = [sys.executable, "-m", "pip", "install", "xformers"]
            subprocess.run(cmd3)
            
    except Exception as e:
        print(f"❌ Error installing xformers: {e}")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import xformers
        print(f"✅ xformers {xformers.__version__} is installed!")
        
        # Test memory efficient attention
        from xformers.ops import memory_efficient_attention
        print("✅ Memory efficient attention is available!")
        
        return True
    except ImportError as e:
        print(f"❌ xformers not available: {e}")
        return False
    except Exception as e:
        print(f"⚠️  xformers installed but not fully functional: {e}")
        return False

def verify_diffusers_xformers():
    """Verify xformers works with diffusers"""
    try:
        from diffusers.models.attention_processor import XFormersAttnProcessor
        print("✅ Diffusers xformers support verified!")
        return True
    except:
        print("⚠️  Diffusers xformers support not available")
        return False

def main():
    """Main installation function"""
    print()
    success = install_xformers()
    
    if success:
        print("\n" + "="*60)
        print("✅ Flash Attention (xformers) installation complete!")
        print("\n📋 Next steps:")
        print("1. Restart ComfyUI")
        print("2. Flash attention will be automatically enabled")
        print("3. Check console for 'Enabled Flash/xformers memory efficient attention'")
        
        # Also verify diffusers compatibility
        print("\n🔍 Checking diffusers compatibility...")
        verify_diffusers_xformers()
    else:
        print("\n" + "="*60)
        print("❌ Installation failed. Please try manual installation:")
        print(f"pip install xformers --index-url https://download.pytorch.org/whl/cu{torch.version.cuda.replace('.', '')}")
    
    print("="*60)

if __name__ == "__main__":
    main()