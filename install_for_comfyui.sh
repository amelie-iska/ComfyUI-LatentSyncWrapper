#!/bin/bash
# Installation script for ComfyUI LatentSync MEMSAFE

echo "🎬 LatentSync MEMSAFE - ComfyUI Installation"
echo "==========================================="

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Detected conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No conda environment detected. Make sure you've activated ComfyUI environment!"
    echo "   Run: conda activate comfyui"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install required packages
echo ""
echo "📦 Installing required packages..."
echo ""

# Install xformers for Flash Attention (critical for memory efficiency)
echo "1️⃣ Installing xformers (Flash Attention)..."
pip install xformers>=0.0.22 --no-deps
pip install xformers>=0.0.22

# Install other requirements
echo ""
echo "2️⃣ Installing core dependencies..."
pip install -r requirements.txt

# Check PyTorch version for FlexAttention
echo ""
echo "3️⃣ Checking PyTorch version..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check if PyTorch 2.5+ is available
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
MAJOR_VERSION=$(echo $PYTORCH_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTORCH_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -ge 2 ] && [ "$MINOR_VERSION" -ge 5 ]; then
    echo "✅ PyTorch 2.5+ detected - FlexAttention will be available!"
else
    echo "⚠️  PyTorch version is $PYTORCH_VERSION - FlexAttention requires 2.5+"
    echo "   FlexAttention will fallback to standard attention."
    echo ""
    echo "   To upgrade PyTorch for FlexAttention support (optional):"
    echo "   pip install torch>=2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
fi

# Verify xformers installation
echo ""
echo "4️⃣ Verifying xformers installation..."
python -c "import xformers; print(f'✅ xformers version: {xformers.__version__}')" 2>/dev/null || echo "❌ xformers not installed properly"

# Run verification script
echo ""
echo "5️⃣ Running installation verification..."
echo ""
python verify_installation.py

echo ""
echo "==========================================="
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Download the required models (see README)"
echo "2. Restart ComfyUI"
echo "3. Look for '🎬 LatentSync 1.6 (MEMSAFE)' in the node menu"
echo ""
echo "💡 Tips:"
echo "- Flash attention (xformers) is now the default for better compatibility"
echo "- FlexAttention requires PyTorch 2.5+ but will gracefully fallback"
echo "- The node will auto-detect and optimize for your GPU"
echo ""