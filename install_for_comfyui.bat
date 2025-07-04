@echo off
REM Installation script for ComfyUI LatentSync MEMSAFE (Windows)

echo ======================================================
echo  LatentSync MEMSAFE - ComfyUI Installation (Windows)
echo ======================================================
echo.

REM Check if we're in a conda environment
if defined CONDA_DEFAULT_ENV (
    echo ✅ Detected conda environment: %CONDA_DEFAULT_ENV%
) else (
    echo ⚠️  No conda environment detected. Make sure you've activated ComfyUI environment!
    echo    Run: conda activate comfyui
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo 📦 Installing required packages...
echo.

REM Install xformers for Flash Attention
echo 1️⃣ Installing xformers (Flash Attention)...
pip install xformers>=0.0.22

echo.
echo 2️⃣ Installing core dependencies...
pip install -r requirements.txt

echo.
echo 3️⃣ Checking PyTorch version...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

REM Check PyTorch version
for /f "tokens=*" %%i in ('python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')"') do set PYTORCH_VER=%%i

echo.
echo 4️⃣ Verifying xformers installation...
python -c "import xformers; print(f'✅ xformers version: {xformers.__version__}')" 2>nul || echo ❌ xformers not installed properly

echo.
echo 5️⃣ Running installation verification...
echo.
python verify_installation.py

echo.
echo ======================================================
echo ✅ Installation complete!
echo.
echo 📋 Next steps:
echo 1. Download the required models (see README)
echo 2. Restart ComfyUI
echo 3. Look for '🎬 LatentSync 1.6 (MEMSAFE)' in the node menu
echo.
echo 💡 Tips:
echo - Flash attention (xformers) is now the default
echo - The node will auto-detect and optimize for your GPU
echo - Check the console output for optimization details
echo.
pause