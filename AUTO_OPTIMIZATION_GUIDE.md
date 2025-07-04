# LatentSync Auto-Optimization Guide

LatentSync 1.6 now includes intelligent auto-optimization that automatically configures the best settings for your specific GPU and use case.

## 🚀 Quick Start

### Step 1: Benchmark Your GPU (One-Time Setup)
Run the benchmark tool to analyze your GPU:

```bash
cd ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
python benchmark_my_gpu.py
```

This will:
- Test various speed/quality configurations on your GPU
- Save optimal settings for automatic use
- Generate a performance report
- Takes only 1-2 minutes (use `--full` for comprehensive test)

### Step 2: Use Auto Mode
In the LatentSync node, simply set:
- **Quality Preset**: `auto`

That's it! The node will automatically select optimal settings based on:
- Your GPU capabilities
- Video length
- Available VRAM

## 📊 Benchmark Options

### Quick Benchmark (Default)
```bash
python benchmark_my_gpu.py
```
- Tests key configurations
- Takes 1-2 minutes
- Sufficient for most users

### Full Benchmark
```bash
python benchmark_my_gpu.py --full
```
- Tests all possible configurations
- Takes 5-10 minutes
- More accurate results

### Clear Cache & Re-benchmark
```bash
python benchmark_my_gpu.py --clear
```
- Removes cached results
- Useful after driver updates

## 🤖 Auto-Optimization Features

### 1. Automatic Configuration
When using `quality_preset: auto`, the system will:
- Detect your GPU model and VRAM
- Load cached benchmark results
- Analyze video length
- Select optimal speed_mode, quality settings, and DeepCache

### 2. Video Length Adaptation
Different settings for different video lengths:
- **Short videos (<5s)**: Prioritize quality, disable DeepCache
- **Medium videos (5-20s)**: Balanced settings
- **Long videos (>20s)**: Aggressive optimization, enable all speedups

### 3. GPU-Specific Optimization
Tailored settings based on your hardware:
- **RTX 4090 (24GB)**: Ultra mode support, large batches
- **RTX 4070/3080 (12GB)**: Turbo mode, medium batches
- **RTX 4060/3070 (8GB)**: Fast mode, conservative settings
- **Lower-end GPUs**: Safe defaults, no DeepCache

## 🎯 New Nodes

### 1. LatentSync Auto-Optimize Node
Provides fine control over auto-optimization:
- **Optimization Goal**: preview/balanced/quality/auto
- **Video Length**: Specify exact length for better optimization
- **Run Benchmark**: Re-run benchmark from within ComfyUI

### 2. LatentSync GPU Benchmark Node
Advanced benchmarking within ComfyUI:
- **Benchmark Mode**: quick/full/stress_test
- **Test Resolution**: 256x256/512x512/768x768/auto
- **Save Report**: Generate detailed performance analysis

## 📈 Performance Expectations

Based on GPU tier:

### High-End (RTX 4090, 24GB)
- Preview Mode: 4-5x speedup
- Balanced: 2-3x speedup
- Supports all optimization features

### Mid-Range (RTX 4070, 12GB)
- Preview Mode: 3-4x speedup
- Balanced: 1.5-2x speedup
- Most features supported

### Entry-Level (RTX 4060, 8GB)
- Preview Mode: 2-3x speedup
- Balanced: 1.2-1.5x speedup
- Core features only

## 🔧 Manual Override

You can still manually set options:
1. **Speed Mode**: Override auto-selected speed
2. **Quality Preset**: Use specific preset instead of auto
3. **Enable DeepCache**: Force on/off

## 📊 Benchmark Results

After benchmarking, check:
- `~/latentsync_benchmark_report.txt` - Detailed performance analysis
- `~/latentsync_gpu_summary.txt` - Quick reference settings
- `~/.latentsync_benchmark_cache.json` - Cached results

## 💡 Tips

1. **First Time Users**: Always run the benchmark first
2. **After GPU/Driver Updates**: Re-run benchmark with `--clear`
3. **Testing Settings**: Use preview mode for quick tests
4. **Final Renders**: Switch to quality mode or manual settings
5. **Long Videos**: Auto mode will enable all optimizations

## 🐛 Troubleshooting

### "No cached benchmark results found"
- Run `python benchmark_my_gpu.py`

### "Could not create test pipeline"
- Ensure all dependencies are installed
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### "Benchmark failed"
- Close other GPU applications
- Ensure sufficient VRAM is available
- Try quick mode first: `python benchmark_my_gpu.py --quick`

## 🎉 Benefits

- **Zero Configuration**: Just use "auto" and it works
- **Optimal Performance**: Settings tailored to your exact GPU
- **Adaptive**: Adjusts based on video length
- **Safe**: Conservative settings for stability
- **Cached**: Benchmark only needs to run once

The auto-optimization system ensures you get the best possible performance without manual tuning!