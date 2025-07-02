# Adaptive GPU Configuration Guide

## Overview

The ComfyUI-LatentSyncWrapper now includes an intelligent adaptive GPU configuration system that automatically detects your hardware and optimizes settings for the best performance.

## Features

### 1. **Automatic GPU Detection**
- Detects your GPU model (RTX 4090, 4080, 3090, etc.)
- Identifies VRAM amount and compute capability
- Selects optimal profile based on hardware

### 2. **Pre-configured Profiles**
Optimized settings for popular GPUs:
- **RTX 4090** (24GB): Batch size 12, 16 frames, 85% VRAM
- **RTX 4080** (16GB): Batch size 8, 12 frames, 80% VRAM
- **RTX 3090** (24GB): Batch size 10, 16 frames, 80% VRAM
- **RTX 3080** (10GB): Batch size 4, 8 frames, 75% VRAM
- **RTX 3070** (8GB): Batch size 2, 8 frames, 70% VRAM

### 3. **Dynamic VRAM Control**
- Adjustable VRAM usage percentage (0-95%)
- 0% = Automatic based on GPU profile
- Higher values allow more VRAM usage but may impact system stability

### 4. **Smart Optimizations**
Each profile enables/disables optimizations based on GPU capability:
- **Flash Attention**: For newer GPUs (8.x compute capability)
- **VAE Slicing/Tiling**: For memory-constrained GPUs
- **CPU Offloading**: For low VRAM situations
- **TF32**: For RTX 40-series cards

## Using the Adaptive System

### In LatentSyncNode

The main node now includes a **vram_fraction** parameter:
- **0.0** (default): Use automatic profile-based settings
- **0.5-0.95**: Manual VRAM usage control

Example settings:
- **RTX 4090 Users**: Try 0.85-0.90 for maximum performance
- **RTX 3080 Users**: Stay at 0.0 (auto) or 0.70-0.75
- **Low VRAM GPUs**: Use 0.60-0.65 for stability

### GPU Configuration Node

Use the **GPU Configuration** node to:
1. View detected GPU and current settings
2. Override VRAM fraction
3. Override batch size
4. Save preferences for future use

### GPU Benchmark Node

Use the **GPU Benchmark** node to:
1. Test your GPU's compute performance
2. Measure memory bandwidth
3. Compare FP32 vs FP16 performance

## Performance Tips

### For RTX 4090 Users
Your card can handle higher settings:
```
VRAM Fraction: 0.85-0.90
Batch Size: 12-16
Inference Steps: 20-30
```

### For Mid-Range GPUs (8-16GB)
Use balanced settings:
```
VRAM Fraction: 0.0 (auto) or 0.70-0.75
Batch Size: Auto
Inference Steps: 15-20
```

### For Low VRAM GPUs (<8GB)
Conservative settings:
```
VRAM Fraction: 0.60-0.65
Batch Size: Auto
Inference Steps: 10-15
```

## Troubleshooting

### Still experiencing lag?
1. Reduce VRAM fraction (try 0.10 lower)
2. Close other GPU applications
3. Use GPU Configuration node to manually reduce batch size
4. Reduce inference steps

### Out of Memory errors?
1. Set VRAM fraction to 0.60
2. Enable CPU offloading in code
3. Process shorter video clips

### Want maximum speed?
1. Increase VRAM fraction (up to 0.90)
2. Use GPU Configuration node to increase batch size
3. Ensure no other apps are using GPU

## Advanced Configuration

### Saving Custom Settings
The GPU Configuration node can save your preferences to `gpu_config.json`:
```json
{
  "batch_size": 16,
  "vram_fraction": 0.85,
  "inference_steps": 25
}
```

### Manual Profile Editing
Edit `adaptive_gpu_config.py` to add custom GPU profiles or modify existing ones.

## Technical Details

The adaptive system considers:
1. **GPU Model**: Specific optimizations for each GPU
2. **VRAM Amount**: Determines batch sizes and frame counts
3. **Compute Capability**: Enables features like TF32 and Flash Attention
4. **User Preferences**: Respects manual overrides and saved settings

The system dynamically adjusts:
- Batch processing size
- Number of frames per batch
- Memory allocation percentage
- Optimization features
- Mixed precision settings

## Benefits

1. **No more manual tweaking** - Works out of the box
2. **Optimal performance** - Uses your GPU to its full potential
3. **Prevents crashes** - Conservative memory management
4. **Flexibility** - Manual overrides when needed
5. **Future-proof** - Easy to add new GPU profiles