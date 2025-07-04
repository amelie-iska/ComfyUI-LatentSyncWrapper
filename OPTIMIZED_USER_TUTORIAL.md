# ComfyUI-LatentSyncWrapper Optimized User Tutorial

## Quick Start Guide

This tutorial will help you get the most out of the optimized ComfyUI-LatentSyncWrapper nodes. Our optimizations ensure smooth processing of both short and long videos without crashes or memory issues.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Understanding the Nodes](#understanding-the-nodes)
3. [Recommended Settings by GPU](#recommended-settings-by-gpu)
4. [Processing Different Video Types](#processing-different-video-types)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## Basic Setup

### Simple Workflow
```
Load Video → Load Audio → LatentSync 1.6 → Save Video
```

### Enhanced Workflow for Long Videos
```
Load Video → Video Length Adjuster → LatentSync 1.6 → Save Video
```

## Understanding the Nodes

### 1. LatentSync 1.6 Node - Main Processing Node

**Key Parameters:**
- **video_path**: Your input video
- **audio**: Audio input from ComfyUI audio loader
- **seed**: For reproducible results (default: 1247)
- **lips_expression**: Controls lip movement intensity (1.0-3.0, default: 1.5)
- **inference_steps**: Quality vs speed trade-off (10-50, default: 20)
- **vram_fraction**: GPU memory usage (0.0=auto, 0.5-0.95=manual)
- **optimization_level**: Speed settings ("conservative", "balanced", "aggressive")
- **memory_mode**: Memory management ("aggressive", "balanced", "conservative")
- **output_mode**: Output type ("auto", "video_file", "frames")
- **enable_disk_cache**: Use disk for long videos (recommended: True)

### 2. Video Length Adjuster Node - Match Video to Audio Length

**Key Parameters:**
- **mode**: How to adjust length
  - "normal": Pad with last frame or truncate
  - "pingpong": Create forward-backward loops (great for short videos)
  - "loop_to_audio": Simple frame looping
- **fps**: Target frame rate (default: 25.0)
- **silent_padding_sec**: Add silence at end (default: 0.5)

### 3. GPU Configuration Node - View and Override GPU Settings

**Key Parameters:**
- **vram_fraction**: Override automatic VRAM usage
- **batch_size_override**: Manual batch size control
- **save_settings**: Save preferences for future use

### 4. Efficient Video Loader - For Large Video Files

**Key Parameters:**
- **batch_size**: Frames to load at once (10-200)
- **start_frame/end_frame**: Process video segments

## Recommended Settings by GPU

### RTX 4090 (24GB) - Maximum Performance
```
VRAM Fraction: 0.85-0.90
Optimization Level: aggressive
Memory Mode: aggressive (short videos) / balanced (long videos)
Inference Steps: 15-20
Enable Disk Cache: True (for videos >300 frames)
```

### RTX 3090 (24GB) - Balanced Performance
```
VRAM Fraction: 0.80 (or 0.0 for auto)
Optimization Level: balanced
Memory Mode: balanced
Inference Steps: 20
Enable Disk Cache: True (for videos >200 frames)
```

### RTX 3080/4070 (10-16GB) - Conservative Settings
```
VRAM Fraction: 0.70-0.75
Optimization Level: balanced
Memory Mode: conservative
Inference Steps: 15-20
Enable Disk Cache: True (always)
```

### RTX 3070 or Lower (8GB) - Memory-Safe Settings
```
VRAM Fraction: 0.60-0.65
Optimization Level: conservative
Memory Mode: conservative
Inference Steps: 10-15
Enable Disk Cache: True (always)
Output Mode: video_file (recommended)
```

## Processing Different Video Types

### Short Videos (<30 seconds / <750 frames)
- Use default settings
- Memory Mode: aggressive or balanced
- Output Mode: auto
- Disk Cache: False (optional)

### Medium Videos (30-60 seconds / 750-1500 frames)
- Memory Mode: balanced
- Enable Disk Cache: True
- Output Mode: auto
- Consider using Video Length Adjuster if matching audio

### Long Videos (1-3 minutes / 1500-4500 frames)
- Memory Mode: conservative
- Enable Disk Cache: True
- Output Mode: video_file (recommended)
- Use Efficient Video Loader for very large files
- Process in chunks if needed

### Very Long Videos (3+ minutes)
1. Use Efficient Video Loader with small batch_size (50-100)
2. Set Memory Mode: conservative
3. Enable Disk Cache: True
4. Output Mode: video_file
5. Consider processing segments separately

## Advanced Features

### 1. Speed Optimization (RTX 4090 Users)
Set optimization_level to "aggressive" for:
- 50-66% faster processing
- Slight quality trade-off
- Perfect for previews or quick tests

### 2. Video Length Adjustment Modes
- **Normal Mode**: Best for speeches/presentations
- **Pingpong Mode**: Perfect for extending short clips to match long audio
- **Loop Mode**: Good for repetitive content

### 3. Output Modes Explained
- **auto**: Smart selection based on video length (recommended)
- **video_file**: Always saves to disk (safest for memory)
- **frames**: Returns frame tensor (for chaining nodes)

### 4. Memory Modes Explained
- **aggressive**: Max performance, keeps more in VRAM
- **balanced**: Good for most use cases
- **conservative**: Maximum stability, uses disk caching

## Troubleshooting

### "Out of Memory" Errors
1. Reduce vram_fraction to 0.60
2. Set memory_mode to "conservative"
3. Enable disk_cache
4. Use output_mode="video_file"
5. Close other GPU applications

### Slow Processing
1. For RTX 4090: Use optimization_level="aggressive"
2. Reduce inference_steps to 12-15
3. Check GPU utilization with GPU Benchmark node
4. Ensure no other apps are using GPU

### Video/Audio Sync Issues
1. Use Video Length Adjuster node
2. Try different adjustment modes
3. Add silent_padding_sec for better sync

### Poor Quality Results
1. Increase inference_steps to 25-30
2. Use optimization_level="balanced" or "conservative"
3. Adjust lips_expression parameter (1.5-2.5)

## Best Practices

1. **Always use auto settings first** - The system intelligently configures based on your hardware
2. **Enable disk cache for long videos** - Prevents crashes with minimal performance impact
3. **Start conservative, then optimize** - Better to process slowly than crash
4. **Monitor first run** - Watch console output to understand what settings work
5. **Save working configurations** - Use GPU Configuration node to save preferences

## Performance Tips

### For Maximum Speed:
- Close all other applications
- Use aggressive optimization level
- Reduce inference steps to 12-15
- Process shorter segments

### For Maximum Quality:
- Use balanced/conservative optimization
- Increase inference steps to 30+
- Process at original resolution
- Use normal output mode

### For Maximum Stability:
- Use conservative memory mode
- Enable disk cache always
- Output to video file
- Process in smaller chunks

## Example Workflows

### Basic Lip Sync:
```
Load Video → Load Audio → LatentSync 1.6 (defaults) → Save Video
```

### Long Video with Audio Matching:
```
Load Video → Video Length Adjuster (pingpong) → Load Audio → LatentSync 1.6 (conservative) → Save Video
```

### High-Performance Preview:
```
Load Video → LatentSync 1.6 (aggressive optimization, 12 steps) → Preview
```

### Memory-Safe Large Video:
```
Efficient Video Loader (batch=50) → LatentSync 1.6 (conservative, disk cache, video_file output) → Save
```

## Summary

The optimized LatentSyncWrapper provides:
- **Automatic configuration** based on your GPU
- **Crash prevention** through intelligent memory management
- **Flexible output options** for different workflows
- **Speed optimizations** for high-end GPUs
- **Long video support** through disk caching

Start with default settings and adjust based on your needs. The system will automatically prevent most common issues while maximizing performance for your hardware.