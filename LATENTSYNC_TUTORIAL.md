# 🎭 LatentSync 1.6 (MEMSAFE) - Complete Tutorial

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Input/Output Ports](#inputoutput-ports)
5. [Parameter Guide](#parameter-guide)
6. [Memory Management](#memory-management)
7. [Workflow Examples](#workflow-examples)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Tips](#advanced-tips)

---

## Overview

LatentSync 1.6 is a powerful lip-sync generation node for ComfyUI that synchronizes video with audio using advanced AI models. The MEMSAFE version includes extensive memory optimization features to handle videos of any length without running out of VRAM.

### Key Features:
- 🎯 High-quality lip synchronization
- 💾 Memory-safe processing for long videos
- 🚀 GPU optimization for RTX 30/40 series
- 📁 Flexible output modes (frames or video file)
- ⚙️ Adaptive performance settings

---

## Installation

The node should already be installed if you're seeing it in ComfyUI. If you need to reinstall:

1. Navigate to: `ComfyUI/custom_nodes/`
2. Clone the repository: `git clone https://github.com/amelie-iska/ComfyUI-LatentSyncWrapper`
3. Install dependencies: `pip install -r requirements.txt`
4. Restart ComfyUI

---

## Basic Usage

### Quick Start Workflow:
1. Load your video/image sequence → Connect to `images` input
2. Load your audio file → Connect to `audio` input  
3. Set basic parameters (usually defaults work well)
4. Click "Queue Prompt"
5. Get synchronized output!

### Minimum Setup:
```
[Load Image/Video] → images → [LatentSync 1.6] → images → [Save Image/Video]
[Load Audio] → audio → [LatentSync 1.6] → audio → [Save Audio]
                                           → video_path → [Copy/Save Video File]
```

---

## Input/Output Ports

### Inputs (Left Side):
| Port | Type | Description |
|------|------|-------------|
| **images** | IMAGE | Video frames or image sequence to process |
| **audio** | AUDIO | Audio file for lip synchronization |

### Outputs (Right Side):
| Port | Type | Description |
|------|------|-------------|
| **images** | IMAGE | Processed video frames (when return_frames=True) |
| **audio** | AUDIO | Resampled audio (16kHz) |
| **video_path** | STRING | Path to saved video file (when return_frames=False) |

---

## Parameter Guide

### Essential Parameters

#### 🎲 **seed** (INT)
- **Default**: 1247
- **Range**: Any integer
- **Purpose**: Controls randomness for reproducible results
- **Tip**: Use the same seed to get identical results

#### 👄 **lips_expression** (FLOAT)  
- **Default**: 1.5
- **Range**: 1.0 - 3.0
- **Purpose**: Controls lip movement intensity
- **Guidelines**:
  - `1.0`: Subtle, natural movements
  - `1.5`: Balanced expression (recommended)
  - `2.0-3.0`: Exaggerated movements

#### 🔄 **inference_steps** (INT)
- **Default**: 20
- **Range**: 1 - 999
- **Purpose**: Processing quality vs speed
- **Guidelines**:
  - `10-15`: Fast processing, slightly lower quality
  - `20`: Balanced (recommended)
  - `30-50`: Higher quality, slower processing

### Memory Management Parameters

#### 💾 **vram_fraction** (FLOAT)
- **Default**: 0.0 (auto)
- **Range**: 0.0 - 0.95
- **Purpose**: Limits GPU memory usage
- **Guidelines by GPU**:
  - RTX 4090/3090 (24GB): `0.85-0.95`
  - RTX 4070/3080 (12GB): `0.70-0.85`
  - RTX 4060/3060 (8GB): `0.60-0.70`
  - `0.0`: Auto-detect optimal setting

#### ⚡ **optimization_level** (COMBO)
- **Options**: `conservative`, `balanced`, `aggressive`
- **Default**: `balanced`
- **Purpose**: Speed vs quality trade-off
- **When to use**:
  - `conservative`: Maximum quality, slower
  - `balanced`: Good for most cases
  - `aggressive`: Fastest, may reduce quality

#### 🧠 **memory_mode** (COMBO)
- **Options**: `aggressive`, `balanced`, `conservative`
- **Default**: `balanced`
- **Purpose**: Memory usage strategy
- **Guidelines**:
  - `aggressive`: Uses more VRAM, faster
  - `balanced`: Good for most videos
  - `conservative`: Minimal VRAM, best for long videos

#### 💿 **enable_disk_cache** (BOOLEAN)
- **Default**: False
- **Purpose**: Use disk for temporary storage
- **When to enable**:
  - Videos over 500 frames
  - Limited VRAM (8GB or less)
  - Getting OOM errors

#### 🎬 **return_frames** (BOOLEAN)
- **Default**: False  
- **Purpose**: Return frames in memory vs video file path
- **Guidelines**:
  - `True`: For further processing in ComfyUI
  - `False`: For direct video saving (memory efficient)

#### 📤 **output_mode** (COMBO)
- **Options**: `auto`, `video_file`, `frames`
- **Default**: `auto`
- **Purpose**: Control output format
- **Behavior**:
  - `auto`: Decides based on video length
  - `video_file`: Always save to disk
  - `frames`: Always return frames

---

## Memory Management

### Video Length Guidelines

| Video Length | memory_mode | vram_fraction | enable_disk_cache | output_mode |
|--------------|-------------|---------------|-------------------|-------------|
| <100 frames | aggressive | 0.85-0.95 | False | frames |
| 100-300 frames | balanced | 0.70-0.85 | False | auto |
| 300-500 frames | balanced | 0.60-0.70 | True | auto |
| 500+ frames | conservative | 0.50-0.60 | True | video_file |

### Memory Optimization Tips:
1. **Start conservative**: Begin with lower settings and increase if stable
2. **Monitor VRAM**: Use GPU-Z or nvidia-smi to watch memory usage
3. **Batch processing**: For very long videos, consider splitting into segments
4. **Close other apps**: Free up VRAM by closing browsers, Discord, etc.

---

## Workflow Examples

### 1. Short Video (<100 frames)
```
Settings:
- optimization_level: aggressive
- memory_mode: aggressive  
- enable_disk_cache: False
- output_mode: frames
```

### 2. Medium Video (100-300 frames)
```
Settings:
- optimization_level: balanced
- memory_mode: balanced
- enable_disk_cache: False
- output_mode: auto
```

### 3. Long Video (500+ frames)
```
Settings:
- optimization_level: conservative
- memory_mode: conservative
- enable_disk_cache: True
- output_mode: video_file
- vram_fraction: 0.6
```

### 4. Maximum Quality
```
Settings:
- inference_steps: 30-50
- optimization_level: conservative
- lips_expression: 1.5
```

### 5. Maximum Speed
```
Settings:
- inference_steps: 10-15
- optimization_level: aggressive
- memory_mode: aggressive
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory (OOM) Error**
```
Solutions:
- Reduce vram_fraction to 0.6 or lower
- Set memory_mode to "conservative"
- Enable disk_cache
- Use output_mode="video_file"
- Reduce batch size in video loader
- Close other GPU applications
```

#### 2. **Slow Processing**
```
Solutions:
- Increase vram_fraction if you have headroom
- Set optimization_level to "aggressive"
- Reduce inference_steps to 15
- Check if disk_cache is slowing things down
- Ensure GPU is not thermal throttling
```

#### 3. **Poor Lip Sync Quality**
```
Solutions:
- Increase inference_steps to 25-30
- Ensure audio is clear and properly aligned
- Try different lips_expression values
- Check video FPS matches expected (default 25)
- Use "conservative" optimization_level
```

#### 4. **Video/Audio Length Mismatch**
```
Solutions:
- Use VideoLengthAdjuster node before LatentSync
- Ensure audio covers full video duration
- Check FPS settings in video loader
```

#### 5. **Node Won't Load**
```
Solutions:
- Check all dependencies are installed
- Restart ComfyUI
- Check for error messages in console
- Verify CUDA is properly installed
```

---

## Advanced Tips

### Performance Optimization

1. **GPU-Specific Settings**:
   - RTX 4090: Can handle aggressive settings for most videos
   - RTX 3080/4070: Use balanced settings, monitor VRAM
   - RTX 3060/4060: Conservative settings recommended

2. **Batch Processing**:
   - Use VideoChunkProcessor for very long videos
   - Process in segments and concatenate results

3. **Quality Enhancement**:
   - Preprocess video to ensure face is clearly visible
   - Use good lighting in source video
   - Higher resolution inputs generally give better results

### Integration Tips

1. **With Video Editors**:
   - Export at 25 FPS for best compatibility
   - Use lossless or high-quality codecs
   - Maintain consistent frame dimensions

2. **Audio Preparation**:
   - Clean audio (remove background noise)
   - Normalize audio levels
   - Ensure clear speech

3. **Workflow Optimization**:
   - Save successful settings as workflow templates
   - Use the GPU Benchmark node to find optimal settings
   - Create presets for different video types

### Best Practices

1. **Always test with short clips first**
2. **Save original files before processing**
3. **Use consistent settings for video series**
4. **Monitor system resources during processing**
5. **Keep ComfyUI and nodes updated**

---

## Example Preset Configurations

### 🎥 **Talking Head Videos**
```
seed: 1247
lips_expression: 1.5
inference_steps: 20
optimization_level: balanced
memory_mode: balanced
```

### 🎤 **Music Videos**
```
seed: [any]
lips_expression: 2.0-2.5
inference_steps: 25
optimization_level: conservative
memory_mode: balanced
```

### 📱 **Social Media Content**
```
seed: [any]
lips_expression: 1.8
inference_steps: 15
optimization_level: aggressive
memory_mode: aggressive
output_mode: video_file
```

### 🎬 **Film/High Quality**
```
seed: [fixed for consistency]
lips_expression: 1.3-1.5
inference_steps: 30-50
optimization_level: conservative
memory_mode: conservative
enable_disk_cache: True
```

---

## Conclusion

LatentSync 1.6 (MEMSAFE) provides powerful lip-sync capabilities with extensive memory management options. Start with default settings and adjust based on your specific needs and hardware capabilities. The key to success is finding the right balance between quality, speed, and memory usage for your particular use case.

Remember: When in doubt, start conservative and gradually increase performance settings until you find the optimal configuration for your system.

---

## Need Help?

- Check the console for detailed error messages
- Review the workflow examples in `/workflows/`
- Adjust settings incrementally
- Consider your hardware limitations

Happy lip-syncing! 🎭