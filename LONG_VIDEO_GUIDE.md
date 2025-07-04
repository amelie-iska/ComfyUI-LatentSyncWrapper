# Long Video Processing Guide

## Overview

The ComfyUI-LatentSyncWrapper now includes advanced features for processing long videos that would normally exceed GPU memory limits. These features are inspired by VideoBasicLatentSync but enhanced with our optimization framework.

## New Features

### 1. Memory Mode Selection

The LatentSyncNode now includes a **memory_mode** parameter with three options:

- **Aggressive**: Maximum performance, keeps frames in VRAM (best for short videos)
- **Balanced**: Smart memory management, good for medium-length videos
- **Conservative**: Disk-based processing, handles very long videos

### 2. Disk-Based Processing

Enable **enable_disk_cache** to automatically use disk-based processing for long videos:

- Videos are extracted to individual frames on disk
- Frames are loaded in batches during processing
- Significantly reduces VRAM usage
- Allows processing of videos with 500+ frames

### 3. Video Length Adjuster Node

New node: **Video Length Adjuster (Enhanced)**

Adjust video length to match audio with three modes:

- **Normal**: Pads with last frame or truncates
- **Pingpong**: Creates forward-backward loops (great for extending short videos)
- **Loop**: Simple frame looping

Example use cases:
- Short video + long audio: Use pingpong mode
- Perfect sync needed: Use normal mode
- Seamless loops: Use loop mode

### 4. Memory Mode Configuration Node

Control memory behavior with the **Memory Mode Configuration** node:

- Select memory mode (aggressive/balanced/conservative)
- Enable/disable disk caching
- Set long video threshold (frames)

### 5. Video Chunk Processor Node

For manual control over video chunking:

- Split videos into manageable chunks
- Configure chunk size and overlap
- Process chunks independently

## Usage Examples

### Processing a Long Video (500+ frames)

1. Set **memory_mode** to "conservative"
2. Enable **enable_disk_cache**
3. Use default settings for other parameters

### Processing with Limited VRAM (8GB GPU)

1. Set **memory_mode** to "conservative"
2. Set **optimization_level** to "conservative"
3. Set **vram_fraction** to 0.60-0.65
4. Enable **enable_disk_cache**

### Extending a Short Video Loop

1. Connect **Video Length Adjuster** node
2. Set mode to "pingpong"
3. Adjust FPS as needed
4. Add silent padding if desired

## Memory Mode Details

### Aggressive Mode
- Max frames in memory: 32
- VRAM fraction: 95%
- Batch size multiplier: 1.5x
- Use for: Short videos (<100 frames)

### Balanced Mode (Default)
- Max frames in memory: 16
- VRAM fraction: 85%
- Batch size multiplier: 1.0x
- Use for: Medium videos (100-200 frames)

### Conservative Mode
- Max frames in memory: 8
- VRAM fraction: 70%
- Batch size multiplier: 0.5x
- Use for: Long videos (200+ frames)

## Performance Tips

### For RTX 4090 Users
- Long videos (300+ frames): Use balanced mode with disk cache
- Medium videos (100-300): Use aggressive mode without disk cache
- Short videos (<100): Use aggressive mode with aggressive optimization

### For Mid-Range GPUs (8-16GB)
- Always enable disk cache for videos >150 frames
- Use conservative mode for videos >200 frames
- Consider reducing resolution for very long videos

### For Low VRAM GPUs (<8GB)
- Always use conservative mode
- Enable disk cache for all videos
- Process at lower resolutions when possible

## Troubleshooting

### Out of Memory with Long Videos
1. Enable disk cache
2. Switch to conservative memory mode
3. Reduce vram_fraction to 0.60
4. Process video in chunks using Video Chunk Processor

### Slow Processing with Disk Cache
- Disk-based processing trades speed for stability
- Use SSD for temp directory if possible
- Reduce chunk size for faster per-chunk processing

### Video/Audio Sync Issues
- Use Video Length Adjuster node
- Try different adjustment modes
- Add silent padding for better sync

## Advanced Configuration

### Custom Thresholds
Edit `long_video_handler.py` to customize:
```python
self.thresholds = {
    "aggressive": {"max_frames_in_memory": 32, "max_video_length": 100},
    "balanced": {"max_frames_in_memory": 16, "max_video_length": 200},
    "conservative": {"max_frames_in_memory": 8, "max_video_length": 500}
}
```

### Progressive Processing
The system supports checkpoint-based processing:
- Automatically saves progress
- Can resume if interrupted
- Useful for very long videos (1000+ frames)

## Workflow Examples

### Basic Long Video Workflow
```
Load Video → Video Length Adjuster → LatentSyncNode (conservative mode) → Save Video
```

### Advanced Workflow with Chunks
```
Load Video → Video Chunk Processor → LatentSyncNode → Combine Chunks → Save Video
```

### Memory-Optimized Workflow
```
Memory Mode Config → Load Video → LatentSyncNode → Save Video
```