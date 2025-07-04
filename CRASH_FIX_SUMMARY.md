# LatentSync Crash Fix Summary

## Problem
The ComfyUI-LatentSyncWrapper was crashing when processing large videos (1+ minute) at 100% completion during frame restoration. The crash occurred because all frames were being loaded into memory at once using `torch.cat()`, causing memory exhaustion.

## Solution Implemented

### 1. Dual Output Mode
- Added `output_mode` parameter with three options:
  - `"auto"` (default): Automatically choose based on video length
  - `"video_file"`: Always return video file path
  - `"frames"`: Always return frame tensor

- Changed `return_frames` default to `False` for safer default behavior

### 2. Auto Mode Logic
The system now automatically determines the best output mode based on:
- Video length (frame count)
- Memory mode setting
- Thresholds:
  - Aggressive mode: 300 frames
  - Balanced mode: 200 frames  
  - Conservative mode: 100 frames

### 3. Disk-Based Frame Loading
For large videos when frames are requested:
- Uses `LongVideoHandler` to extract frames to disk
- Loads frames in small batches (8-32 frames based on memory mode)
- Progressively concatenates frames without holding all in memory
- Cleans up temporary files after loading

## Usage Recommendations

### For Normal Workflows
- Use default settings (`output_mode="auto"`)
- System will automatically return video path for large videos
- Small videos will still return frames for compatibility

### For Video Combine Node Workflows
- Set `output_mode="frames"` explicitly
- Enable `enable_disk_cache=True` for large videos
- Use `memory_mode="conservative"` for videos over 200 frames

### For Maximum Safety
- Set `output_mode="video_file"`
- Process the saved video file in subsequent steps
- This completely eliminates memory issues

## Benefits
1. **No more crashes** on large video inference
2. **Backward compatible** with existing workflows
3. **Automatic optimization** based on video size
4. **Flexible output options** for different use cases
5. **Memory-efficient** frame loading when needed