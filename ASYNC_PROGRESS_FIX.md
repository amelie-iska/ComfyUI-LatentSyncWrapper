# Async Progress Bar Fix for LatentSync

## Problem Identified

The issue with "sample faces taking long to load but loading eventually" is caused by:

1. **Blocking tqdm Progress Bars**: The pipeline uses `tqdm.tqdm` which prints to console and blocks the UI thread
2. **Synchronous Face Processing**: Face detection/processing happens synchronously without progress updates to ComfyUI
3. **No Progress Callback Integration**: Progress updates aren't properly sent to ComfyUI's progress system

## Issues Found

### 1. In `lipsync_pipeline.py` (line 335):
```python
with tqdm.tqdm(total=num_frames) as pbar:
    for i in range(0, num_frames, batch_size):
        # Face processing...
        pbar.update(batch_end - i)
```
This creates a console progress bar that doesn't update ComfyUI's UI.

### 2. In `lipsync_pipeline.py` (line 380):
```python
with tqdm.tqdm(total=num_faces) as pbar:
    for i in range(0, num_faces, batch_size):
        # Face restoration...
        pbar.update(batch_end - i)
```
Same issue during face restoration.

### 3. In `lipsync_pipeline.py` (line 615):
```python
for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
```
Main inference loop also uses blocking tqdm.

## Solution Implemented

Created `progress_fix.py` that:

1. **Replaces tqdm with ComfyUI-compatible progress bars**
2. **Implements async progress updates with rate limiting**
3. **Provides proper callback integration**

### Key Features:

1. **ComfyUIProgressBar**: Drop-in replacement for tqdm
   - Rate-limited updates (100ms intervals)
   - Proper progress calculation with ETA
   - Callback support for UI updates

2. **AsyncProgressWrapper**: Handles updates asynchronously
   - Background thread for progress updates
   - Queue-based system prevents UI flooding
   - Automatic cleanup

3. **Pipeline Patches**: Monkey patches the pipeline
   - Replaces all tqdm usage
   - Integrates with ComfyUI callbacks
   - Non-blocking progress updates

## How It Works

1. **Import Fix**: The fix is automatically applied when imported:
   ```python
   from .progress_fix import patch_pipeline_progress
   ```

2. **Callback Integration**: Pipeline now accepts ComfyUI callbacks:
   ```python
   pipeline.set_comfy_callback(comfyui_progress_callback)
   ```

3. **Non-blocking Updates**: Progress updates happen in background thread

## Benefits

1. **Responsive UI**: No more freezing during face processing
2. **Real-time Updates**: See progress as faces are processed
3. **Proper Integration**: Works with ComfyUI's progress system
4. **Backwards Compatible**: Falls back to console output if no callback

## Technical Details

### Rate Limiting
- Updates limited to 10Hz (100ms intervals)
- Prevents UI flooding with too many updates
- Maintains smooth progress display

### Memory Safety
- Progress updates don't interfere with GPU operations
- Async updates prevent blocking main thread
- Proper cleanup prevents memory leaks

### Error Handling
- Graceful fallback to console output
- Exception handling prevents crashes
- Thread-safe operations

## Testing

To verify the fix works:

1. Run a LatentSync generation
2. Watch for "Processing faces" progress bar
3. UI should remain responsive
4. Progress should update smoothly

## Future Improvements

1. Add preview frames during face processing
2. Show individual face detection status
3. Add time estimates for each phase
4. Implement cancellation support