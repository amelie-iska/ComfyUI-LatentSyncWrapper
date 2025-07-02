# Memory Fix Summary for ComfyUI-LatentSyncWrapper v1.6

## Changes Applied

### 1. Fixed UnboundLocalError (Critical Bug Fix)
- Removed duplicate `del frames_uint8` statements that were causing the UnboundLocalError
- Fixed duplicate video writing and audio saving code blocks

### 2. Removed All Duplicate Code Blocks in nodes.py
- Removed duplicate import statements (lines 15-21)
- Removed duplicate `limit_gpu_memory()` calls
- Removed duplicate frame processing code (lines 565-575)
- Removed duplicate audio saving code
- Removed duplicate video writing code
- Removed duplicate inference calls

### 3. Updated Batch Sizes (More Conservative)
- High-end GPUs (>20GB): Reduced from 32 to 8
- Mid-range GPUs (>8GB): Reduced from 16 to 4
- Lower-end GPUs: Reduced from 8 to 2

### 4. Updated Memory Fractions in memory_limiter.py
- 24GB+ GPUs: Reduced from 0.95 to 0.75 (75%)
- 16GB+ GPUs: Reduced from 0.90 to 0.75 (75%)
- Other GPUs: Reduced from 0.83 to 0.70 (70%)

### 5. Added Model Cleanup After Inference
- Added explicit deletion of pipeline components (unet, vae, audio_encoder)
- Added garbage collection after model cleanup
- Added GPU cache clearing with logging

### 6. Added Memory Usage Logging
- Added `log_memory_usage()` function to memory_limiter.py
- Logs allocated, reserved, and free VRAM at key stages
- Warns when VRAM usage exceeds 18GB

### 7. Added @torch.no_grad() Decorators
- Added to `affine_transform_video()` in lipsync_pipeline.py
- Added to `restore_video()` in lipsync_pipeline.py
- Added to `loop_video()` in lipsync_pipeline.py
- Prevents gradient accumulation during inference operations

## Expected Improvements

1. **No More UnboundLocalError** - The duplicate deletion bug is fixed
2. **Better VRAM Management** - More conservative memory allocation leaves room for ComfyUI overhead
3. **Reduced Memory Usage** - Smaller batch sizes and no gradient tracking
4. **Better Visibility** - Memory logging helps debug usage patterns
5. **Model Cleanup** - Freeing models after use prevents memory accumulation

## Testing Recommendations

1. Test with the same workflow that previously failed
2. Monitor VRAM usage through the new logging outputs
3. Verify that the 20GB cap is now respected
4. Check that longer videos can be processed without OOM errors

## Additional Notes

- The memory fractions are now more conservative to account for ComfyUI's own memory needs
- Batch sizes are significantly reduced but should still provide good performance
- The @torch.no_grad() decorators will prevent unnecessary gradient computation
- Memory logging will help identify any remaining bottlenecks