# Lag Reduction Summary for ComfyUI-LatentSyncWrapper

## Changes Applied to Reduce Inference Lag

### 1. Further Reduced Batch Sizes
- High-end GPUs (>20GB): Reduced from 8 to 4
- Mid-range GPUs (>8GB): Reduced from 4 to 2  
- Lower-end GPUs: Reduced from 8 to 1
- This significantly reduces memory pressure during inference

### 2. Reduced Frame Processing Count
- Override config to process maximum 8 frames at once (down from 16)
- Forces batch_size to 1 in the config
- This reduces the amount of data processed in each iteration

### 3. Added Inference Optimizer
Created `inference_optimizer.py` with:
- `optimized_inference_context`: Aggressive memory management context
- `optimize_inference_pipeline`: Enables memory-efficient features like:
  - Flash Attention (if available)
  - VAE slicing and tiling
  - Attention slicing
  - Sequential CPU offload
- `reduce_inference_lag`: Further reduces batch sizes and frame counts

### 4. Added Memory Clearing in Pipeline
Modified `lipsync_pipeline.py` to:
- Clear GPU cache every 2 inference iterations
- Delete intermediate tensors after each batch
- Force garbage collection every 3 iterations
- This prevents memory accumulation during long videos

### 5. Added GPU Monitoring
Created `gpu_monitor.py` to:
- Monitor GPU memory usage every 2 seconds during inference
- Alert when VRAM usage exceeds 15GB or 18GB
- Helps diagnose memory spikes causing lag

### 6. Enhanced Memory Management
- More aggressive garbage collection threshold (0.6)
- Smaller memory split size (128MB)
- Disabled cudnn benchmarking during inference
- Added synchronization points for better memory release

## Expected Improvements

1. **Reduced Lag**: Processing fewer frames at once reduces GPU load
2. **Lower Memory Spikes**: Aggressive clearing prevents accumulation
3. **Better Visibility**: GPU monitoring shows when lag occurs
4. **Smoother Processing**: Sequential offloading prevents GPU saturation

## If Still Experiencing Lag

Try these additional steps:

1. **Reduce inference_steps**: Lower from 20 to 10-15
2. **Reduce video resolution**: Process at 256x256 instead of 512x512
3. **Process shorter clips**: Split long videos into segments
4. **Close other GPU applications**: Free up more VRAM
5. **Use CPU inference**: Set device to 'cpu' (much slower but no lag)

## Monitoring Output

During inference, you'll see:
- Memory usage logs before/after inference
- GPU monitoring alerts if VRAM exceeds thresholds
- Progress bars showing frame processing

This should significantly reduce the lag during inference frames.