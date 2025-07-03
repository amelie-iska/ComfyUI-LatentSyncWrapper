# Speed Optimization Guide for RTX 4090 and High-End GPUs

## Overview

The ComfyUI-LatentSyncWrapper now includes advanced speed optimizations that can significantly reduce inference time, especially for RTX 4090 users.

## Optimization Levels

### Conservative (Default for <12GB VRAM)
- Maintains maximum quality
- No inference step reduction
- Standard batch sizes
- Suitable for GPUs with limited VRAM

### Balanced (Default for 12-24GB VRAM)
- 20% reduction in inference steps
- Moderate batch size increases
- Enables TF32 precision on supported GPUs
- Good balance between speed and quality

### Aggressive (Recommended for RTX 4090)
- 40% reduction in inference steps
- Maximum batch sizes (up to 20 for RTX 4090)
- Enables all optimizations:
  - DeepCache for feature reuse
  - torch.compile() for kernel fusion
  - CUDA graphs for reduced overhead
  - TF32 precision
  - VAE tiling and optimization

## Performance Improvements

### RTX 4090 Expected Performance
- **Conservative**: ~2 minutes (baseline)
- **Balanced**: ~1.5 minutes (25% faster)
- **Aggressive**: ~45-60 seconds (50-66% faster)

## Usage

In the LatentSyncNode, set:
1. **VRAM Fraction**: 0.85-0.90 for RTX 4090
2. **Optimization Level**: "aggressive" for maximum speed
3. **Inference Steps**: Can be reduced to 12-15 for faster results

## Tips for Maximum Speed

### For RTX 4090 Users:
```
VRAM Fraction: 0.85
Optimization Level: aggressive
Inference Steps: 15
Batch Size: (auto-configured to 20)
```

### Quality vs Speed Trade-off:
- **Aggressive** mode reduces inference steps by 40%
- May have slight quality impact on complex scenes
- Perfect for previews or when speed is critical
- Use **balanced** for production quality

### Memory Considerations:
- Aggressive mode uses more VRAM for larger batches
- If you get OOM errors, reduce VRAM fraction to 0.75
- Close other GPU applications for best performance

## Technical Details

### Optimizations Applied:

1. **DeepCache**: Reuses intermediate features across denoising steps
2. **torch.compile()**: JIT compilation for 10-20% kernel speedup
3. **CUDA Graphs**: Reduces kernel launch overhead
4. **Dynamic Batching**: Increases batch size for RTX 4090 to 20
5. **TF32 Precision**: Uses Tensor Cores for faster matrix operations
6. **VAE Optimization**: Tiled decoding for efficient memory usage

### Monitoring Performance:
- Watch the console output for optimization messages
- GPU monitor will show utilization statistics
- Memory usage is logged before/after inference

## Troubleshooting

### Still slow?
1. Ensure no other apps are using GPU
2. Try restarting ComfyUI
3. Check if Windows GPU scheduling is disabled
4. Update to latest NVIDIA drivers (545.x or newer)

### Out of Memory?
1. Reduce VRAM fraction to 0.75
2. Use "balanced" instead of "aggressive"
3. Process shorter video clips

### Quality issues?
1. Switch to "balanced" optimization
2. Increase inference steps to 25-30
3. Use "conservative" for final renders