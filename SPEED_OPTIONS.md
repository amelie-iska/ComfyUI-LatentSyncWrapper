# LatentSync 1.6 Speed Options Guide

The LatentSync 1.6 (MEMSAFE+SPEED) node now includes built-in speed optimization options that can provide 2-5x speedup while maintaining quality.

## New Speed Options

### 1. Speed Mode
Controls the overall speed optimization level:
- **normal**: Default mode, no speed optimizations
- **fast**: ~1.5x speedup with minimal quality impact
- **turbo**: ~2x speedup with slight quality trade-off
- **ultra**: ~3x speedup for draft/preview purposes

### 2. Quality Preset
Fine-tune the quality vs speed balance:
- **auto**: Automatically adjust based on content
- **draft**: Fastest rendering for previews (50% fewer steps)
- **fast**: Quick processing with good quality (70% steps)
- **balanced**: Default balanced approach (100% steps)
- **quality**: Higher quality output (150% steps)
- **ultra**: Maximum quality (200% steps)

### 3. Enable DeepCache
Toggle DeepCache optimization:
- **OFF**: Standard processing
- **ON**: Additional ~1.5x speedup by caching intermediate features

## Usage Examples

### Maximum Speed (Preview/Draft)
- Speed Mode: `ultra`
- Quality Preset: `draft`
- Enable DeepCache: `ON`
- Expected speedup: ~4.5x

### Fast Production
- Speed Mode: `turbo`
- Quality Preset: `fast`
- Enable DeepCache: `ON`
- Expected speedup: ~3x

### Balanced Performance
- Speed Mode: `fast`
- Quality Preset: `balanced`
- Enable DeepCache: `ON`
- Expected speedup: ~2.2x

### High Quality
- Speed Mode: `normal`
- Quality Preset: `quality`
- Enable DeepCache: `OFF`
- Expected speedup: None (quality priority)

## How It Works

1. **Speed Mode** adjusts:
   - Minimum inference steps
   - Temporal coherence optimization
   - Frame similarity detection

2. **Quality Preset** modifies:
   - Number of inference steps
   - Attention mechanism settings
   - Feature reuse strategies

3. **DeepCache** provides:
   - Intelligent feature caching
   - Reduced UNet computations
   - Adaptive cache intervals

## Tips

- Start with `fast` mode and `balanced` preset for most use cases
- Enable DeepCache for videos longer than 2 seconds
- Use `ultra` mode for quick previews before final rendering
- Combine with lower inference steps (10-15) for additional speed
- Higher VRAM GPUs (RTX 4090) can use more aggressive settings

## Performance Notes

- Actual speedup depends on GPU, video length, and resolution
- First frame may be slower due to cache initialization
- Memory usage is slightly higher with DeepCache enabled
- Quality impact is minimal with `fast` mode
- `Ultra` mode is best for previews, not final output