# ComfyUI LatentSync MEMSAFE - Detailed Tutorial

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding Parameters](#understanding-parameters)
3. [Workflow Examples](#workflow-examples)
4. [Advanced Techniques](#advanced-techniques)
5. [Optimization Strategies](#optimization-strategies)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### First Time Setup

1. **Verify GPU Compatibility**
   ```bash
   # Check CUDA version
   nvidia-smi
   # Should show CUDA 11.7 or higher
   ```

2. **Install Prerequisites**
   ```bash
   # Ensure FFmpeg is installed
   ffmpeg -version
   
   # Verify Python version (3.8-3.11)
   python --version
   ```

3. **Model Setup**
   After downloading models, your structure should look like:
   ```
   comfyui_latentsync_memsafe/
   ├── checkpoints/
   │   ├── latentsync_unet.pt (3.4GB)
   │   ├── whisper/
   │   │   └── tiny.pt (39MB)
   │   └── vae/
   │       └── sd-vae-ft-mse.safetensors (335MB)
   ```

### Your First Lip Sync

1. **Basic Workflow Setup**
   - Add "Load Video" node → Connect to video file
   - Add "Load Audio" node → Connect to audio file
   - Add "🎬 LatentSync 1.6 (MEMSAFE)" node
   - Connect video → images, audio → audio
   - Add "Preview Image" or "Save Video" node

2. **Starter Settings**
   ```yaml
   seed: 1247
   lips_expression: 1.5
   inference_steps: 20
   attention_mode: flex
   # Leave other settings as default
   ```

## Understanding Parameters

### Core Parameters Deep Dive

#### 🎯 lips_expression (1.0-3.0)
Controls how pronounced the lip movements are:

- **1.0-1.2**: Minimal movement
  - Use for: Background characters, subtle speech
  - Example: News anchor in distance

- **1.3-1.7**: Natural speech
  - Use for: Normal conversation, interviews
  - Example: Talking head videos, podcasts

- **1.8-2.5**: Enhanced movement
  - Use for: Singing, emotional speech
  - Example: Music videos, dramatic scenes

- **2.6-3.0**: Exaggerated
  - Use for: Animation, special effects
  - Example: Cartoon characters, stylized content

#### 🔧 inference_steps (1-999)
Balances quality vs speed:

```
10 steps  = 2x faster, 70% quality
15 steps  = 1.5x faster, 85% quality  
20 steps  = Baseline, 95% quality (recommended)
25 steps  = 0.8x speed, 98% quality
30+ steps = 0.5x speed, 99%+ quality
```

**Pro tip**: Use 15 steps for previews, 20-25 for final renders

#### 🧠 attention_mode Selection

**FlexAttention (flex)** - Our Custom Implementation
```python
Pros:
- Superior lip sync accuracy
- Emphasizes mouth region (2x weight)
- Better audio-visual alignment
- Smart caching for patterns

Cons:
- Requires PyTorch 2.5+
- 10-15% slower than Flash
- Higher memory usage

When to use:
- Final production renders
- Close-up face shots
- Music videos with precise sync
- Quality is top priority
```

**Flash Attention (flash)** - Speed Optimized
```python
Pros:
- 20-30% faster processing
- Lower memory usage
- Stable and tested
- Works with most GPUs

Cons:
- Slightly less accurate lip sync
- No custom attention patterns

When to use:
- Long videos (3+ minutes)
- Batch processing
- Limited VRAM (8-10GB)
- Speed is priority
```

### Memory Management Explained

#### memory_mode Settings

**Conservative Mode**
```yaml
Best for:
- GPUs with 6-8GB VRAM
- Videos over 200 frames
- Shared GPU systems
- Maximum stability

Trade-offs:
- 30-40% slower processing
- More disk I/O
- Consistent memory usage
```

**Balanced Mode** (Recommended)
```yaml
Best for:
- GPUs with 10-16GB VRAM
- Most use cases
- Good speed/stability balance

Trade-offs:
- Adaptive behavior
- Occasional disk caching
- Optimal for most videos
```

**Aggressive Mode**
```yaml
Best for:
- GPUs with 20GB+ VRAM
- Short videos (<100 frames)
- Maximum speed needed
- Dedicated GPU systems

Trade-offs:
- Higher crash risk
- Maximum memory usage
- 30-40% faster
```

## Workflow Examples

### Example 1: Professional Interview Video

**Scenario**: 5-minute interview, clear speech, professional quality

```yaml
# Node Settings
seed: 1247
lips_expression: 1.4
inference_steps: 25
attention_mode: flex
optimization_level: balanced
memory_mode: balanced
output_mode: auto
vram_fraction: 0.0 (auto)
```

**Workflow**:
1. Load interview video
2. Load cleaned audio (noise removed)
3. Process with above settings
4. Save as high-quality MP4

### Example 2: Music Video with Singing

**Scenario**: 3-minute song, expressive performance

```yaml
# Node Settings
seed: 2468
lips_expression: 2.2
inference_steps: 30
attention_mode: flex
optimization_level: balanced
memory_mode: conservative
output_mode: video_file
```

**Tips**:
- Pre-process audio to enhance vocals
- Use higher lips_expression for singing
- Conservative memory for stability

### Example 3: Batch Processing Multiple Clips

**Scenario**: 50 short clips for social media

```yaml
# Node Settings
seed: -1 (random each time)
lips_expression: 1.6
inference_steps: 15
attention_mode: flash
optimization_level: aggressive
memory_mode: balanced
output_mode: video_file
```

**Automation Setup**:
```python
# Use ComfyUI's queue system
# Process one at a time to avoid OOM
# Save to organized folders
```

### Example 4: Ultra-Long Educational Video

**Scenario**: 30-minute lecture video

```yaml
# Node Settings
seed: 5555
lips_expression: 1.3
inference_steps: 20
attention_mode: flash
optimization_level: balanced
memory_mode: conservative
output_mode: video_file
enable_disk_cache: true
vram_fraction: 0.7
```

**Special Considerations**:
- Split into 5-minute segments if needed
- Use disk cache for stability
- Monitor temperature during long runs

## Advanced Techniques

### Technique 1: Multi-Pass Quality Enhancement

For maximum quality on important segments:

**Pass 1**: Quick preview
```yaml
inference_steps: 10
attention_mode: flash
# Generate preview to check sync
```

**Pass 2**: Final render
```yaml
inference_steps: 30
attention_mode: flex
# Use same seed as preview
```

### Technique 2: Face Region Optimization

For videos with small or distant faces:

1. **Pre-crop the face region**
   - Use video editor to crop close to face
   - Process cropped version
   - Composite back to original

2. **Adjust parameters**
   ```yaml
   lips_expression: 1.8 (increase for small faces)
   inference_steps: 25 (more steps for detail)
   ```

### Technique 3: Audio Preprocessing

Improve sync quality with audio preparation:

```bash
# Remove background noise
ffmpeg -i input.mp3 -af "highpass=f=200,lowpass=f=3000" clean_voice.mp3

# Normalize audio levels
ffmpeg -i clean_voice.mp3 -af loudnorm output.mp3

# Extract voice frequency range
ffmpeg -i input.mp3 -af "bandpass=f=300:width_type=h:width=2700" voice_only.mp3
```

### Technique 4: GPU Memory Optimization

Monitor and optimize GPU usage:

```python
# Before processing
torch.cuda.empty_cache()

# Monitor during processing
nvidia-smi -l 1  # Watch memory usage

# Settings for maximum efficiency
vram_fraction: 0.8  # Leave 20% free
memory_mode: "balanced"
optimization_level: "balanced"
```

## Optimization Strategies

### For RTX 4090 Users

Maximize your powerful GPU:

```yaml
# Optimal 4090 Settings
attention_mode: flex  # Quality priority
optimization_level: aggressive
memory_mode: aggressive
inference_steps: 25
batch_size: 20  # Auto-set based on profile
vram_fraction: 0.9
```

Expected performance:
- 50 frames: ~35 seconds
- 200 frames: ~140 seconds
- 500 frames: ~350 seconds

### For RTX 3080/3070 Users

Balance performance and stability:

```yaml
# Optimal 3080 Settings (10GB)
attention_mode: flash
optimization_level: balanced
memory_mode: balanced
inference_steps: 20
vram_fraction: 0.75

# Optimal 3070 Settings (8GB)
attention_mode: flash
optimization_level: balanced  
memory_mode: conservative
inference_steps: 18
vram_fraction: 0.7
```

### For Older/Lower VRAM GPUs

Maximum compatibility:

```yaml
# 6GB VRAM Settings
attention_mode: standard
optimization_level: conservative
memory_mode: conservative
inference_steps: 15
output_mode: video_file
enable_disk_cache: true
vram_fraction: 0.6
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Immediate Solutions**:
1. Set `output_mode="video_file"`
2. Set `memory_mode="conservative"`
3. Enable `disk_cache=true`
4. Reduce `vram_fraction` to 0.6

**If still failing**:
```bash
# Clear GPU memory
nvidia-smi
# Find and kill other GPU processes

# Reduce video resolution
ffmpeg -i input.mp4 -vf scale=512:512 small.mp4
```

### Issue: "Poor lip sync quality"

**Diagnostic Steps**:
1. Check audio quality (clear speech?)
2. Verify face is clearly visible
3. Ensure audio/video are synced

**Solutions**:
```yaml
# Increase quality settings
lips_expression: 2.0  # Higher expression
inference_steps: 30   # More steps
attention_mode: flex  # Best quality

# Preprocess audio
# Remove noise, enhance speech frequencies
```

### Issue: "Processing is too slow"

**Speed Optimization Checklist**:
- ✅ Use `attention_mode="flash"`
- ✅ Set `optimization_level="aggressive"`
- ✅ Reduce `inference_steps` to 15
- ✅ Ensure no other GPU tasks running
- ✅ Check GPU thermal throttling
- ✅ Use hardware encoding (automatic)

### Issue: "Node not appearing in ComfyUI"

**Troubleshooting Steps**:
1. Check folder name is correct
2. Restart ComfyUI completely
3. Check console for errors
4. Verify Python version compatibility
5. Manually install requirements:
   ```bash
   cd comfyui_latentsync_memsafe
   pip install -r requirements.txt
   ```

## Best Practices

### 1. Video Preparation
- **Resolution**: 512x512 optimal, up to 1024x1024 supported
- **Format**: MP4 with H.264 codec
- **Frame rate**: 25-30 fps recommended
- **Face position**: Centered, well-lit, clear

### 2. Audio Preparation  
- **Format**: MP3 or WAV
- **Sample rate**: 16kHz or higher
- **Quality**: Clean speech, minimal background noise
- **Sync**: Ensure audio matches video timing

### 3. Workflow Organization
```
project/
├── inputs/
│   ├── video/
│   └── audio/
├── outputs/
│   ├── previews/
│   └── final/
└── settings/
    └── presets.json
```

### 4. Performance Monitoring
```python
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Track processing time
import time
start = time.time()
# ... processing ...
print(f"Processed in {time.time()-start:.1f} seconds")
```

### 5. Quality Assurance
- Always preview with low settings first
- Check lip sync at multiple points
- Verify audio levels are consistent
- Test on target playback device

## Conclusion

ComfyUI LatentSync MEMSAFE provides professional-grade lip sync with unprecedented stability and performance. By understanding the parameters and following these guidelines, you can achieve excellent results for any use case.

Remember:
- Start with default settings
- Adjust based on your specific needs
- Monitor GPU resources
- Preprocess media when needed
- Use appropriate attention mode

Happy lip syncing! 🎬🔊