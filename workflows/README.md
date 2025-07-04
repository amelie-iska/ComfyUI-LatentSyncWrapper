# MEMSAFE LatentSync Workflows

This folder contains optimized workflows for the LatentSync 1.6 MEMSAFE nodes.

## 🎭 MEMSAFE Complete Workflow

**File:** `MEMSAFE_Complete_Workflow.json`

### Overview
A comprehensive workflow showcasing all MEMSAFE nodes working together for standard video processing.

### Workflow Structure:
1. **GPU Configuration**
   - 📊 GPU Benchmark - Tests your GPU performance
   - 🖥️ GPU Configuration - Sets 85% VRAM usage for optimal performance

2. **Input Loading**
   - Load Image/Video frames
   - Load Audio file

3. **Pre-Processing**
   - 🎬 Video Length Adjuster - Syncs video length to audio duration

4. **LatentSync Processing**
   - 🎭 LatentSync 1.6 (MEMSAFE) with:
     - Balanced optimization
     - Balanced memory mode
     - Disk cache enabled
     - Auto output mode

5. **Output Options**
   - Save Audio (resampled to 16kHz)
   - Save Image (if frames returned)
   - 💾 Copy/Save Video File (if path returned)

### Best For:
- Videos up to 200 frames
- General purpose lip-sync
- Users who want flexibility in output format

---

## 🎞️ Long Video MEMSAFE Workflow

**File:** `MEMSAFE_LongVideo_Workflow.json`

### Overview
Specialized workflow for processing very long videos (500+ frames) without memory issues.

### Workflow Structure:
1. **Efficient Loading**
   - 📁 Efficient Video Loader - Loads video in 50-frame batches

2. **Chunk Processing**
   - 🎞️ Video Chunk Processor - Splits into 100-frame chunks with 10-frame overlap
   - 🎬 Video Length Adjuster - Ensures audio sync

3. **Conservative Processing**
   - 🎭 LatentSync 1.6 (MEMSAFE) with:
     - Conservative memory mode
     - 70% VRAM limit
     - Disk cache enabled
     - Forced video_file output
     - Reduced inference steps (15 vs 20)

4. **Space-Efficient Output**
   - 💾 Move (not copy) to final location
   - Preview first 100 frames

### Best For:
- Videos with 500+ frames
- Limited VRAM systems
- Processing hours of footage

---

## Usage Tips

### Memory Management
- **Short videos (<100 frames)**: Use "aggressive" memory mode
- **Medium videos (100-300 frames)**: Use "balanced" memory mode
- **Long videos (300+ frames)**: Use "conservative" memory mode

### VRAM Settings
- **RTX 4090/3090 (24GB)**: 0.85-0.95 vram_fraction
- **RTX 3080/4070 (10-12GB)**: 0.70-0.85 vram_fraction
- **RTX 3060/4060 (8GB)**: 0.60-0.70 vram_fraction

### Output Modes
- **"auto"**: Let the system decide based on video length
- **"frames"**: For further processing in ComfyUI
- **"video_file"**: For direct saving (memory efficient)

### Optimization Levels
- **"conservative"**: Safest, slightly slower
- **"balanced"**: Good speed/quality trade-off
- **"aggressive"**: Fastest, may reduce quality slightly

## Loading Workflows

1. In ComfyUI, click "Load" in the menu
2. Navigate to: `ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/workflows/`
3. Select the desired workflow JSON file
4. Replace example inputs with your files
5. Adjust settings as needed
6. Click "Queue Prompt" to run

## Troubleshooting

### Out of Memory Errors
1. Reduce `vram_fraction` (try 0.6)
2. Switch to "conservative" memory mode
3. Enable `disk_cache`
4. Use "video_file" output mode
5. Reduce batch size in Video Loader

### Slow Processing
1. Increase `vram_fraction` (if you have headroom)
2. Switch to "aggressive" optimization level
3. Reduce `inference_steps` to 15 or 10
4. Use "aggressive" memory mode (if not OOM)

### Audio Sync Issues
1. Ensure audio is properly loaded
2. Try different Video Length Adjuster modes
3. Check that FPS matches your video (default 25)