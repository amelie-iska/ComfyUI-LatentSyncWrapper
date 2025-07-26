# TensorRT Acceleration Guide

This guide explains how to convert the LatentSync 1.6 U-Net model into an optimized TensorRT engine.

## Conceptual Overview

We first export the PyTorch model to the ONNX format, which is an open standard for representing neural networks. ONNX models can be consumed by NVIDIA TensorRT, a high-performance deep learning inference SDK. TensorRT compiles the network into an optimized engine that leverages Tensor Cores, layer fusion, and other hardware-specific features for improved throughput and reduced latency compared to running the model directly in PyTorch.

The overall pipeline is:

```
PyTorch (latentsync_unet.pt) -> ONNX -> TensorRT Engine -> Inference
```

By using TensorRT you can achieve faster inference and lower VRAM usage, especially on GPUs with Tensor Core support.

## 1. Export PyTorch Model to ONNX

Create a script `export_unet_to_onnx.py` with the following content:

```python
import os
import torch
from latentsync.models.unet import UNet3DConditionModel

# Path to the original checkpoint
CKPT_PATH = os.path.join('checkpoints', 'latentsync_unet.pt')
ONNX_PATH = 'latentsync_unet.onnx'

def load_unet(path: str) -> torch.nn.Module:
    state = torch.load(path, map_location='cpu')
    model = UNet3DConditionModel(**state['config'])
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

def main():
    model = load_unet(CKPT_PATH)

    dummy = torch.randn(1, 4, 2, 64, 64)  # (batch, channels, frames, H, W)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=['latents'],
        output_names=['noise_pred'],
        opset_version=17,
        dynamic_axes={'latents': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
                      'noise_pred': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'}},
    )
    print(f"ONNX model saved to {ONNX_PATH}")

if __name__ == '__main__':
    main()
```

Run the script:

```bash
python export_unet_to_onnx.py
```

This creates `latentsync_unet.onnx` in the working directory.

## 2. Compile ONNX to TensorRT Engine

TensorRT can be used via the ONNX Runtime execution provider. The following script builds the engine and saves it for later use.

Create `build_trt_engine.py`:

```python
import onnxruntime as ort

ONNX_PATH = 'latentsync_unet.onnx'
ENGINE_PATH = 'latentsync_unet.plan'

providers = [
    (
        'TensorrtExecutionProvider',
        {
            'trt_max_workspace_size': str(8 << 30),  # 8GB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': '.',
        },
    ),
    'CUDAExecutionProvider',
]

sess = ort.InferenceSession(ONNX_PATH, providers=providers)

# Building the TensorRT engine happens during the first session creation.
# Saving the engine to disk allows reusing it without recompilation.
engine_bytes = sess.get_tensorrt_engine()
with open(ENGINE_PATH, 'wb') as f:
    f.write(engine_bytes)
print(f"TensorRT engine saved to {ENGINE_PATH}")
```

### Precision Options

- **FP16 (default)**: good balance between speed and precision. Requires a GPU with Tensor Core support.
- **INT8**: offers further speed and VRAM reductions but may introduce slight accuracy loss. Calibration with representative data is required. Set `trt_int8_enable=True` and provide calibration data via ONNX Runtime APIs.

### Engine Type

The above configuration builds a dynamic engine that supports varying batch sizes and resolutions. To build a static engine for a specific input shape, set the shapes explicitly when exporting to ONNX and remove the `dynamic_axes` dictionary. Static engines can be slightly faster but are limited to the fixed dimensions.

## 3. Run Inference with the TensorRT Engine

After building the engine, you can load it and run inference:

Create `infer_with_trt.py`:

```python
import numpy as np
import onnxruntime as ort

ENGINE_PATH = 'latentsync_unet.plan'

providers = [
    (
        'TensorrtExecutionProvider',
        {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': '.',
        },
    ),
    'CUDAExecutionProvider',
]

sess = ort.InferenceSession(ENGINE_PATH, providers=providers)

# Dummy input (batch=1, channels=4, frames=2, height=64, width=64)
latents = np.random.randn(1, 4, 2, 64, 64).astype(np.float32)

outputs = sess.run(None, {'latents': latents})
print('Output shape:', outputs[0].shape)
```

Running:

```bash
python infer_with_trt.py
```

will execute the model using TensorRT.

## Notes on Accuracy vs. Speed

- **FP16** generally provides a large performance boost with minimal quality degradation.
- **INT8** delivers the highest speed and memory savings but requires careful calibration and may slightly reduce lip-sync accuracy.
- **Dynamic vs. Static**: Dynamic engines handle multiple resolutions or batch sizes but incur a small runtime overhead. If your deployment uses a fixed resolution, consider a static engine for maximum throughput.

These scripts should serve as a starting point for accelerating the LatentSync 1.6 U-Net on NVIDIA GPUs.
