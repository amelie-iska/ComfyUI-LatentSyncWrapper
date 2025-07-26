import os
import torch

# Default precision settings for each component
quantization_config = {
    "whisper": "fp16",
    "vae": "int8",
    "syncnet": "int8",
    "unet": "fp8",
    "videomae": "int8",
    "face_detector": "int8",
}

# Mapping of precision strings to torch dtypes
_precision_map = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8": getattr(torch, "float8_e4m3fn", torch.float16),
}


def _cast_module(module: torch.nn.Module, precision: str) -> torch.nn.Module:
    """Attempt to cast a module to the requested precision."""
    if precision == "int8":
        try:
            from torch.ao.quantization import quantize_dynamic

            return quantize_dynamic(module, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            return module
    dtype = _precision_map.get(precision)
    if dtype is None:
        return module
    try:
        return module.to(dtype)
    except Exception as e:
        print(f"Casting to {precision} failed: {e}")
        return module


def apply_pipeline_quantization(pipeline, config=None):
    """Apply quantization to pipeline modules based on config."""
    config = config or quantization_config

    if hasattr(pipeline, "audio_encoder") and pipeline.audio_encoder is not None:
        model = getattr(pipeline.audio_encoder, "model", pipeline.audio_encoder)
        pipeline.audio_encoder.model = _cast_module(model, config.get("whisper", "fp16"))

    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae = _cast_module(pipeline.vae, config.get("vae", "int8"))

    if hasattr(pipeline, "unet") and pipeline.unet is not None:
        pipeline.unet = _cast_module(pipeline.unet, config.get("unet", "fp8"))

    if hasattr(pipeline, "syncnet"):
        pipeline.syncnet = _cast_module(pipeline.syncnet, config.get("syncnet", "int8"))

    if hasattr(pipeline, "videomae"):
        pipeline.videomae = _cast_module(pipeline.videomae, config.get("videomae", "int8"))

    if hasattr(pipeline, "face_detector"):
        pipeline.face_detector = _cast_module(pipeline.face_detector, config.get("face_detector", "int8"))

    return pipeline
