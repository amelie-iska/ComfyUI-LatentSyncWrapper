"""
Enhanced Video Length Adjuster Node for ComfyUI
Provides multiple modes for adjusting video length to match audio
"""
import torch
import numpy as np
from typing import Tuple
import torchaudio
from .long_video_handler import VideoLengthAdjuster

class VideoLengthAdjusterNode:
    """
    Adjust video length to match audio duration with multiple modes
    Similar to VideoBasicLatentSync but with our optimizations
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust_length"
    CATEGORY = "LatentSyncNode"
    
    def adjust_length(self, images, audio, mode="normal", fps=25.0, silent_padding_sec=0.5):
        """
        Adjust video frames to match audio duration
        
        Args:
            images: Video frames tensor (batch, height, width, channels)
            audio: Audio data dict with 'waveform' and 'sample_rate'
            mode: Adjustment mode - "normal", "pingpong", or "loop_to_audio"
            fps: Target frames per second
            silent_padding_sec: Silent padding to add at the end
        """
        # Extract audio info - handle both dict and tuple formats
        if isinstance(audio, dict):
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        else:
            waveform = audio["waveform"] if hasattr(audio, "__getitem__") else audio[0]
            sample_rate = audio["sample_rate"] if hasattr(audio, "__getitem__") else audio[1]
        
        # Ensure waveform is 2D
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        audio_duration = waveform.shape[1] / sample_rate
        
        # Convert mode name
        if mode == "loop_to_audio":
            mode = "loop"
        
        # Convert images tensor to list of frames
        if images.dim() == 4:
            frames_list = [images[i].cpu().numpy() for i in range(images.shape[0])]
        else:
            frames_list = [images.cpu().numpy()]
        
        # Calculate target duration with padding
        target_duration = audio_duration + silent_padding_sec
        
        # Adjust video length based on mode
        adjusted_frames = VideoLengthAdjuster.adjust_video_length(
            frames_list, target_duration, fps, mode
        )
        
        # Convert back to tensor
        adjusted_tensor = torch.stack([torch.from_numpy(f) for f in adjusted_frames])
        
        # Add silent padding to audio if requested
        if silent_padding_sec > 0:
            silence_samples = int(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_waveform = torch.cat([waveform, silence], dim=1)
        else:
            padded_waveform = waveform
            
        # Return in ComfyUI audio format
        audio_output = {
            "waveform": padded_waveform.unsqueeze(0) if padded_waveform.dim() == 2 else padded_waveform,
            "sample_rate": sample_rate
        }
        
        return (adjusted_tensor, audio_output)


class VideoChunkProcessor:
    """
    Process video in chunks for very long videos
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "chunk_size": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "overlap": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("VIDEO_CHUNKS",)
    RETURN_NAMES = ("chunks",)
    FUNCTION = "create_chunks"
    CATEGORY = "LatentSyncNode"
    
    def create_chunks(self, images, chunk_size=100, overlap=10):
        """Split video into overlapping chunks"""
        total_frames = images.shape[0]
        chunks = []
        
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunk = images[start:end]
            
            chunks.append({
                "frames": chunk,
                "start_idx": start,
                "end_idx": end,
                "overlap": overlap if start > 0 else 0
            })
            
            start = end - overlap if end < total_frames else end
        
        return (chunks,)


# Node mappings for registration
VIDEO_NODE_CLASS_MAPPINGS = {
    "VideoLengthAdjuster": VideoLengthAdjusterNode,  # Legacy name for compatibility
    "VideoLengthAdjusterNode": VideoLengthAdjusterNode,
    "VideoChunkProcessor": VideoChunkProcessor,
}

VIDEO_NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLengthAdjuster": "🎬 Video Length Adjuster (MEMSAFE)",  # Legacy name for compatibility
    "VideoLengthAdjusterNode": "🎬 Video Length Adjuster Enhanced (MEMSAFE)",
    "VideoChunkProcessor": "🎞️ Video Chunk Processor (MEMSAFE)",
}