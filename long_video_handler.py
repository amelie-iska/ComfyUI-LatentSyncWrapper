"""
Long video processing handler with disk-based frame extraction
Optimized for handling videos that exceed GPU memory limits
"""
import os
import shutil
import tempfile
import subprocess
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Generator
import json

class LongVideoHandler:
    """Handles processing of long videos using disk-based frame storage"""
    
    def __init__(self, temp_base_dir: str, memory_mode: str = "balanced"):
        """
        Args:
            temp_base_dir: Base directory for temporary files
            memory_mode: "aggressive" | "balanced" | "conservative"
        """
        self.temp_base_dir = temp_base_dir
        self.memory_mode = memory_mode
        self.frame_dir = None
        self.metadata = {}
        
        # Memory mode thresholds
        self.thresholds = {
            "aggressive": {"max_frames_in_memory": 32, "max_video_length": 100},
            "balanced": {"max_frames_in_memory": 16, "max_video_length": 200},
            "conservative": {"max_frames_in_memory": 8, "max_video_length": 500}
        }
    
    def should_use_disk_processing(self, num_frames: int) -> bool:
        """Determine if video should be processed using disk-based approach"""
        threshold = self.thresholds[self.memory_mode]["max_video_length"]
        return num_frames > threshold
    
    def extract_frames_to_disk(self, video_path: str, fps: float = 25.0) -> Tuple[str, dict]:
        """Extract video frames to disk using FFmpeg
        
        Returns:
            Tuple of (frame_directory, metadata)
        """
        # Create unique frame directory
        import uuid
        self.frame_dir = os.path.join(self.temp_base_dir, f"frames_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.frame_dir, exist_ok=True)
        
        # Extract frames using FFmpeg
        output_pattern = os.path.join(self.frame_dir, "frame_%06d.png")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-r", str(fps),  # Force output FPS
            "-pix_fmt", "rgb24",
            "-start_number", "0",
            output_pattern,
            "-loglevel", "error"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames: {e}")
            raise
        
        # Count extracted frames
        frame_files = sorted([f for f in os.listdir(self.frame_dir) if f.endswith('.png')])
        num_frames = len(frame_files)
        
        # Get video info
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(result.stdout)["streams"][0]
        
        self.metadata = {
            "num_frames": num_frames,
            "fps": fps,
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "frame_dir": self.frame_dir,
            "frame_pattern": "frame_%06d.png"
        }
        
        return self.frame_dir, self.metadata
    
    def load_frame_batch(self, start_idx: int, batch_size: int) -> torch.Tensor:
        """Load a batch of frames from disk
        
        Returns:
            Tensor of shape (batch_size, height, width, channels)
        """
        frames = []
        
        for i in range(start_idx, min(start_idx + batch_size, self.metadata["num_frames"])):
            frame_path = os.path.join(self.frame_dir, f"frame_{i:06d}.png")
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                frame = np.array(img).astype(np.float32) / 255.0
                img.close()  # Fix: Close PIL Image to prevent memory leak
                frames.append(frame)
            else:
                print(f"Warning: Frame {i} not found at {frame_path}")
        
        if not frames:
            raise ValueError(f"No frames found in batch starting at {start_idx}")
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(np.stack(frames))
        return frames_tensor
    
    def process_in_chunks(self, num_frames: int, chunk_size: Optional[int] = None) -> Generator[Tuple[int, int], None, None]:
        """Generate chunk indices for processing
        
        Yields:
            Tuples of (start_idx, end_idx) for each chunk
        """
        if chunk_size is None:
            chunk_size = self.thresholds[self.memory_mode]["max_frames_in_memory"]
        
        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            yield start_idx, end_idx
    
    def save_processed_frames(self, frames: torch.Tensor, start_idx: int, output_dir: str):
        """Save processed frames back to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_idx = start_idx + i
            output_path = os.path.join(output_dir, f"processed_{frame_idx:06d}.png")
            
            # Convert tensor to PIL Image
            if frame.dim() == 3:
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(frame_np)
                img.save(output_path)
                img.close()  # Fix: Close PIL Image to prevent memory leak
    
    def combine_frames_to_video(self, frame_dir: str, output_path: str, fps: float = 25.0, audio_path: Optional[str] = None):
        """Combine processed frames back into a video using FFmpeg"""
        frame_pattern = os.path.join(frame_dir, "processed_%06d.png")
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-r", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18"
        ]
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])
        
        cmd.append(output_path)
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error combining frames to video: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary frame directories"""
        if self.frame_dir and os.path.exists(self.frame_dir):
            try:
                shutil.rmtree(self.frame_dir)
                print(f"Cleaned up frame directory: {self.frame_dir}")
            except Exception as e:
                print(f"Error cleaning up frame directory: {e}")


class VideoLengthAdjuster:
    """Adjust video length to match audio duration with various modes"""
    
    @staticmethod
    def adjust_video_length(frames: List[np.ndarray], audio_duration: float, 
                          video_fps: float, mode: str = "normal") -> List[np.ndarray]:
        """
        Adjust video frames to match audio duration
        
        Args:
            frames: List of video frames
            audio_duration: Duration of audio in seconds
            video_fps: Video frames per second
            mode: "normal" | "pingpong" | "loop"
        
        Returns:
            Adjusted list of frames
        """
        num_frames = len(frames)
        target_frames = int(audio_duration * video_fps)
        
        if mode == "normal":
            # Pad with last frame or truncate
            if num_frames < target_frames:
                # Repeat last frame
                last_frame = frames[-1]
                padding = [last_frame] * (target_frames - num_frames)
                return frames + padding
            else:
                # Truncate
                return frames[:target_frames]
        
        elif mode == "pingpong":
            # Create forward-backward loop
            adjusted_frames = []
            forward = True
            idx = 0
            
            for _ in range(target_frames):
                adjusted_frames.append(frames[idx])
                
                if forward:
                    idx += 1
                    if idx >= num_frames - 1:
                        forward = False
                else:
                    idx -= 1
                    if idx <= 0:
                        forward = True
            
            return adjusted_frames
        
        elif mode == "loop":
            # Simple loop
            adjusted_frames = []
            for i in range(target_frames):
                idx = i % num_frames
                adjusted_frames.append(frames[idx])
            return adjusted_frames
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @staticmethod
    def adjust_on_disk(frame_dir: str, audio_duration: float, video_fps: float, 
                       mode: str, output_dir: str) -> int:
        """Adjust video length for disk-based processing"""
        # Count existing frames
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_')])
        num_frames = len(frame_files)
        target_frames = int(audio_duration * video_fps)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if mode == "normal":
            # Copy frames, padding with last frame if needed
            for i in range(target_frames):
                if i < num_frames:
                    src = os.path.join(frame_dir, frame_files[i])
                else:
                    src = os.path.join(frame_dir, frame_files[-1])
                
                dst = os.path.join(output_dir, f"adjusted_{i:06d}.png")
                shutil.copy2(src, dst)
        
        elif mode == "pingpong":
            # Pingpong pattern
            forward = True
            idx = 0
            
            for i in range(target_frames):
                src = os.path.join(frame_dir, frame_files[idx])
                dst = os.path.join(output_dir, f"adjusted_{i:06d}.png")
                shutil.copy2(src, dst)
                
                if forward:
                    idx += 1
                    if idx >= num_frames - 1:
                        forward = False
                else:
                    idx -= 1
                    if idx <= 0:
                        forward = True
        
        elif mode == "loop":
            # Simple loop
            for i in range(target_frames):
                idx = i % num_frames
                src = os.path.join(frame_dir, frame_files[idx])
                dst = os.path.join(output_dir, f"adjusted_{i:06d}.png")
                shutil.copy2(src, dst)
        
        return target_frames


class ProgressiveVideoProcessor:
    """Process video progressively with checkpoint support"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, "progress.json")
    
    def save_checkpoint(self, current_chunk: int, total_chunks: int, metadata: dict):
        """Save processing checkpoint"""
        checkpoint = {
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "metadata": metadata,
            "timestamp": os.path.getmtime(self.checkpoint_file) if os.path.exists(self.checkpoint_file) else 0
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load existing checkpoint if available"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def can_resume(self) -> bool:
        """Check if we can resume from checkpoint"""
        checkpoint = self.load_checkpoint()
        return checkpoint is not None and checkpoint["current_chunk"] < checkpoint["total_chunks"]