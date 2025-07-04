import os
import shutil
from datetime import datetime
import folder_paths
import numpy as np
import torch
from PIL import Image
import cv2

class CopyVideoFile:
    """Copy or move a video file to a custom location with optional renaming"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "latentsync_output"}),
                "save_location": (["outputs", "input", "custom"], {"default": "outputs"}),
                "custom_path": ("STRING", {"default": "", "multiline": False}),
                "operation": (["copy", "move"], {"default": "copy"}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "preview_frames": ("INT", {"default": 8, "min": 1, "max": 32}),
            }
        }
    
    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_path", "preview")
    FUNCTION = "copy_video"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False)
    
    def copy_video(self, video_path, filename_prefix="latentsync_output", 
                   save_location="outputs", custom_path="", operation="copy", 
                   add_timestamp=True, overwrite=False, preview_frames=8):
        """Copy or move video file to specified location"""
        
        # Validate input video exists
        if not video_path:
            raise ValueError("No video path provided")
        
        # Convert to string if it's not already
        video_path = str(video_path).strip()
        
        print(f"CopyVideoFile received video_path: '{video_path}'")
        print(f"Path exists: {os.path.exists(video_path)}")
        
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Get file extension
        _, ext = os.path.splitext(video_path)
        if not ext:
            ext = ".mp4"  # Default to mp4 if no extension
        
        # Build filename
        filename = filename_prefix
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        filename = f"{filename}{ext}"
        
        # Sanitize filename to prevent path injection
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        
        # Determine output directory
        if save_location == "outputs":
            output_dir = folder_paths.get_output_directory()
        elif save_location == "input":
            output_dir = folder_paths.get_input_directory()
        elif save_location == "custom" and custom_path:
            output_dir = custom_path
            # Create custom directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Fallback to outputs if custom path is empty
            output_dir = folder_paths.get_output_directory()
        
        # Build full output path
        output_path = os.path.join(output_dir, filename)
        output_path = os.path.abspath(output_path)
        
        # Check if file already exists
        if os.path.exists(output_path) and not overwrite:
            # Add counter to filename to avoid overwriting
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(output_path):
                filename = f"{base_name}_{counter}{ext}"
                output_path = os.path.join(output_dir, filename)
                counter += 1
        
        # Perform the operation
        try:
            if operation == "copy":
                shutil.copy2(video_path, output_path)
                print(f"Copied video to: {output_path}")
            else:  # move
                shutil.move(video_path, output_path)
                print(f"Moved video to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to {operation} video: {str(e)}")
        
        # Generate preview frames (memory-safe approach)
        preview = self.generate_preview(output_path, max_frames=preview_frames)
        
        print(f"Preview shape: {preview.shape}, dtype: {preview.dtype}, range: [{preview.min():.3f}, {preview.max():.3f}]")
        
        # Return the saved path and preview
        return (output_path, preview)
    
    def generate_preview(self, video_path, max_frames=16, target_size=(512, 512)):
        """Generate preview frames from video in a memory-safe way"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video for preview: {video_path}")
                # Return a blank frame if video can't be opened
                blank = np.zeros((1, target_size[1], target_size[0], 3), dtype=np.float32)
                return torch.from_numpy(blank)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate which frames to extract (evenly spaced)
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                # Evenly sample frames across the video
                step = total_frames / max_frames
                frame_indices = [int(i * step) for i in range(max_frames)]
            
            frames = []
            for idx in frame_indices:
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size maintaining aspect ratio
                    h, w = frame.shape[:2]
                    if h > w:
                        new_h = target_size[1]
                        new_w = int(w * (new_h / h))
                    else:
                        new_w = target_size[0]
                        new_h = int(h * (new_w / w))
                    
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Pad to exact target size
                    top = (target_size[1] - new_h) // 2
                    bottom = target_size[1] - new_h - top
                    left = (target_size[0] - new_w) // 2
                    right = target_size[0] - new_w - left
                    
                    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, 
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    
                    # Convert to float32 and normalize to [0, 1]
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                # Break if we have enough frames
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            
            if not frames:
                # Return a blank frame if no frames could be extracted
                blank = np.zeros((1, target_size[1], target_size[0], 3), dtype=np.float32)
                return torch.from_numpy(blank)
            
            # Stack frames and convert to torch tensor
            # ComfyUI expects (batch, height, width, channels)
            frames_array = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_array)
            
            # Ensure the tensor is contiguous and in the right format
            frames_tensor = frames_tensor.contiguous()
            
            return frames_tensor
            
        except Exception as e:
            print(f"Error generating preview: {str(e)}")
            # Return a blank frame on error
            blank = np.zeros((1, target_size[1], target_size[0], 3), dtype=np.float32)
            return torch.from_numpy(blank)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        # This ensures the node always executes when the workflow runs
        return float("nan")


# Node Mappings for this module
COPY_VIDEO_NODE_CLASS_MAPPINGS = {
    "CopyVideoFile": CopyVideoFile,
}

COPY_VIDEO_NODE_DISPLAY_NAME_MAPPINGS = {
    "CopyVideoFile": "💾 Copy/Save Video File (MEMSAFE)",
}