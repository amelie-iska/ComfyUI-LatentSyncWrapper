"""
Real-time preview system for LatentSync processing
Shows intermediate results during processing for better user experience
"""

import numpy as np
import torch
from PIL import Image
import io
import base64
from typing import Optional, Callable, List, Tuple
import threading
import queue
import time


class PreviewFrame:
    """Container for preview frame data"""
    def __init__(self, frame_idx: int, image: np.ndarray, timestamp: float):
        self.frame_idx = frame_idx
        self.image = image
        self.timestamp = timestamp
        self.processing_time = 0.0
        

class LatentSyncPreviewSystem:
    """Real-time preview system for LatentSync"""
    
    def __init__(self, 
                 update_callback: Optional[Callable] = None,
                 preview_interval: int = 5,
                 max_preview_size: Tuple[int, int] = (512, 512)):
        """
        Initialize preview system
        
        Args:
            update_callback: Function to call with preview updates
            preview_interval: Show preview every N frames
            max_preview_size: Maximum size for preview images
        """
        self.update_callback = update_callback
        self.preview_interval = preview_interval
        self.max_preview_size = max_preview_size
        
        # Preview queue for thread-safe updates
        self.preview_queue = queue.Queue(maxsize=10)
        self.preview_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.preview_frames: List[PreviewFrame] = []
        
    def start(self, total_frames: int):
        """Start the preview system"""
        self.total_frames = total_frames
        self.processed_frames = 0
        self.preview_frames.clear()
        self.stop_event.clear()
        
        # Start preview thread
        self.preview_thread = threading.Thread(target=self._preview_worker)
        self.preview_thread.daemon = True
        self.preview_thread.start()
        
    def stop(self):
        """Stop the preview system"""
        self.stop_event.set()
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            
    def add_frame(self, frame_idx: int, frame: torch.Tensor, processing_time: float = 0.0):
        """
        Add a frame for preview
        
        Args:
            frame_idx: Index of the frame
            frame: Frame tensor (C, H, W) or (H, W, C)
            processing_time: Time taken to process this frame
        """
        # Check if we should show this frame
        if frame_idx % self.preview_interval != 0 and frame_idx != self.total_frames - 1:
            return
            
        # Convert tensor to numpy
        if isinstance(frame, torch.Tensor):
            frame_np = self._tensor_to_numpy(frame)
        else:
            frame_np = frame
            
        # Resize for preview
        frame_resized = self._resize_for_preview(frame_np)
        
        # Create preview frame
        preview = PreviewFrame(frame_idx, frame_resized, time.time())
        preview.processing_time = processing_time
        
        # Add to queue (non-blocking)
        try:
            self.preview_queue.put(preview, block=False)
        except queue.Full:
            # Skip if queue is full
            pass
            
    def _preview_worker(self):
        """Worker thread for processing previews"""
        while not self.stop_event.is_set():
            try:
                preview = self.preview_queue.get(timeout=0.1)
                self._process_preview(preview)
            except queue.Empty:
                continue
                
    def _process_preview(self, preview: PreviewFrame):
        """Process and send preview"""
        self.processed_frames = preview.frame_idx + 1
        self.preview_frames.append(preview)
        
        # Keep only recent previews
        if len(self.preview_frames) > 20:
            self.preview_frames = self.preview_frames[-20:]
            
        # Create preview data
        preview_data = {
            'frame_idx': preview.frame_idx,
            'total_frames': self.total_frames,
            'progress': self.processed_frames / self.total_frames,
            'image': self._encode_image(preview.image),
            'processing_time': preview.processing_time,
            'timestamp': preview.timestamp
        }
        
        # Send update
        if self.update_callback:
            self.update_callback(preview_data)
            
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        # Handle different tensor formats
        if tensor.dim() == 4:  # (B, C, H, W)
            tensor = tensor[0]
            
        if tensor.shape[0] <= 4:  # (C, H, W)
            tensor = tensor.permute(1, 2, 0)
            
        # Convert to numpy
        img = tensor.detach().cpu().numpy()
        
        # Ensure proper range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        return img
        
    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        """Resize image for preview"""
        h, w = image.shape[:2]
        max_h, max_w = self.max_preview_size
        
        # Calculate scaling factor
        scale = min(max_h / h, max_w / w, 1.0)
        
        if scale < 1.0:
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Use PIL for high-quality resizing
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            image = np.array(pil_img)
            
        return image
        
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string"""
        # Convert to PIL
        if image.shape[-1] == 1:  # Grayscale
            pil_img = Image.fromarray(image.squeeze(), mode='L')
        else:
            pil_img = Image.fromarray(image, mode='RGB')
            
        # Encode to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG', optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    def get_preview_grid(self, max_previews: int = 9) -> Optional[np.ndarray]:
        """
        Get a grid of recent preview frames
        
        Args:
            max_previews: Maximum number of previews to include
            
        Returns:
            Grid image as numpy array
        """
        if not self.preview_frames:
            return None
            
        # Get recent previews
        recent_previews = self.preview_frames[-max_previews:]
        
        # Calculate grid dimensions
        n = len(recent_previews)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Get frame size
        h, w = recent_previews[0].image.shape[:2]
        
        # Create grid
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        for i, preview in enumerate(recent_previews):
            row = i // cols
            col = i % cols
            
            y1 = row * h
            y2 = (row + 1) * h
            x1 = col * w
            x2 = (col + 1) * w
            
            # Handle grayscale
            img = preview.image
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
                
            grid[y1:y2, x1:x2] = img
            
        return grid
        
    def create_progress_video(self, output_path: str, fps: int = 10):
        """
        Create a video showing the processing progress
        
        Args:
            output_path: Path to save the video
            fps: Frames per second for the video
        """
        if not self.preview_frames:
            return
            
        import cv2
        
        # Get frame size
        h, w = self.preview_frames[0].image.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write frames
        for preview in self.preview_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(preview.image, cv2.COLOR_RGB2BGR)
            
            # Add frame number overlay
            cv2.putText(frame_bgr, f"Frame {preview.frame_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            out.write(frame_bgr)
            
        out.release()
        print(f"✅ Progress video saved to: {output_path}")


class ComfyUIPreviewNode:
    """ComfyUI node for real-time preview"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preview_interval": ("INT", {"default": 5, "min": 1, "max": 50}),
                "show_grid": ("BOOLEAN", {"default": True}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "preview_data")
    FUNCTION = "process_preview"
    CATEGORY = "LatentSync"
    
    def __init__(self):
        self.preview_system = LatentSyncPreviewSystem(
            update_callback=self.send_preview_update
        )
        
    def process_preview(self, images, preview_interval, show_grid):
        """Process images and generate preview"""
        # Start preview system
        self.preview_system.preview_interval = preview_interval
        self.preview_system.start(len(images))
        
        # Process frames
        for i, img in enumerate(images):
            self.preview_system.add_frame(i, img)
            
        # Get preview grid if requested
        preview_output = images
        if show_grid:
            grid = self.preview_system.get_preview_grid()
            if grid is not None:
                # Convert to tensor
                grid_tensor = torch.from_numpy(grid).float() / 255.0
                grid_tensor = grid_tensor.unsqueeze(0)
                preview_output = grid_tensor
                
        # Create preview data JSON
        preview_data = {
            "total_frames": len(images),
            "preview_count": len(self.preview_system.preview_frames),
            "preview_interval": preview_interval
        }
        
        self.preview_system.stop()
        
        return (preview_output, json.dumps(preview_data))
        
    def send_preview_update(self, preview_data):
        """Send preview update to ComfyUI frontend"""
        # This would integrate with ComfyUI's WebSocket system
        # For now, just log it
        print(f"Preview: Frame {preview_data['frame_idx']}/{preview_data['total_frames']} "
              f"({preview_data['progress']*100:.1f}%)")


# Integration helper for pipeline
def add_preview_to_pipeline(pipeline, preview_system: LatentSyncPreviewSystem):
    """Add preview system to LatentSync pipeline"""
    
    # Store reference
    pipeline._preview_system = preview_system
    
    # Wrap the decode_latents method
    original_decode = pipeline.decode_latents
    
    def decode_with_preview(latents):
        # Decode normally
        frames = original_decode(latents)
        
        # Add to preview
        if hasattr(pipeline, '_preview_system') and pipeline._preview_system:
            # Get current chunk info
            chunk_idx = getattr(pipeline, '_current_chunk', 0)
            frames_per_chunk = frames.shape[0]
            
            for i in range(frames.shape[0]):
                frame_idx = chunk_idx * frames_per_chunk + i
                pipeline._preview_system.add_frame(frame_idx, frames[i])
                
        return frames
        
    pipeline.decode_latents = decode_with_preview
    
    return preview_system