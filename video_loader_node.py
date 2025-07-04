import os
import torch
import numpy as np
from PIL import Image
import gc
import ffmpeg

class EfficientVideoLoader:
    """Load video files efficiently in batches to avoid memory crashes"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 50, "min": 10, "max": 200, "step": 10}),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "end_frame": ("INT", {"default": -1, "min": -1}),
            }
        }
    
    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_video"
    
    def load_video(self, video_path, batch_size=50, start_frame=0, end_frame=-1):
        """Load video frames efficiently in batches"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_frames = int(video_info['nb_frames'])
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Determine frame range
        if end_frame == -1 or end_frame > total_frames:
            end_frame = total_frames
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}")
        
        frames_to_load = end_frame - start_frame
        print(f"Loading {frames_to_load} frames from {video_path}")
        
        # Try to use decord for efficient video reading if available
        try:
            from decord import VideoReader
            from decord import cpu
            
            vr = VideoReader(video_path, ctx=cpu(0))
            processed_frames_list = []
            
            for batch_start in range(start_frame, end_frame, batch_size):
                batch_end = min(batch_start + batch_size, end_frame)
                batch_frames = vr.get_batch(list(range(batch_start, batch_end)))
                batch_frames = torch.from_numpy(batch_frames.asnumpy()).float() / 255.0
                processed_frames_list.append(batch_frames)
                
                # Clear memory periodically
                if batch_start + batch_size < end_frame:
                    gc.collect()
                
                print(f"Loaded frames {batch_start}-{batch_end} of {end_frame}")
            
            processed_frames = torch.cat(processed_frames_list, dim=0)
            del processed_frames_list
            
        except ImportError:
            print("Decord not available, using ffmpeg extraction (slower)...")
            
            # Create a temporary directory for frames
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract frames using ffmpeg
                output_pattern = os.path.join(temp_dir, "frame_%06d.png")
                
                # Build ffmpeg command to extract specific frame range
                stream = ffmpeg.input(video_path)
                if start_frame > 0:
                    stream = stream.filter('select', f'gte(n,{start_frame})')
                if end_frame < total_frames:
                    stream = stream.filter('select', f'lt(n,{end_frame})')
                
                stream = stream.output(output_pattern, start_number=start_frame)
                ffmpeg.run(stream, quiet=True, overwrite_output=True)
                
                # Load frames in batches
                frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
                processed_frames_list = []
                
                for i in range(0, len(frame_files), batch_size):
                    batch_files = frame_files[i:i+batch_size]
                    batch_frames = []
                    
                    for frame_file in batch_files:
                        frame_path = os.path.join(temp_dir, frame_file)
                        frame = Image.open(frame_path).convert('RGB')
                        frame_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
                        batch_frames.append(frame_tensor)
                    
                    batch_tensor = torch.stack(batch_frames)
                    processed_frames_list.append(batch_tensor)
                    
                    # Clear memory
                    del batch_frames
                    gc.collect()
                    
                    print(f"Loaded batch {i//batch_size + 1} of {(len(frame_files) + batch_size - 1)//batch_size}")
                
                processed_frames = torch.cat(processed_frames_list, dim=0)
                del processed_frames_list
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Successfully loaded {processed_frames.shape[0]} frames")
        return (processed_frames,)


# Node Mappings for this module
VIDEO_LOADER_NODE_CLASS_MAPPINGS = {
    "EfficientVideoLoader": EfficientVideoLoader,
}

VIDEO_LOADER_NODE_DISPLAY_NAME_MAPPINGS = {
    "EfficientVideoLoader": "📁 Efficient Video Loader (MEMSAFE)",
}