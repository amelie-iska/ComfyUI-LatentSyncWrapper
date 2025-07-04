"""
Enhanced progress reporting for LatentSync with ComfyUI integration
"""

import time
from typing import Optional, Callable
import threading
import queue

class EnhancedProgressReporter:
    """Better progress reporting that works with ComfyUI's system"""
    
    def __init__(self, comfy_progress_callback: Optional[Callable] = None):
        self.comfy_callback = comfy_progress_callback
        self.start_time = None
        self.total_frames = 0
        self.processed_frames = 0
        self.current_chunk = 0
        self.total_chunks = 0
        self.current_step = 0
        self.total_steps = 0
        self.memory_usage = 0
        self.last_update_time = 0
        self.update_interval = 0.1  # Update every 100ms
        
    def start(self, total_frames: int, total_chunks: int, steps_per_frame: int):
        """Initialize progress tracking"""
        self.start_time = time.time()
        self.total_frames = total_frames
        self.total_chunks = total_chunks
        self.total_steps = steps_per_frame
        self.processed_frames = 0
        self.current_chunk = 0
        self.current_step = 0
        
    def update_chunk(self, chunk_idx: int):
        """Update current chunk being processed"""
        self.current_chunk = chunk_idx
        self.update_display()
        
    def update_step(self, step: int, force: bool = False):
        """Update denoising step progress"""
        self.current_step = step
        current_time = time.time()
        
        # Rate limit updates unless forced
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return
            
        self.last_update_time = current_time
        self.update_display()
        
    def update_memory(self, memory_gb: float):
        """Update current memory usage"""
        self.memory_usage = memory_gb
        
    def complete_frames(self, num_frames: int):
        """Mark frames as completed"""
        self.processed_frames += num_frames
        self.update_display()
        
    def update_display(self):
        """Update the progress display"""
        if self.total_frames == 0:
            return
            
        # Calculate overall progress
        chunk_progress = (self.current_chunk / self.total_chunks) if self.total_chunks > 0 else 0
        step_progress = (self.current_step / self.total_steps) if self.total_steps > 0 else 0
        chunk_weight = 1.0 / self.total_chunks if self.total_chunks > 0 else 0
        
        # Combined progress calculation
        overall_progress = chunk_progress + (step_progress * chunk_weight)
        
        # Time estimation
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        if overall_progress > 0:
            total_time = elapsed_time / overall_progress
            remaining_time = total_time - elapsed_time
        else:
            remaining_time = 0
            
        # Format progress message
        progress_msg = self._format_progress(
            overall_progress,
            self.current_chunk + 1,
            self.total_chunks,
            self.current_step,
            self.total_steps,
            self.processed_frames,
            self.total_frames,
            elapsed_time,
            remaining_time,
            self.memory_usage
        )
        
        # Send to ComfyUI if callback available
        if self.comfy_callback:
            self.comfy_callback(overall_progress, progress_msg)
        else:
            # Fallback to print
            print(f"\r{progress_msg}", end="", flush=True)
            
    def _format_progress(self, overall, chunk, total_chunks, step, total_steps, 
                        frames, total_frames, elapsed, remaining, memory):
        """Format a nice progress message"""
        # Progress bar
        bar_length = 40
        filled = int(bar_length * overall)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Time formatting
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)
        
        # Memory formatting
        memory_str = f"{memory:.1f}GB" if memory > 0 else "N/A"
        
        # Build message
        msg = (
            f"[{bar}] {overall*100:.1f}% | "
            f"Chunk {chunk}/{total_chunks} | "
            f"Step {step}/{total_steps} | "
            f"Frames {frames}/{total_frames} | "
            f"Time: {elapsed_str} / ETA: {remaining_str} | "
            f"VRAM: {memory_str}"
        )
        
        return msg
        
    def _format_time(self, seconds):
        """Format seconds into readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
            
    def finish(self):
        """Mark processing as complete"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n✅ Processing complete! Total time: {self._format_time(total_time)}")
            print(f"   Processed {self.total_frames} frames ({self.total_frames/total_time:.1f} fps)")


class ComfyUIProgressIntegration:
    """Integration layer for ComfyUI's progress system"""
    
    def __init__(self, comfy_node):
        self.node = comfy_node
        self.progress_reporter = EnhancedProgressReporter(self.update_comfy)
        self.message_queue = queue.Queue()
        
    def update_comfy(self, progress: float, message: str):
        """Update ComfyUI's progress display"""
        try:
            # ComfyUI expects progress between 0 and 1
            if hasattr(self.node, 'progress_update'):
                self.node.progress_update(progress, message)
            
            # Also try updating through the server if available
            if hasattr(self.node, 'server'):
                self.node.server.send_sync("progress", {
                    "node": self.node.node_id,
                    "value": progress,
                    "max": 1.0,
                    "text": message
                })
        except Exception as e:
            # Fallback to console
            print(f"\r{message}", end="", flush=True)
            
    def get_reporter(self) -> EnhancedProgressReporter:
        """Get the progress reporter instance"""
        return self.progress_reporter


# Monkey-patch integration for the pipeline
def integrate_enhanced_progress(pipeline_instance, comfy_node=None):
    """Integrate enhanced progress into existing pipeline"""
    
    if comfy_node:
        integration = ComfyUIProgressIntegration(comfy_node)
        progress = integration.get_reporter()
    else:
        progress = EnhancedProgressReporter()
    
    # Store original progress_bar method
    original_progress_bar = pipeline_instance.progress_bar
    
    class ProgressBarWrapper:
        def __init__(self, total):
            self.total = total
            self.current = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self):
            self.current += 1
            progress.update_step(self.current, force=(self.current == 1 or self.current == self.total))
    
    # Replace progress_bar method
    pipeline_instance.progress_bar = lambda total: ProgressBarWrapper(total)
    
    # Store progress reporter on pipeline
    pipeline_instance._enhanced_progress = progress
    
    return progress