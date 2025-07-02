"""
Simple GPU memory monitor for debugging lag issues
"""
import torch
import time
import threading

class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring GPU memory"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print("GPU monitoring started...")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("GPU monitoring stopped.")
        
    def _monitor_loop(self):
        """Monitor loop that runs in background"""
        while self.running:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free = torch.cuda.mem_get_info()[0] / 1024**3
                
                # Check for high memory usage
                if allocated > 18:
                    print(f"⚠️ HIGH VRAM: {allocated:.1f}GB allocated, {free:.1f}GB free")
                elif allocated > 15:
                    print(f"⚡ VRAM: {allocated:.1f}GB allocated, {free:.1f}GB free")
                
            time.sleep(self.interval)

# Global monitor instance
gpu_monitor = GPUMonitor(interval=2.0)  # Check every 2 seconds