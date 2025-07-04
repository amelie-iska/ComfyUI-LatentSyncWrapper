#!/usr/bin/env python
"""
LatentSync GPU Benchmark Tool
Run this script to benchmark your GPU and get optimal settings
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_benchmark import GPUBenchmark, benchmark_gpu

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark your GPU for optimal LatentSync settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_my_gpu.py              # Quick benchmark
  python benchmark_my_gpu.py --full       # Full comprehensive benchmark
  python benchmark_my_gpu.py --clear      # Clear cache and re-benchmark
  
The benchmark will:
  1. Test various speed/quality configurations
  2. Measure performance and memory usage
  3. Save optimal settings for automatic use
  4. Generate a detailed report
        """
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run full benchmark (slower but more accurate)"
    )
    
    parser.add_argument(
        "--clear", 
        action="store_true", 
        help="Clear cached results and re-run benchmark"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick benchmark (default)"
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*60)
    print("🚀 LATENTSYNC GPU BENCHMARK TOOL")
    print("="*60)
    print("\nThis tool will benchmark your GPU to find optimal settings")
    print("for LatentSync 1.6. The results will be cached and used")
    print("automatically when you select 'auto' quality preset.\n")
    
    # Clear cache if requested
    if args.clear:
        import os
        cache_file = os.path.expanduser("~/.latentsync_benchmark_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("✅ Cleared benchmark cache\n")
    
    # Run benchmark
    try:
        quick_mode = not args.full
        
        if args.full:
            print("Running FULL benchmark (this may take 5-10 minutes)...")
        else:
            print("Running QUICK benchmark (this should take 1-2 minutes)...")
            
        print("\nPlease wait while we test your GPU...\n")
        
        # Run the benchmark
        settings = benchmark_gpu(quick_mode=quick_mode)
        
        # Success message
        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETE!")
        print("="*60)
        print("\nYour optimal settings have been saved and will be used")
        print("automatically when you select 'auto' quality preset in")
        print("the LatentSync node.")
        print("\nYou can also use the specific recommendations shown above")
        print("for different use cases (preview, balanced, quality).")
        
        # Save a summary file
        summary_path = os.path.expanduser("~/latentsync_gpu_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"GPU: {GPUBenchmark().gpu_info['name']}\n")
            f.write(f"VRAM: {GPUBenchmark().gpu_info['vram_gb']:.1f} GB\n\n")
            f.write("Recommended Settings:\n")
            f.write(f"Preview: {settings.preview_mode['speed_mode']}/{settings.preview_mode['quality_preset']}\n")
            f.write(f"Balanced: {settings.balanced_mode['speed_mode']}/{settings.balanced_mode['quality_preset']}\n")
            f.write(f"Quality: {settings.quality_mode['speed_mode']}/{settings.quality_mode['quality_preset']}\n")
            
        print(f"\nSummary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Benchmark cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        print("\nPlease ensure:")
        print("  1. You have a CUDA-capable GPU")
        print("  2. PyTorch is installed with CUDA support")
        print("  3. You have sufficient GPU memory available")
        sys.exit(1)

if __name__ == "__main__":
    main()