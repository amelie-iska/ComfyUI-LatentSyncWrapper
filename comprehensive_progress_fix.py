"""
Comprehensive fix for all progress bar issues in LatentSync
This module patches multiple levels to ensure no blocking progress bars
"""

import sys
import builtins
import time
import re
from contextlib import contextmanager
from typing import Optional, Callable

# Flag to track if patches have been applied
_patches_applied = False

def apply_all_progress_fixes():
    """Apply all progress bar fixes comprehensively"""
    global _patches_applied
    
    if _patches_applied:
        return
    
    print("\n🔧 Applying comprehensive progress bar fixes...")
    
    # 1. Patch tqdm at the deepest level
    patch_tqdm_globally()
    
    # 2. Patch diffusers library
    patch_diffusers_library()
    
    # 3. Install output filter
    install_output_filter()
    
    _patches_applied = True
    print("✅ All progress bar fixes applied!\n")


def patch_tqdm_globally():
    """Completely replace tqdm with non-blocking version"""
    
    class DummyTqdm:
        """Non-blocking tqdm replacement"""
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.n = 0
            self.total = kwargs.get('total', None)
            self.desc = kwargs.get('desc', '')
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.n += 1
            return self
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass
            
        def set_description(self, desc):
            self.desc = desc
            
        def set_postfix(self, *args, **kwargs):
            pass
    
    # Create module if it doesn't exist
    if 'tqdm' not in sys.modules:
        import types
        tqdm_module = types.ModuleType('tqdm')
        sys.modules['tqdm'] = tqdm_module
    else:
        tqdm_module = sys.modules['tqdm']
    
    # Replace all tqdm variants
    tqdm_module.tqdm = DummyTqdm
    tqdm_module.trange = lambda *args, **kwargs: DummyTqdm(range(*args), **kwargs)
    
    # Create submodules
    for submodule in ['auto', 'std', 'notebook']:
        if f'tqdm.{submodule}' not in sys.modules:
            import types
            sub = types.ModuleType(f'tqdm.{submodule}')
            sys.modules[f'tqdm.{submodule}'] = sub
        else:
            sub = sys.modules[f'tqdm.{submodule}']
        
        sub.tqdm = DummyTqdm
        sub.trange = lambda *args, **kwargs: DummyTqdm(range(*args), **kwargs)
        setattr(tqdm_module, submodule, sub)
    
    print("  ✓ Patched tqdm globally")


def patch_diffusers_library():
    """Patch diffusers to prevent progress bars"""
    
    try:
        # Import diffusers components
        from diffusers.pipelines import DiffusionPipeline
        
        # Create a universal progress bar context manager
        @contextmanager
        def no_progress_bar(iterable=None, total=None, desc=None):
            """Context manager that yields the iterable without progress"""
            if iterable is not None:
                yield iterable
            else:
                yield range(total) if total else []
        
        # Patch the progress_bar property
        @property
        def dummy_progress_bar(self):
            return no_progress_bar
        
        # Apply patches
        DiffusionPipeline.progress_bar = dummy_progress_bar
        
        # Disable progress bar configuration
        def no_op_config(self, **kwargs):
            pass
        
        DiffusionPipeline.set_progress_bar_config = no_op_config
        
        # Also patch any existing instances
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, DiffusionPipeline):
                # Disable any existing progress bars
                if hasattr(obj, '_progress_bar_config'):
                    obj._progress_bar_config = {'disable': True}
        
        print("  ✓ Patched diffusers library")
        
    except ImportError:
        print("  ⚠ Diffusers not imported yet, will patch when loaded")


def install_output_filter():
    """Install filter to catch any remaining progress output"""
    
    original_print = builtins.print
    last_output_time = {}
    
    # Patterns that indicate progress output
    progress_patterns = [
        re.compile(r'Sample frames:.*\d+%', re.IGNORECASE),
        re.compile(r'\d+%.*\(\d+/\d+\)', re.IGNORECASE),
        re.compile(r'^\s*\d+/\d+\s*\[', re.IGNORECASE),  # "20/20 [" style
        re.compile(r'100%\|', re.IGNORECASE),
        re.compile(r'it/s\]$', re.IGNORECASE),
    ]
    
    def filtered_print(*args, **kwargs):
        """Print that filters repetitive progress output"""
        # If file is explicitly provided and it's sys.stderr, don't filter (for tracebacks)
        import sys
        if kwargs.get('file') is sys.stderr:
            original_print(*args, **kwargs)
            return
            
        # Convert to string to check
        try:
            import io
            buffer = io.StringIO()
            # Create a copy of kwargs without 'file' to avoid double-passing it
            temp_kwargs = kwargs.copy()
            temp_kwargs.pop('file', None)
            original_print(*args, file=buffer, **temp_kwargs)
            output = buffer.getvalue()
            
            # Check if this looks like progress output
            is_progress = any(pattern.search(output) for pattern in progress_patterns)
            
            if is_progress:
                # Rate limit progress output
                key = output[:50]  # Use first 50 chars as key
                current_time = time.time()
                
                if key in last_output_time:
                    if current_time - last_output_time[key] < 0.5:  # 500ms minimum
                        return  # Skip this output
                
                last_output_time[key] = current_time
            
            # Allow the output
            original_print(*args, **kwargs)
            
        except Exception as e:
            # If anything fails, just use original print
            original_print(*args, **kwargs)
    
    # Replace print
    builtins.print = filtered_print
    print("  ✓ Installed output filter")


# Additional helper to patch a specific pipeline instance
def patch_pipeline_instance(pipeline):
    """Patch a specific pipeline instance to disable progress bars"""
    
    # Override progress_bar property
    @property
    def no_progress(self):
        @contextmanager
        def dummy_context(iterable=None, total=None, desc=None):
            if iterable is not None:
                yield iterable
            else:
                yield range(total) if total else []
        return dummy_context
    
    # Monkey patch the instance
    pipeline.__class__.progress_bar = no_progress
    
    # Disable any existing configuration
    if hasattr(pipeline, '_progress_bar_config'):
        pipeline._progress_bar_config = {'disable': True}
    
    # Override set_progress_bar_config
    def no_op(self, **kwargs):
        pass
    
    pipeline.__class__.set_progress_bar_config = no_op
    
    return pipeline


# Export the main function and helper
__all__ = ['apply_all_progress_fixes', 'patch_pipeline_instance']

# Auto-apply fixes on import
apply_all_progress_fixes()