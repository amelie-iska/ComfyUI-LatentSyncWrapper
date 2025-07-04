"""
Ultimate progress bar fix - nuclear option
This completely disables ALL progress output from diffusers and related libraries
"""

import sys
import os
import builtins
import functools

# Store original functions
_original_print = builtins.print
_original_import = builtins.__import__

# Track what we've patched
_patched_modules = set()

def silent_tqdm(*args, **kwargs):
    """Completely silent tqdm replacement"""
    class SilentBar:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.n = 0
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
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
            pass
            
        def __getattr__(self, name):
            # Return self for any attribute access to support chaining
            return lambda *args, **kwargs: self
    
    return SilentBar(*args, **kwargs)


def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Import hook that patches modules as they're loaded"""
    
    # Import the module normally first
    module = _original_import(name, globals, locals, fromlist, level)
    
    # Patch tqdm modules
    if 'tqdm' in name and name not in _patched_modules:
        _patched_modules.add(name)
        
        # Replace all tqdm-like attributes
        for attr in ['tqdm', 'trange', 'tqdm_notebook', 'tnrange']:
            if hasattr(module, attr):
                setattr(module, attr, silent_tqdm)
        
        # Also patch any class named tqdm
        if hasattr(module, '__dict__'):
            for key, value in list(module.__dict__.items()):
                if 'tqdm' in key.lower():
                    module.__dict__[key] = silent_tqdm
    
    # Patch diffusers modules
    if 'diffusers' in name and name not in _patched_modules:
        _patched_modules.add(name)
        
        # If it's a pipeline module, patch progress bar methods
        if hasattr(module, 'DiffusionPipeline'):
            patch_diffusion_pipeline(module.DiffusionPipeline)
        
        # Look for any pipeline classes and patch them
        if hasattr(module, '__dict__'):
            for key, value in module.__dict__.items():
                if hasattr(value, '__mro__') and any('Pipeline' in str(base) for base in value.__mro__):
                    patch_pipeline_class(value)
    
    return module


def patch_diffusion_pipeline(pipeline_class):
    """Patch DiffusionPipeline base class"""
    
    # Create a dummy progress bar property
    @property
    def dummy_progress_bar(self):
        from contextlib import contextmanager
        
        @contextmanager
        def no_op_context(iterable=None, total=None, desc=None):
            if iterable is not None:
                yield iterable
            else:
                yield range(total) if total else []
        
        return no_op_context
    
    # Override methods
    pipeline_class.progress_bar = dummy_progress_bar
    pipeline_class.set_progress_bar_config = lambda self, **kwargs: None
    
    # Also patch _progress_bar if it exists
    if hasattr(pipeline_class, '_progress_bar'):
        pipeline_class._progress_bar = property(lambda self: None)


def patch_pipeline_class(pipeline_class):
    """Patch any pipeline class"""
    
    # Check if it has progress-related methods
    if hasattr(pipeline_class, 'progress_bar'):
        @property
        def silent_progress(self):
            from contextlib import contextmanager
            
            @contextmanager
            def dummy(iterable=None, total=None, desc=None):
                yield iterable if iterable is not None else range(total or 0)
            
            return dummy
        
        pipeline_class.progress_bar = silent_progress
    
    if hasattr(pipeline_class, 'set_progress_bar_config'):
        pipeline_class.set_progress_bar_config = lambda self, **kwargs: None


def filtered_print(*args, **kwargs):
    """Print that filters progress-like output"""
    
    # Convert args to string
    output = ' '.join(str(arg) for arg in args)
    
    # List of patterns to block
    blocked_patterns = [
        'Sample frames:',
        '100%|',
        'it/s]',
        '█' * 3,  # Progress bar characters
        '━' * 3,  # Alternative progress bar
        '=' * 3,  # Another progress bar style
    ]
    
    # Check if output contains blocked patterns
    if any(pattern in output for pattern in blocked_patterns):
        # Check if it's a progress update (has percentage)
        if '%' in output and ('/' in output or '|' in output):
            return  # Block this output
    
    # Allow other output
    _original_print(*args, **kwargs)


def apply_ultimate_fix():
    """Apply the ultimate progress bar fix"""
    
    print("🔨 Applying ULTIMATE progress bar fix...")
    
    # 1. Replace builtins.__import__ to patch modules as they load
    builtins.__import__ = patched_import
    
    # 2. Replace print to filter output
    builtins.print = filtered_print
    
    # 3. Pre-patch tqdm if it exists
    for tqdm_module in ['tqdm', 'tqdm.auto', 'tqdm.std', 'tqdm.notebook']:
        if tqdm_module in sys.modules:
            module = sys.modules[tqdm_module]
            for attr in ['tqdm', 'trange', 'tqdm_notebook']:
                if hasattr(module, attr):
                    setattr(module, attr, silent_tqdm)
    
    # 4. Create fake tqdm modules to prevent real ones from loading
    import types
    for module_name in ['tqdm', 'tqdm.auto', 'tqdm.std', 'tqdm.notebook']:
        if module_name not in sys.modules:
            fake_module = types.ModuleType(module_name)
            fake_module.tqdm = silent_tqdm
            fake_module.trange = lambda *args, **kwargs: silent_tqdm(range(*args), **kwargs)
            sys.modules[module_name] = fake_module
    
    # 5. Set environment variables to disable progress bars
    os.environ['TQDM_DISABLE'] = '1'
    os.environ['DIFFUSERS_DISABLE_PROGRESS_BAR'] = '1'
    
    print("✅ ULTIMATE progress bar fix applied!")
    print("   - All tqdm imports will be silent")
    print("   - All progress output will be filtered")
    print("   - Diffusers progress bars disabled")


# Apply the fix immediately
apply_ultimate_fix()