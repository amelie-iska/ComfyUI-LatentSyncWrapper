"""
Simple tqdm patch to prevent blocking UI updates
"""

import sys
import time
from contextlib import contextmanager

class FakeProgressBar:
    """Non-blocking progress bar replacement for tqdm"""
    
    def __init__(self, iterable=None, total=None, desc=None, **kwargs):
        self.iterable = iterable
        self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
        self.current = 0
        self.desc = desc or ""
        self.start_time = time.time()
        self.last_print = 0
        
    def __iter__(self):
        """Make it iterable"""
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.update(1)
        else:
            # If no iterable, just return self for manual updates
            return self
            
    def __next__(self):
        """For manual iteration"""
        raise StopIteration
        
    def update(self, n=1):
        """Update progress"""
        self.current += n
        
        # Only print updates every 0.5 seconds to avoid console spam
        current_time = time.time()
        if current_time - self.last_print > 0.5:
            if self.total:
                progress = min(100, (self.current / self.total) * 100)
                print(f"\r{self.desc}: {progress:.1f}% ({self.current}/{self.total})", end="", flush=True)
            else:
                print(f"\r{self.desc}: {self.current} items", end="", flush=True)
            self.last_print = current_time
            
    def set_description(self, desc):
        """Update description"""
        self.desc = desc
        
    def close(self):
        """Close progress bar"""
        if self.total:
            print(f"\r{self.desc}: 100% ({self.total}/{self.total})")
        else:
            print(f"\r{self.desc}: {self.current} items")
            
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()


def tqdm_replacement(*args, **kwargs):
    """Smart replacement for tqdm that handles both iterator and context manager usage"""
    # If first argument is an iterable, return an iterator
    if args and hasattr(args[0], '__iter__'):
        return FakeProgressBar(args[0], *args[1:], **kwargs)
    else:
        # Otherwise return a progress bar for manual updates
        return FakeProgressBar(None, *args, **kwargs)


@contextmanager
def tqdm_context(*args, **kwargs):
    """Context manager replacement for tqdm.tqdm"""
    pbar = FakeProgressBar(*args, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()


_already_patched = False

def patch_tqdm():
    """Monkey patch tqdm to use non-blocking progress"""
    global _already_patched
    
    # Skip if already patched
    if _already_patched:
        return
    
    try:
        # Import the real tqdm module
        import tqdm
        import tqdm.auto
        import tqdm.std
        
        # Save original in case we need it
        original_tqdm = getattr(tqdm, '_original_tqdm', None)
        if original_tqdm is None:
            tqdm._original_tqdm = tqdm.tqdm
            tqdm.auto._original_tqdm = tqdm.auto.tqdm
            tqdm.std._original_tqdm = tqdm.std.tqdm
        
        # Replace all tqdm references
        tqdm.tqdm = tqdm_replacement
        tqdm.auto.tqdm = tqdm_replacement
        tqdm.std.tqdm = tqdm_replacement
        
        # Also patch the tqdm class itself
        setattr(tqdm, 'tqdm', tqdm_replacement)
        setattr(tqdm.auto, 'tqdm', tqdm_replacement)
        setattr(tqdm.std, 'tqdm', tqdm_replacement)
        
        # Patch trange as well
        def trange_replacement(*args, **kwargs):
            return tqdm_replacement(range(*args), **kwargs)
        
        if hasattr(tqdm, 'trange'):
            tqdm.trange = trange_replacement
        if hasattr(tqdm.auto, 'trange'):
            tqdm.auto.trange = trange_replacement
        if hasattr(tqdm.std, 'trange'):
            tqdm.std.trange = trange_replacement
            
        # Also patch notebook version if it exists
        try:
            import tqdm.notebook
            tqdm.notebook.tqdm = tqdm_replacement
            tqdm.notebook.trange = trange_replacement
        except:
            pass
            
        print("✅ Patched tqdm to use non-blocking progress")
        _already_patched = True
        
    except Exception as e:
        print(f"Warning: Could not patch tqdm: {e}")
        _already_patched = True


def patch_diffusers_progress():
    """Patch diffusers library to disable all progress bars"""
    try:
        # Import diffusers modules
        import diffusers
        from diffusers.pipelines import pipeline_utils
        
        # Override the progress_bar function in pipeline_utils
        def dummy_progress_bar(self, *args, **kwargs):
            """Dummy progress bar that does nothing"""
            from contextlib import contextmanager
            
            @contextmanager
            def dummy_context(iterable=None, total=None, desc=None):
                if iterable is not None:
                    yield iterable
                else:
                    yield range(total) if total else []
            
            return dummy_context()
        
        # Patch the base DiffusionPipeline class
        if hasattr(diffusers.pipelines, 'DiffusionPipeline'):
            diffusers.pipelines.DiffusionPipeline.progress_bar = dummy_progress_bar
            diffusers.pipelines.DiffusionPipeline.set_progress_bar_config = lambda self, **kwargs: None
            
        # Also patch pipeline_utils directly
        if hasattr(pipeline_utils, 'DiffusionPipeline'):
            pipeline_utils.DiffusionPipeline.progress_bar = dummy_progress_bar
            pipeline_utils.DiffusionPipeline.set_progress_bar_config = lambda self, **kwargs: None
            
        # Patch logging.tqdm if it exists
        if hasattr(diffusers.utils, 'logging'):
            diffusers.utils.logging.tqdm = tqdm_replacement
            
        print("✅ Patched diffusers to disable progress bars")
        
    except ImportError:
        pass  # Diffusers not installed
    except Exception as e:
        print(f"Warning: Could not patch diffusers: {e}")



# Progress output interceptor
import builtins
import re
import time
from io import StringIO

class ProgressOutputFilter:
    """Filters repeated progress output"""
    
    def __init__(self):
        self.original_print = builtins.print
        self.last_output = {}
        self.min_interval = 0.5
        self.progress_patterns = [
            r'Sample frames:.*\d+%',
            r'\d+%.*\(\d+/\d+\)',
            r'Processing.*\d+/\d+',
            r'\|\s*\d+%\s*\|',
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.progress_patterns]
    
    def is_progress(self, text):
        text_str = str(text)
        for pattern in self.compiled_patterns:
            if pattern.search(text_str):
                return True
        return False
    
    def patched_print(self, *args, **kwargs):
        # Check if this is progress output
        output = StringIO()
        # Extract file parameter if present
        file_param = kwargs.get('file', None)
        temp_kwargs = kwargs.copy()
        temp_kwargs.pop('file', None)
        
        try:
            self.original_print(*args, file=output, **temp_kwargs)
            text = output.getvalue()
            
            if self.is_progress(text):
                # Rate limit progress output
                key = text[:30]
                current_time = time.time()
                if key in self.last_output:
                    if current_time - self.last_output[key] < self.min_interval:
                        return  # Skip this output
                self.last_output[key] = current_time
            
            # Allow the output - restore original file parameter
            if file_param is not None:
                kwargs['file'] = file_param
            self.original_print(*args, **kwargs)
        except Exception:
            # Fallback to original print if anything goes wrong
            self.original_print(*args, **kwargs)
    
    def install(self):
        builtins.print = self.patched_print
        print("✅ Installed progress output filter")

# Create and install filter
progress_filter = ProgressOutputFilter()
progress_filter.install()

# Call both patches
patch_tqdm()
patch_diffusers_progress()