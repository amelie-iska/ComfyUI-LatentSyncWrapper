"""
Enhanced torch.compile optimizations for maximum speed
"""
import torch
import functools
from packaging import version

# Check PyTorch version
PYTORCH_VERSION = version.parse(torch.__version__.split('+')[0])
COMPILE_AVAILABLE = PYTORCH_VERSION >= version.parse("2.0.0")
INDUCTOR_AVAILABLE = PYTORCH_VERSION >= version.parse("2.1.0")

def get_compile_config(gpu_name: str):
    """Get optimal compile configuration based on GPU"""
    
    # RTX 40-series specific optimizations
    if "4090" in gpu_name or "4080" in gpu_name or "4070" in gpu_name:
        return {
            "mode": "max-autotune-no-cudagraphs",  # Avoid CUDA graph issues
            "dynamic": False,  # Static shapes for better optimization
            "fullgraph": False,  # Allow fallbacks for compatibility
            "backend": "inductor",
            "options": {
                "triton.cudagraphs": False,  # Disable CUDA graphs
                "triton.autotune_pointwise": True,
                "triton.autotune_gemm": True,
                "max_autotune": True,
                "coordinate_descent_tuning": True,
                "epilogue_fusion": True,
                "conv_1x1_as_mm": True,
                "use_mixed_mm": True,  # Use TF32 for matmuls
            }
        }
    
    # RTX 30-series
    elif "3090" in gpu_name or "3080" in gpu_name or "3070" in gpu_name:
        return {
            "mode": "reduce-overhead",
            "dynamic": False,
            "fullgraph": False,
            "backend": "inductor",
            "options": {
                "triton.cudagraphs": False,
                "max_autotune": False,  # Less aggressive for stability
                "use_mixed_mm": True,
            }
        }
    
    # Default configuration
    else:
        return {
            "mode": "default",
            "dynamic": True,
            "fullgraph": False,
            "backend": "inductor",
            "options": {}
        }


def compile_model_components(model, gpu_name: str, verbose: bool = True):
    """Compile individual model components for better optimization"""
    
    if not COMPILE_AVAILABLE:
        if verbose:
            print("torch.compile not available in this PyTorch version")
        return model
    
    config = get_compile_config(gpu_name)
    
    # Compile UNet blocks individually for better optimization
    if hasattr(model, 'unet'):
        if verbose:
            print(f"🔧 Compiling UNet with {config['mode']} mode...")
        
        try:
            # Compile encoder
            if hasattr(model.unet, 'down_blocks'):
                for i, block in enumerate(model.unet.down_blocks):
                    if hasattr(block, 'attentions'):
                        for j, attn in enumerate(block.attentions):
                            model.unet.down_blocks[i].attentions[j] = torch.compile(
                                attn, 
                                mode=config['mode'],
                                dynamic=config['dynamic'],
                                backend=config['backend']
                            )
            
            # Compile decoder
            if hasattr(model.unet, 'up_blocks'):
                for i, block in enumerate(model.unet.up_blocks):
                    if hasattr(block, 'attentions'):
                        for j, attn in enumerate(block.attentions):
                            model.unet.up_blocks[i].attentions[j] = torch.compile(
                                attn,
                                mode=config['mode'],
                                dynamic=config['dynamic'],
                                backend=config['backend']
                            )
            
            # Compile mid block
            if hasattr(model.unet, 'mid_block') and hasattr(model.unet.mid_block, 'attentions'):
                for i, attn in enumerate(model.unet.mid_block.attentions):
                    model.unet.mid_block.attentions[i] = torch.compile(
                        attn,
                        mode=config['mode'],
                        dynamic=config['dynamic'],
                        backend=config['backend']
                    )
            
            if verbose:
                print("✅ UNet compilation complete")
                
        except Exception as e:
            if verbose:
                print(f"⚠️ UNet compilation failed: {e}")
    
    # Compile VAE decoder (most used part)
    if hasattr(model, 'vae') and hasattr(model.vae, 'decoder'):
        if verbose:
            print("🔧 Compiling VAE decoder...")
        
        try:
            model.vae.decoder = torch.compile(
                model.vae.decoder,
                mode=config['mode'],
                dynamic=config['dynamic'],
                backend=config['backend']
            )
            if verbose:
                print("✅ VAE decoder compilation complete")
        except Exception as e:
            if verbose:
                print(f"⚠️ VAE compilation failed: {e}")
    
    return model


def create_compiled_forward(forward_fn, config):
    """Create a compiled version of a forward function with fallback"""
    
    compiled_fn = None
    
    @functools.wraps(forward_fn)
    def wrapped_forward(*args, **kwargs):
        nonlocal compiled_fn
        
        # Try to use compiled version
        if compiled_fn is not None:
            try:
                return compiled_fn(*args, **kwargs)
            except Exception:
                # Fallback to original if compiled fails
                compiled_fn = None
        
        # Compile on first call or after failure
        if compiled_fn is None and COMPILE_AVAILABLE:
            try:
                compiled_fn = torch.compile(
                    forward_fn,
                    mode=config['mode'],
                    dynamic=config['dynamic'],
                    fullgraph=config['fullgraph'],
                    backend=config['backend']
                )
                return compiled_fn(*args, **kwargs)
            except Exception:
                # If compilation fails, use original
                pass
        
        # Fallback to original function
        return forward_fn(*args, **kwargs)
    
    return wrapped_forward


def optimize_pipeline_with_compile(pipeline, gpu_info: dict):
    """Apply torch.compile optimizations to entire pipeline"""
    
    gpu_name = gpu_info.get("name", "")
    config = get_compile_config(gpu_name)
    
    print(f"🚀 Applying torch.compile optimizations for {gpu_name}")
    
    # Set torch backends for maximum performance
    if INDUCTOR_AVAILABLE:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        
        # Enable aggressive fusion
        torch._inductor.config.aggressive_fusion = True
        torch._inductor.config.max_fusion_size = 64
    
    # Compile the main inference method
    if hasattr(pipeline, '__call__'):
        original_call = pipeline.__call__
        
        # Create a wrapper that handles compilation
        def compiled_call(self, *args, **kwargs):
            # For first call, compile the pipeline
            if not hasattr(self, '_compiled_inference'):
                try:
                    print("🔧 Compiling pipeline inference (this may take a minute)...")
                    
                    # Pre-compile with dummy inputs to warm up
                    if hasattr(self, '_create_dummy_inputs'):
                        dummy_inputs = self._create_dummy_inputs()
                        _ = original_call(self, **dummy_inputs)
                    
                    self._compiled_inference = True
                    print("✅ Pipeline compilation complete")
                except Exception as e:
                    print(f"⚠️ Pipeline compilation failed: {e}")
                    self._compiled_inference = False
            
            return original_call(self, *args, **kwargs)
        
        # Bind the method
        pipeline.__call__ = compiled_call.__get__(pipeline, pipeline.__class__)
    
    # Compile individual components
    pipeline = compile_model_components(pipeline, gpu_name)
    
    return pipeline


# Specialized optimizations for specific operations
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def fast_attention(query, key, value, attention_mask=None):
    """Optimized attention computation"""
    # Use Flash Attention if available, otherwise use standard
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False
        )
    else:
        # Fallback implementation
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, value)


@torch.compile(mode="reduce-overhead", dynamic=False)
def fast_conv2d(x, weight, bias=None, stride=1, padding=0):
    """Optimized 2D convolution"""
    return torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def fast_group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    """Optimized group normalization"""
    return torch.nn.functional.group_norm(x, num_groups, weight, bias, eps)