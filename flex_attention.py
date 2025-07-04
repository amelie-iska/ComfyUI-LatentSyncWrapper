"""FlexAttention implementation for LatentSync with lip-sync optimizations"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
import math

# Check if FlexAttention is available (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    print("FlexAttention not available - requires PyTorch 2.5+ with flex_attention support")

class FlexAttentionProcessor:
    """Custom attention processor using FlexAttention for lip-sync optimization"""
    
    def __init__(self, lip_region_weight=2.0, audio_visual_weight=1.5):
        self.lip_region_weight = lip_region_weight
        self.audio_visual_weight = audio_visual_weight
        
        if not FLEX_ATTENTION_AVAILABLE:
            print("Warning: FlexAttention not available. Falling back to standard attention.")
    
    def create_lip_attention_bias(self, batch_size, seq_len, height, width):
        """Create attention bias that emphasizes lip regions"""
        # Define lip region (lower third of face)
        lip_start_y = int(height * 0.6)
        lip_end_y = height
        lip_start_x = int(width * 0.3)
        lip_end_x = int(width * 0.7)
        
        # Create 2D attention mask
        mask = torch.ones(height, width)
        mask[lip_start_y:lip_end_y, lip_start_x:lip_end_x] = self.lip_region_weight
        
        # Flatten and expand for attention
        mask_flat = mask.view(-1)  # (H*W,)
        attention_bias = mask_flat.unsqueeze(0) * mask_flat.unsqueeze(1)  # (H*W, H*W)
        
        # Expand for batch and heads
        attention_bias = attention_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, H*W, H*W)
        attention_bias = attention_bias.expand(batch_size, -1, -1, -1)
        
        return attention_bias
    
    def audio_visual_score_mod(self, score, batch_idx, head_idx, q_idx, kv_idx):
        """Modify attention scores based on audio-visual alignment"""
        # Enhance cross-attention between audio and visual features
        # This is a simplified version - in practice you'd analyze the actual features
        if q_idx < kv_idx:  # Cross-attention pattern
            return score * self.audio_visual_weight
        return score
    
    def __call__(self, query, key, value, attention_mask=None, head_dim=None):
        """Apply FlexAttention with lip-sync optimizations"""
        
        if not FLEX_ATTENTION_AVAILABLE or flex_attention is None:
            # Fallback to standard attention
            scale = 1.0 / math.sqrt(query.size(-1))
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)
            return torch.matmul(probs, value)
        
        # Use FlexAttention with custom score modifier
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Estimate spatial dimensions (assuming square for simplicity)
        spatial_size = int(math.sqrt(seq_len))
        
        # Create lip region attention bias
        lip_bias = self.create_lip_attention_bias(batch_size, seq_len, spatial_size, spatial_size)
        
        # Apply FlexAttention with custom scoring
        def combined_score_mod(score, b, h, q, kv):
            # Apply lip region bias
            biased_score = score + lip_bias[b, h, q, kv] * 0.1  # Scale down bias
            # Apply audio-visual enhancement
            return self.audio_visual_score_mod(biased_score, b, h, q, kv)
        
        # Use FlexAttention
        output = flex_attention(
            query, key, value,
            score_mod=combined_score_mod,
            enable_gqa=True  # Enable grouped query attention for efficiency
        )
        
        return output


class OptimizedFlexAttentionProcessor:
    """Optimized version with caching and better performance"""
    
    def __init__(self, lip_region_weight=1.8, cache_size=16):
        self.lip_region_weight = lip_region_weight
        self.cache_size = cache_size
        self.bias_cache = {}
        
    def get_cached_bias(self, key, create_fn):
        """Cache attention biases for reuse"""
        if key not in self.bias_cache:
            if len(self.bias_cache) >= self.cache_size:
                # Remove oldest entry
                self.bias_cache.pop(next(iter(self.bias_cache)))
            self.bias_cache[key] = create_fn()
        return self.bias_cache[key]
    
    def create_sparse_lip_pattern(self, seq_len):
        """Create sparse attention pattern focusing on lip regions"""
        # Define block sparse pattern for efficiency
        block_size = 16
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Create block mask - attend to lip region blocks more
        spatial_size = int(math.sqrt(seq_len))
        lip_blocks = []
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min((i + 1) * block_size, seq_len)
            
            # Check if block contains lip region
            for idx in range(block_start, block_end):
                y = (idx // spatial_size) / spatial_size
                x = (idx % spatial_size) / spatial_size
                if 0.6 <= y <= 1.0 and 0.3 <= x <= 0.7:
                    lip_blocks.append(i)
                    break
        
        return lip_blocks
    
    def __call__(self, query, key, value, attention_mask=None, head_dim=None):
        """Optimized FlexAttention call"""
        
        if not FLEX_ATTENTION_AVAILABLE:
            # Efficient fallback
            scale = 1.0 / math.sqrt(query.size(-1))
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            # Apply lip region emphasis in fallback
            seq_len = query.size(-2)
            spatial_size = int(math.sqrt(seq_len))
            
            # Simple lip region mask
            lip_mask = torch.ones_like(scores)
            lip_start = int(seq_len * 0.6)
            lip_mask[..., lip_start:, lip_start:] *= self.lip_region_weight
            scores = scores * lip_mask
            
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)
            return torch.matmul(probs, value)
        
        # Use FlexAttention with optimizations
        batch_size, num_heads, seq_len = query.shape[:3]
        cache_key = (seq_len, self.lip_region_weight)
        
        # Get cached sparse pattern
        lip_blocks = self.get_cached_bias(
            cache_key,
            lambda: self.create_sparse_lip_pattern(seq_len)
        )
        
        # Simple score modifier focusing on lip sync
        def lip_sync_score_mod(score, b, h, q, kv):
            # Boost attention to lip region blocks
            q_block = q // 16
            kv_block = kv // 16
            if q_block in lip_blocks or kv_block in lip_blocks:
                return score * self.lip_region_weight
            return score
        
        # Apply FlexAttention
        output = flex_attention(
            query, key, value,
            score_mod=lip_sync_score_mod,
            enable_gqa=True,
            scale=1.0 / math.sqrt(query.size(-1))
        )
        
        return output


def create_attention_processor(mode="flex", **kwargs):
    """Factory function to create appropriate attention processor"""
    
    if mode == "flex":
        if FLEX_ATTENTION_AVAILABLE:
            print("Using FlexAttention with lip-sync optimizations")
            return OptimizedFlexAttentionProcessor(**kwargs)
        else:
            print("FlexAttention not available, falling back to standard attention")
            return None
    
    elif mode == "flash" or mode == "xformers":
        # Return None to use the built-in flash/xformers attention
        return None
    
    else:  # standard
        return None


# Utility function to check FlexAttention availability
def check_flex_attention():
    """Check if FlexAttention is available and working"""
    if not FLEX_ATTENTION_AVAILABLE:
        return False, "FlexAttention not found (requires PyTorch 2.5+)"
    
    try:
        # Test FlexAttention
        test_q = torch.randn(1, 8, 64, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
        test_k = test_v = test_q
        result = flex_attention(test_q, test_k, test_v)
        return True, "FlexAttention is available and working"
    except Exception as e:
        return False, f"FlexAttention test failed: {str(e)}"