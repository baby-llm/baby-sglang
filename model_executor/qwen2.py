"""
Minimal Qwen2 model implementation for baby-sglang.

Follows SGLang's design philosophy but simplified for MVP:
- Single GPU, static batching
- No tensor parallelism, quantization, or advanced optimizations  
- PyTorch-only implementation with SGLang-compatible interfaces
- Token-level KV caching using SGLang's memory pool design

Based on sglang/srt/models/qwen2.py but simplified for Phase 3 requirements.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedForwardBatch:
    """
    Simplified ForwardBatch for MVP, containing only essential fields.
    Based on SGLang's ForwardBatch but stripped down for single-GPU static batching.
    """
    # Basic batch info
    batch_size: int
    input_ids: torch.Tensor  # [total_tokens] 
    req_pool_indices: torch.Tensor  # [batch_size] - request indices in req_to_token_pool
    seq_lens: torch.Tensor  # [batch_size] - sequence lengths
    out_cache_loc: torch.Tensor  # [total_tokens] - token positions in KV cache
    
    # Position info
    positions: torch.Tensor  # [total_tokens] - absolute positions
    
    # Memory pools (SGLang-compatible)
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool
    
    # Forward mode flags
    is_prefill: bool = True
    
    @classmethod
    def create_prefill_batch(
        cls,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor, 
        out_cache_loc: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
    ):
        """
        Factory method for creating prefill batch with proper position calculation.
        
        Phase 4: Handles absolute position computation for RoPE.
        """
        batch_size = len(req_pool_indices)
        total_tokens = input_ids.shape[0]
        
        # Compute absolute positions for each token
        positions = []
        for seq_len in seq_lens:
            positions.extend(range(seq_len.item()))
        positions = torch.tensor(positions, device=input_ids.device, dtype=torch.long)
        
        return cls( # cls refers to the class itself (SimplifiedForwardBatch), creating a new instance
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            is_prefill=True,
        )
    
    @classmethod
    def create_decode_batch(
        cls,
        input_ids: torch.Tensor,  # [batch_size] - single new token per request
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,  # Current sequence lengths (including new token)
        out_cache_loc: torch.Tensor,  # [batch_size] - positions for new tokens
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
    ):
        """
        Factory method for creating decode batch.
        
        Phase 4: In decode mode, each request contributes exactly one new token.
        """
        batch_size = len(req_pool_indices)
        
        # For decode, positions are the current sequence length minus 1 (0-indexed position for new token)
        positions = seq_lens - 1  # seq_lens represents total length, so new token is at seq_lens-1 position
        
        return cls(
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices, 
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            is_prefill=False,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (SGLang-compatible)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """Forward with optional fused residual (SGLang pattern)."""
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
            
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype), residual


class RotaryEmbedding(nn.Module):
    """RoPE implementation compatible with SGLang's interface."""
    
    def __init__(
        self, 
        head_dim: int, 
        max_position_embeddings: int = 32768, 
        base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Handle rope scaling if present (simplified)
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type")
            scaling_factor = rope_scaling.get("factor", 1.0)
            if scaling_type == "linear":
                self.base = base * scaling_factor
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        """
        Apply RoPE to q and k tensors (SGLang-compatible signature).
        positions: [total_tokens]
        q: [total_tokens, num_heads, head_dim]
        k: [total_tokens, num_kv_heads, head_dim]  # GQA supported
        """
        # Compute cos/sin without expanding to the head dimension to support GQA.
        freqs = torch.outer(positions.float(), self.inv_freq)  # [total_tokens, head_dim//2]
        cos = torch.cos(freqs)  # [total_tokens, head_dim//2]
        sin = torch.sin(freqs)  # [total_tokens, head_dim//2]

        # Apply rotation (broadcast across head dimension)
        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)

        return q_rot, k_rot

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Apply rotary position embedding with broadcasting across heads.
        x: [total_tokens, any_head_count, head_dim]
        cos/sin: [total_tokens, head_dim//2]
        """
        # Prepare cos/sin for broadcasting over the head axis and match dtype
        cos = cos.unsqueeze(1).to(x.dtype)  # [total_tokens, 1, head_dim//2]
        sin = sin.unsqueeze(1).to(x.dtype)  # [total_tokens, 1, head_dim//2]

        # Split last dimension into two halves and apply rotation
        x1, x2 = x.chunk(2, dim=-1)  # [total_tokens, any_head_count, head_dim//2] each
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SimpleAttentionBackend:
    """
    SGLang-aligned attention backend using token-level flat storage.
    
    Follows SGLang's actual design: simple token indexing with flat KV buffers,
    not complex paged attention. Based on SGLang source code analysis.
    """
    
    @staticmethod
    def forward(q, k, v, attention_layer, forward_batch: SimplifiedForwardBatch, save_kv_cache: bool = True):
        """
        Attention forward following SGLang's security model.
        
        SECURITY IMPLEMENTED: Processes each request separately to prevent cross-request token leakage.
        SGLang's approach: input tensors are concatenated for efficiency but attention is computed per-request.
        
        This implementation exactly follows SGLang's torch_native_backend.py approach.
        """
        layer_id = attention_layer.layer_id
        scaling = attention_layer.scaling
        
        # Step 1: Save K,V to memory pool (SGLang's exact pattern)
        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer_id, forward_batch.out_cache_loc, k, v)
        
        if forward_batch.is_prefill:
            # PREFILL: Process each request separately (SGLang's security approach)
            return SimpleAttentionBackend._prefill_attention_per_request(q, k, v, scaling, forward_batch)
        else:
            # DECODE: Process each request separately (SGLang's security approach)
            return SimpleAttentionBackend._decode_attention(q, k, v, scaling, forward_batch, attention_layer.layer_id)
    
    @staticmethod
    def _prefill_attention_per_request(q, k, v, scaling, forward_batch: SimplifiedForwardBatch):
        """
        Prefill attention with per-request processing (SGLang's approach).
        
        SECURITY: Each request is processed separately to prevent cross-request token leakage.
        This follows SGLang's torch_native_backend implementation.
        """
        outputs = []
        q_offset = 0
        kv_offset = 0
        
        # Process each request separately (like SGLang's _run_sdpa_forward_extend)
        for seq_len in forward_batch.seq_lens:
            seq_len = seq_len.item() if hasattr(seq_len, 'item') else seq_len
            
            # Extract Q, K, V for this specific request only
            per_req_q = q[q_offset:q_offset + seq_len]      # [seq_len, num_heads, head_dim]
            per_req_k = k[kv_offset:kv_offset + seq_len]    # [seq_len, num_kv_heads, head_dim]
            per_req_v = v[kv_offset:kv_offset + seq_len]    # [seq_len, num_kv_heads, head_dim]
            
            # Handle GQA: repeat K,V heads to match Q heads if needed
            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                # Repeat K,V heads to match Q heads (GQA)
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)  # [seq_len, num_heads, head_dim]
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)  # [seq_len, num_heads, head_dim]
            
            # Reshape for PyTorch SDPA: [batch=1, num_heads, seq_len, head_dim]  
            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0)
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0)
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0)
            
            # Per-request causal attention (SGLang's approach)
            per_req_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None, # None because is_causal=True handles causal masking internally in PyTorch SDPA
                dropout_p=0.0,
                is_causal=True  # Causal only within this request
            )
            
            # Reshape back and flatten: [seq_len, num_heads * head_dim]
            per_req_out = per_req_out.squeeze(0).transpose(0, 1).flatten(-2)
            outputs.append(per_req_out)
            
            q_offset += seq_len
            kv_offset += seq_len
        
        # Concatenate outputs back to original batch order
        return torch.cat(outputs, dim=0)
    
    @staticmethod
    def _prefill_attention(q, k, v, scaling):
        """Prefill attention with causal masking (SGLang pattern)."""
        seq_len, num_heads, head_dim = q.shape
        
        # Reshape for PyTorch attention: [batch=1, num_heads, seq_len, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)  
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        
        # Causal self-attention
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        
        # Reshape back: [seq_len, num_heads * head_dim]
        return out.squeeze(0).transpose(0, 1).flatten(-2)
    
    @staticmethod
    def _decode_attention(q, k, v, scaling, forward_batch, layer_id):
        """
        Decode attention following SGLang's per-request approach.
        
        SECURITY: Each request processes only its own context, naturally preventing leakage.
        This follows SGLang's _run_sdpa_forward_decode implementation exactly.
        """
        # In decode mode, each request has exactly 1 new query token
        # but needs to attend to its full context from KV cache
        
        outputs = []
        q_offset = 0
        
        for i, seq_len in enumerate(forward_batch.seq_lens):
            # Normalize scalar tensor to Python int for slicing and indexing
            seq_len = int(seq_len.item()) if hasattr(seq_len, "item") else int(seq_len)

            # Single query token for this request (SGLang's approach)
            per_req_q = q[q_offset:q_offset + 1]  # [1, num_heads, head_dim]
            
            # Get this request's full context from KV cache using SGLang's indexing
            # Ensure row index is a Python int and token indices are torch.long for advanced indexing
            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            token_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len].long()
            
            # Gather full context K,V for this request from cache (SGLang's core operation)
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
            per_req_k = k_buffer[token_indices]  # [seq_len, num_kv_heads, head_dim]
            per_req_v = v_buffer[token_indices]  # [seq_len, num_kv_heads, head_dim]
            
            # Handle GQA: repeat K,V heads to match Q heads if needed
            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                # Repeat K,V heads to match Q heads (GQA)
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)  # [seq_len, num_heads, head_dim]
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)  # [seq_len, num_heads, head_dim]
            
            # Reshape for PyTorch SDPA: [batch=1, num_heads, seq_len, head_dim]
            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0)     # [1, num_heads, 1, head_dim]
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0)     # [1, num_heads, seq_len, head_dim]
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0)
            
            # Per-request attention (SGLang's security approach - no causal mask needed for decode)
            per_req_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False  # No causal mask needed for decode (SGLang's approach)
            )
            
            # Reshape back: [1, num_heads * head_dim]
            per_req_out = per_req_out.squeeze(0).transpose(0, 1).flatten(-2)
            outputs.append(per_req_out)
            
            q_offset += 1  # Each request contributes 1 query token in decode
        
        return torch.cat(outputs, dim=0)  # [batch_size, num_heads * head_dim]


class QKVLinear(nn.Module):
    """
    Merged QKV linear layer (simplified version of SGLang's QKVParallelLinear).
    Combines Q, K, V projections for efficiency.
    """
    
    def __init__(self, input_size: int, num_heads: int, num_kv_heads: int, head_dim: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Compute output sizes
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.total_size = self.q_size + 2 * self.kv_size  # Q + K + V
        
        # Single merged linear layer
        self.qkv_proj = nn.Linear(input_size, self.total_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning split Q, K, V tensors."""
        qkv = self.qkv_proj(x)  # [seq_len, q_size + 2*kv_size]
        
        # Split into Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        return q, k, v


class BabyRadixAttention(nn.Module):
    """
    Simplified version of SGLang's RadixAttention.
    
    Maintains SGLang's interface but uses simplified PyTorch backend.
    No radix caching, sliding windows, or advanced optimizations.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads  
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.layer_id = layer_id
    
    def forward(self, q, k, v, forward_batch: SimplifiedForwardBatch, save_kv_cache: bool = True):
        """Forward pass using simplified attention backend (SGLang-compatible interface)."""
        if k is not None:
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.head_dim)
            v = v.view(-1, self.tp_v_head_num, self.head_dim)
        
        return SimpleAttentionBackend.forward(q, k, v, self, forward_batch, save_kv_cache)


class Qwen2Attention(nn.Module):
    """Qwen2 attention following SGLang's structure but simplified."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # Merged QKV projection (SGLang pattern)
        self.qkv_proj = QKVLinear(hidden_size, num_heads, num_kv_heads, self.head_dim, bias=True)
        
        # Output projection
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings,
            rope_theta,
            rope_scaling,
        )
        
        # Simplified RadixAttention
        self.attn = BabyRadixAttention(
            num_heads, self.head_dim, self.scaling, num_kv_heads, layer_id
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
    ) -> torch.Tensor:
        """Forward pass (SGLang-compatible signature)."""
        # Merged QKV projection
        q, k, v = self.qkv_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)  
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention computation
        attn_output = self.attn(q, k, v, forward_batch)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class GateUpLinear(nn.Module):
    """Merged gate and up projections (simplified version of MergedColumnParallelLinear)."""
    
    def __init__(self, input_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = nn.Linear(input_size, 2 * intermediate_size, bias=bias)
        self.intermediate_size = intermediate_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning gate and up projections."""
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return gate, up


class Qwen2MLP(nn.Module):
    """Qwen2 MLP with merged gate/up projections (SGLang pattern)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu supported.")
        
        # Merged gate and up projections
        self.gate_up_proj = GateUpLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with SwiGLU activation."""
        gate, up = self.gate_up_proj(x)
        return self.down_proj(F.silu(gate) * up)  # SwiGLU: SiLU(gate) * up


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer (SGLang-compatible)."""
    
    def __init__(
        self,
        config,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self attention
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
        )
        
        # MLP
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (SGLang-compatible signature)."""
        
        # Self attention with residual
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        
        # MLP with residual  
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class BabyQwen2Model(nn.Module):
    """
    Simplified Qwen2 model for baby-sglang.
    
    Maintains SGLang's interface and structure but simplified for MVP requirements:
    - Single GPU, static batching
    - No tensor parallelism or quantization
    - PyTorch-only implementation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Token embeddings (simplified - no parallel embedding)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers  
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass (SGLang-compatible signature)."""
        
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class BabyQwen2ForCausalLM(nn.Module):
    """Qwen2 causal language model (SGLang-compatible but simplified)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BabyQwen2Model(config)
        
        # LM head
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass returning logits (SGLang-compatible signature)."""
        
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        
        # Get logits
        if hasattr(self.lm_head, 'weight') and self.lm_head.weight is self.model.embed_tokens.weight:
            # Tied weights case
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        return logits
    
    def load_weights_from_hf(self, hf_model_path: str, device: str = "auto"):
        """
        Load weights from a HuggingFace checkpoint into the simplified model.
        Chooses dtype by device: float16 for CUDA/MPS, float32 for CPU.
        Avoids requiring `accelerate` by not using `device_map`.
        """
        try:
            from transformers import AutoModelForCausalLM
            import torch

            logger.info(f"Loading Qwen2 weights from: {hf_model_path}")

            # Resolve device and dtype
            target_device = device
            if device == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            torch_dtype = torch.float16 if target_device in ("cuda", "mps") else torch.float32

            # Load HuggingFace model on CPU to read state dict (no accelerate needed)
            logger.info("Loading HuggingFace model (CPU) to fetch state_dict...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=False  # ensure standard torch.load path without accelerate
            )

            state = hf_model.state_dict()
            logger.info(f"Loaded {len(state)} tensors from HuggingFace model")

            # Copy into our simplified model
            self.load_weights(state)

            # Move to target device and dtype
            self.to(device=target_device, dtype=torch_dtype)

            # Cleanup
            del hf_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Successfully loaded Qwen2 weights from {hf_model_path} onto {target_device} with dtype={torch_dtype}")

        except Exception as e:
            logger.error(f"Failed to load HF weights: {e}")
            raise
    
    def load_weights(self, weights):
        """
        Load weights from a HuggingFace state_dict into the simplified model.
        Supports merged QKV and merged Gate/Up projections.
        """
        import re
        import torch

        # Accept dict or iterator
        state = weights if isinstance(weights, dict) else dict(weights)
        params = dict(self.named_parameters())

        # Shapes derived from config
        cfg = self.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q_size = cfg.num_attention_heads * head_dim
        kv_size = cfg.num_key_value_heads * head_dim
        inter = cfg.intermediate_size

        def copy_param(dst_name: str, src_tensor: torch.Tensor):
            if dst_name not in params:
                return
            dst = params[dst_name]
            if dst.data.shape != src_tensor.shape:
                logger.warning(f"Size mismatch for {dst_name}: {tuple(dst.data.shape)} vs {tuple(src_tensor.shape)}")
                return
            dst.data.copy_(src_tensor.to(dst.dtype))

        layer_pat = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

        for name, tensor in state.items():
            # Top-level weights
            if name == "model.embed_tokens.weight":
                copy_param("model.embed_tokens.weight", tensor)
                continue
            if name == "model.norm.weight":
                copy_param("model.norm.weight", tensor)
                continue
            if name == "lm_head.weight":
                # Copy only if not tied to embeddings
                if hasattr(self, "lm_head") and (not hasattr(self.model, "embed_tokens") or self.lm_head is not self.model.embed_tokens):
                    copy_param("lm_head.weight", tensor)
                continue

            # Per-layer weights
            m = layer_pat.match(name)
            if not m:
                # Skip unknown keys (e.g., rotary tables, caches)
                continue
            lid = int(m.group(1))
            rest = m.group(2)

            # Attention QKV merged into qkv_proj.qkv_proj
            if rest == "self_attn.q_proj.weight":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
                if p is not None:
                    p.data[:q_size].copy_(tensor.to(p.dtype))
                continue
            if rest == "self_attn.q_proj.bias":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
                if p is not None:
                    p.data[:q_size].copy_(tensor.to(p.dtype))
                continue

            if rest == "self_attn.k_proj.weight":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
                if p is not None:
                    p.data[q_size:q_size + kv_size].copy_(tensor.to(p.dtype))
                continue
            if rest == "self_attn.k_proj.bias":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
                if p is not None:
                    p.data[q_size:q_size + kv_size].copy_(tensor.to(p.dtype))
                continue

            if rest == "self_attn.v_proj.weight":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
                if p is not None:
                    p.data[q_size + kv_size:].copy_(tensor.to(p.dtype))
                continue
            if rest == "self_attn.v_proj.bias":
                p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
                if p is not None:
                    p.data[q_size + kv_size:].copy_(tensor.to(p.dtype))
                continue

            if rest == "self_attn.o_proj.weight":
                copy_param(f"model.layers.{lid}.self_attn.o_proj.weight", tensor)
                continue
            if rest == "self_attn.o_proj.bias":
                copy_param(f"model.layers.{lid}.self_attn.o_proj.bias", tensor)
                continue

            # MLP merged Gate/Up into gate_up_proj.gate_up_proj
            if rest == "mlp.gate_proj.weight":
                p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.weight", None)
                if p is not None:
                    p.data[:inter].copy_(tensor.to(p.dtype))
                continue
            if rest == "mlp.gate_proj.bias":
                p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.bias", None)
                if p is not None:
                    p.data[:inter].copy_(tensor.to(p.dtype))
                continue

            if rest == "mlp.up_proj.weight":
                p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.weight", None)
                if p is not None:
                    p.data[inter:].copy_(tensor.to(p.dtype))
                continue
            if rest == "mlp.up_proj.bias":
                p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.bias", None)
                if p is not None:
                    p.data[inter:].copy_(tensor.to(p.dtype))
                continue

            if rest == "mlp.down_proj.weight":
                copy_param(f"model.layers.{lid}.mlp.down_proj.weight", tensor)
                continue
            if rest == "mlp.down_proj.bias":
                copy_param(f"model.layers.{lid}.mlp.down_proj.bias", tensor)
                continue

            # Norms
            if rest == "input_layernorm.weight":
                copy_param(f"model.layers.{lid}.input_layernorm.weight", tensor)
                continue
            if rest == "post_attention_layernorm.weight":
                copy_param(f"model.layers.{lid}.post_attention_layernorm.weight", tensor)
                continue
    
    def _load_weight_shard(self, param: torch.Tensor, loaded_weight: torch.Tensor, shard_id):
        """Load weight into a specific shard (for merged parameters like qkv_proj)."""
        if shard_id == "q":
            # Q shard - first portion
            q_size = self.model.layers[0].self_attn.qkv_proj.q_size
            param.data[:q_size].copy_(loaded_weight)
        elif shard_id == "k":  
            # K shard - middle portion
            q_size = self.model.layers[0].self_attn.qkv_proj.q_size
            kv_size = self.model.layers[0].self_attn.qkv_proj.kv_size
            param.data[q_size:q_size + kv_size].copy_(loaded_weight)
        elif shard_id == "v":
            # V shard - last portion  
            q_size = self.model.layers[0].self_attn.qkv_proj.q_size
            kv_size = self.model.layers[0].self_attn.qkv_proj.kv_size
            param.data[q_size + kv_size:].copy_(loaded_weight)
        elif shard_id == 0:
            # Gate shard (first half)
            intermediate_size = loaded_weight.size(0)
            param.data[:intermediate_size].copy_(loaded_weight)
        elif shard_id == 1:
            # Up shard (second half)
            intermediate_size = loaded_weight.size(0) 
            param.data[intermediate_size:].copy_(loaded_weight)
        else:
            # Fallback to direct copy
            param.data.copy_(loaded_weight)
    
    def _load_weight_direct(self, param: torch.Tensor, loaded_weight: torch.Tensor):
        """Load weight directly (for non-merged parameters)."""
        if param.size() != loaded_weight.size():
            logger.warning(f"Size mismatch: param {param.size()} vs loaded {loaded_weight.size()}")
            return
        param.data.copy_(loaded_weight)
# ========= Convenience config and HF loader/test helpers (appended for TODO#1) =========

@dataclass
class BabyQwenConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, model_id: str):
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cls(
            hidden_size=getattr(hf_cfg, "hidden_size"),
            num_attention_heads=getattr(hf_cfg, "num_attention_heads"),
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", getattr(hf_cfg, "num_kv_heads", None)),
            intermediate_size=getattr(hf_cfg, "intermediate_size"),
            num_hidden_layers=getattr(hf_cfg, "num_hidden_layers"),
            vocab_size=getattr(hf_cfg, "vocab_size"),
            rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
            hidden_act=getattr(hf_cfg, "hidden_act", "silu"),
            max_position_embeddings=getattr(hf_cfg, "max_position_embeddings", 32768),
            rope_theta=getattr(hf_cfg, "rope_theta", 1000000.0),
            rope_scaling=getattr(hf_cfg, "rope_scaling", None),
            tie_word_embeddings=getattr(hf_cfg, "tie_word_embeddings", False),
        )


def build_model_from_hf(model_id: str, device: str = "auto") -> BabyQwen2ForCausalLM:
    """
    Create BabyQwen2ForCausalLM from a HuggingFace model id by adapting the HF config
    and then copying weights into the simplified architecture.
    """
    cfg = BabyQwenConfig.from_hf(model_id)
    # Some HF configs may miss num_key_value_heads; default to num_attention_heads
    if cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads

    model = BabyQwen2ForCausalLM(cfg)
    model.eval()
    model.load_weights_from_hf(model_id, device=device)
    return model


def _best_device_str(device: str = "auto") -> Tuple[str, torch.dtype]:
    if device != "auto":
        dt = torch.float16 if device in ("cuda", "mps") else torch.float32
        return device, dt
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def simple_prefill_test(model_id: str = "Qwen/Qwen2.5-0.5B", prompt: str = "Hello, how are you?"):
    """
    Minimal verification: load HF weights into simplified model, run a single prefill forward,
    and print the next-token greedy prediction.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Build model and tokenizer
    model = build_model_from_hf(model_id, device="auto")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].squeeze(0)  # [L]
    L = input_ids.numel()

    # Resolve device/dtype from model
    p = next(model.parameters())
    device = p.device
    dtype = p.dtype

    input_ids = input_ids.to(device)

    # Create memory pools
    # KV pool uses num_kv_heads and head_dim
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    kv_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    token_to_kv_pool = MHATokenToKVPool(
        size=L,
        dtype=kv_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=model.config.num_hidden_layers,
        device=str(device),
    )

    req_to_token_pool = ReqToTokenPool(
        size=1,
        max_context_len=L,
        device=str(device),
    )

    # Allocate KV slots and map request->token slots
    out_cache_loc = token_to_kv_pool.alloc(L)  # int32 on device
    mapping_row = torch.zeros(L, dtype=torch.int32, device=device)
    mapping_row[:L] = out_cache_loc
    # Write into req_to_token_pool for request index 0
    req_to_token_pool.write(0, mapping_row)

    # Build forward batch (prefill)
    req_pool_indices = torch.tensor([0], dtype=torch.long, device=device)
    seq_lens = torch.tensor([L], dtype=torch.long, device=device)

    fwd_batch = SimplifiedForwardBatch.create_prefill_batch(
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
    )

    # Positions [0..L-1]
    positions = torch.arange(L, dtype=torch.long, device=device)

    # Forward to get logits and greedy next token
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=fwd_batch,
            input_embeds=None,
        )

    last_logits = logits[-1]  # [vocab_size]
    next_id = int(torch.argmax(last_logits, dim=-1).item())

    # Decode and print
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    next_token_text = tokenizer.decode(next_id, skip_special_tokens=True)
    print("=" * 40)
    print("baby-sglang Qwen2.5-0.5B simple prefill test")
    print(f"Prompt: {prompt}")
    print(f"Next token id: {next_id}")
    print(f"Next token text: {repr(next_token_text)}")
    print("=" * 40)


def simple_decode_test(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 64,
):
    """
    Simple end-to-end decode test on several prompts using the baby-sglang execution path:
    - Prefill to populate KV for the prompt
    - Iterative decode with per-token KV write + per-request attention read
    Greedy decoding only for MVP.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Default prompts to cover English/Chinese and code
    if prompts is None:
        prompts = [
            "Hello, introduce yourself briefly.",
            "用中文介绍一下量子计算的基本原理。",
            "Write a Python function to compute Fibonacci numbers efficiently.",
            "给出三条关于上海旅游的建议。",
        ]

    # Build model + tokenizer once
    model = build_model_from_hf(model_id, device="auto")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Resolve device/dtype from model
    p = next(model.parameters())
    device = p.device
    dtype = p.dtype

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    for prompt in prompts:
        # Tokenize prompt
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids_full = enc["input_ids"].squeeze(0).to(device)
        L = int(input_ids_full.numel())

        # Create pools sized for prompt + generation
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // num_heads
        kv_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

        token_to_kv_pool = MHATokenToKVPool(
            size=L + max_new_tokens,
            dtype=kv_dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=model.config.num_hidden_layers,
            device=str(device),
        )
        req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=L + max_new_tokens,
            device=str(device),
        )

        # Allocate KV slots for prefill and map request->token slots
        prefill_locs = token_to_kv_pool.alloc(L)  # [L] int32 on device
        mapping_row = torch.zeros(L + max_new_tokens, dtype=torch.int32, device=device)
        mapping_row[:L] = prefill_locs
        req_to_token_pool.write(0, mapping_row)

        # Prefill batch
        fwd_batch = SimplifiedForwardBatch.create_prefill_batch(
            input_ids=input_ids_full,
            req_pool_indices=torch.tensor([0], dtype=torch.long, device=device),
            seq_lens=torch.tensor([L], dtype=torch.long, device=device),
            out_cache_loc=prefill_locs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
        )
        positions = torch.arange(L, dtype=torch.long, device=device)

        # Prefill forward to get first next-token logits
        with torch.no_grad():
            logits = model(
                input_ids=input_ids_full,
                positions=positions,
                forward_batch=fwd_batch,
                input_embeds=None,
            )
        last_logits = logits[-1]
        next_id = int(torch.argmax(last_logits, dim=-1).item())

        # Greedy decode loop
        generated_ids: List[int] = []
        steps = 0
        while steps < max_new_tokens:
            # Append the token we are about to feed (so EOS appears in output if selected)
            generated_ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

            # Allocate KV slot for this new token and update request->token map
            new_loc = token_to_kv_pool.alloc(1)  # shape [1], int32 on device
            # Current sequence length after adding this token
            cur_len = L + len(generated_ids)
            # Write new location into the mapping
            req_to_token_pool.req_to_token[0, cur_len - 1] = new_loc[0]

            # Build decode batch: feed the last generated token
            dec_input = torch.tensor([next_id], dtype=torch.long, device=device)
            dec_fwd = SimplifiedForwardBatch.create_decode_batch(
                input_ids=dec_input,
                req_pool_indices=torch.tensor([0], dtype=torch.long, device=device),
                seq_lens=torch.tensor([cur_len], dtype=torch.long, device=device),
                out_cache_loc=new_loc,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool=token_to_kv_pool,
            )
            dec_positions = dec_fwd.positions  # seq_lens - 1

            # Forward for next logits
            with torch.no_grad():
                dec_logits = model(
                    input_ids=dec_input,
                    positions=dec_positions,
                    forward_batch=dec_fwd,
                    input_embeds=None,
                )
            # Since batch=1 and one token, take [0]
            next_id = int(torch.argmax(dec_logits[0], dim=-1).item())
            steps += 1

        # Decode completion text
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("=" * 40)
        print("baby-sglang Qwen2.5-0.5B simple decode test")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion!r}")
        print("=" * 40)
        
if __name__ == "__main__":
    # Run quick verifications when invoked directly
    try:
        simple_prefill_test()
        simple_decode_test()
    except Exception as _e:
        logger.exception(f"Simple test failed: {_e}")