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
        """Apply RoPE to q and k tensors (SGLang-compatible signature)."""
        # positions: [total_tokens], q/k: [total_tokens, num_heads, head_dim]
        freqs = torch.outer(positions.float(), self.inv_freq)  # [total_tokens, head_dim//2]
        cos = torch.cos(freqs).unsqueeze(1)  # [total_tokens, 1, head_dim//2]
        sin = torch.sin(freqs).unsqueeze(1)
        
        # Expand cos/sin to match q/k num_heads dimension
        num_heads = q.shape[1]
        cos = cos.expand(-1, num_heads, -1)  # [total_tokens, num_heads, head_dim//2]
        sin = sin.expand(-1, num_heads, -1)
        
        # Apply rotation
        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary position embedding."""
        # x: [total_tokens, num_heads, head_dim]
        # cos/sin: [total_tokens, num_heads, head_dim//2]
        x1, x2 = x.chunk(2, dim=-1)  # Each: [total_tokens, num_heads, head_dim//2]
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
                is_causal=True,  # Causal only within this request
                scale=scaling
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
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scaling
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
            # Single query token for this request (SGLang's approach)
            per_req_q = q[q_offset:q_offset + 1]  # [1, num_heads, head_dim]
            
            # Get this request's full context from KV cache using SGLang's indexing
            req_pool_idx = forward_batch.req_pool_indices[i]
            token_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
            
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
                is_causal=False,  # No causal mask needed for decode (SGLang's approach)
                scale=scaling
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
        Load weights from HuggingFace checkpoint using transformers library.
        
        Simplified approach following SGLang patterns - let transformers handle the heavy lifting.
        
        Args:
            hf_model_path: Path to HF model directory
            device: Device to load weights on ("auto", "cpu", "cuda", "mps")
        """
        try:
            from transformers import AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Qwen2 weights from: {hf_model_path}")
            
            # Load HuggingFace model to get the state_dict
            logger.info("Loading HuggingFace model...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.float16,  # Use FP16 for efficiency
                device_map="cpu",  # Load to CPU first
                trust_remote_code=True
            )
            
            # Get the state dict from HuggingFace model
            hf_state_dict = hf_model.state_dict()
            logger.info(f"Loaded {len(hf_state_dict)} weight tensors from HuggingFace model")
            
            # Load weights using SGLang-style iterator
            weights_iterator = ((name, tensor) for name, tensor in hf_state_dict.items())
            self.load_weights(weights_iterator)
            
            # Clean up HF model to save memory
            del hf_model
            torch.cuda.empty_cache()
            
            logger.info(f"Successfully loaded Qwen2 weights from {hf_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load HF weights: {e}")
            raise
    
    def load_weights(self, weights):
        """
        Load weights using SGLang's approach with proper weight mapping.
        
        This method follows SGLang's load_weights pattern from qwen2.py.
        """
        # SGLang-style stacked parameter mappings for merged weights
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)  
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"), 
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Skip rotary embeddings and other unused weights
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
                
            # Handle stacked parameter mapping (like SGLang)  
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                
                # Load the shard
                self._load_weight_shard(param, loaded_weight, shard_id)
                break
            else:
                # Direct weight loading
                if name not in params_dict:
                    continue  # Skip unknown weights
                param = params_dict[name]
                self._load_weight_direct(param, loaded_weight)
    
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