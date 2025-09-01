"""
Minimal Qwen2 model implementation for baby-sglang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from forward_batch import SimplifiedForwardBatch
from memory_pool import ReqToTokenPool, MHATokenToKVPool

class BabyQwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BabyQwen2Model(config)

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
        
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        if hasattr(self.lm_head, 'weight') and self.lm_head.weight is self.model.embed_tokens.weight:
            # if tie word embeddings
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        return logits
    
class BabyQwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # 1. embed layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 2. decode layer
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # 3. final norm layer
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states

class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
        )

        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
            
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype), residual

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
        
        self.qkv_proj = QKVLinear(hidden_size, num_heads, num_kv_heads, self.head_dim, bias=True)
        
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings,
            rope_theta,
            rope_scaling,
        )
        
        self.attn = BabyRadixAttention(
            num_heads, self.head_dim, self.scaling, num_kv_heads, layer_id
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: SimplifiedForwardBatch,
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states)
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        
        output = self.o_proj(attn_output)
        
        return output

class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu supported.")
        
        self.gate_up_proj = GateUpLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)
        return self.down_proj(F.silu(gate) * up) # SwiGLU
    
class GateUpLinear(nn.Module):    
    def __init__(self, input_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = nn.Linear(input_size, 2 * intermediate_size, bias=bias)
        self.intermediate_size = intermediate_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return gate, up
    
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
        # if k is not None:
        #     assert v is not None
        #     k = k.view(-1, self.tp_k_head_num, self.head_dim)
        #     v = v.view(-1, self.tp_v_head_num, self.head_dim)
        
        return SimpleAttentionBackend.forward(q, k, v, self, forward_batch, save_kv_cache)
    
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
            return SimpleAttentionBackend._prefill_attention(q, k, v, scaling, forward_batch)
        else:
            # DECODE: Process each request separately (SGLang's security approach)
            return SimpleAttentionBackend._decode_attention(q, k, v, scaling, forward_batch, attention_layer.layer_id)
    
    @staticmethod
    def _prefill_attention(q, k, v, scaling, forward_batch: SimplifiedForwardBatch):
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
        
        return torch.cat(outputs, dim=0)