import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention

def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """
    一个模拟的 PagedAttention KV 缓存写入操作。
    在真实系统中，这是一个高性能的 CUDA Kernel。
    
    Args:
        key: 当前计算出的 Key，形状为 [num_tokens, num_heads, head_dim]
        value: 当前计算出的 Value，形状为 [num_tokens, num_heads, head_dim]
        key_cache: 全局的 K 缓存池
        value_cache: 全局的 V 缓存池
        slot_mapping: 一个一维张量，指明每个 token 应该被写入到缓存池的哪个“槽位”（slot）。
                      slot_mapping[i] = physical_block_idx * block_size + intra_block_offset
    """
    # 将 key 和 value 的形状调整为适合 scatter 操作
    # SGLang/vLLM 中，key/value_cache 的形状通常是 [num_blocks, num_heads, head_dim, block_size]
    # 或类似的变体。为简化，这里我们使用更简单的 [num_total_slots, num_heads, head_dim]
    # 然后直接用 index_copy_ 进行写入。
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)

class LlamaAttentionWithPagedCache(nn.Module):
    """
    一个支持 PagedAttention 的 LlamaAttention 实现。
    它将替换掉原始模型中的 LlamaAttention 模块。
    """
    def __init__(self, original_attn: LlamaAttention):
        super().__init__()
        # 从原始的 Attention 模块复制所有权重和配置
        self.config = original_attn.config
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_params: dict,
        **kwargs, # 吸收掉 attention_mask 等不再需要的参数
    ):
        # 1. 从 kv_cache_params 中解包出 PagedAttention 需要的全部信息
        context_lens = kv_cache_params["context_lens"]
        block_tables = kv_cache_params["block_tables"]
        key_cache_pool, value_cache_pool = kv_cache_params["kv_cache_pool"]
        
        # 2. 计算 Q, K, V
        bsz, q_len, _ = hidden_states.size() # 在 prefill 阶段, q_len > 1
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用旋转位置编码 (RoPE)
        # rotary_emb 需要知道每个 token 在其序列中的真实位置
        kv_seq_len = context_lens.max() # 获取当前批次中最长序列的长度
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = torch.nn.functional.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # 4. 将计算出的 K/V 写入到物理缓存池中 (PagedAttention的核心)
        # TODO: 这里需要一个高效的 CUDA kernel, 我们用 torch 操作模拟
        # reshape_and_cache(key_states, value_states, key_cache_pool, value_cache_pool, slot_mapping)

        # 5. 执行注意力计算
        # 在一个完整的 PagedAttention 实现中，这里的 K 和 V 是从 `key_cache_pool` 
        # 和 `value_cache_pool` 中根据 `block_tables` gather 出来的。
        # 为简化，我们暂时仍然使用当前计算出的 KV，但这并未完全利用 PagedAttention 的优势。
        # 这是一个关键的 TODO，完整的实现需要一个 PagedAttention Kernel。
        # 我们使用 PyTorch 2.0 的 `scaled_dot_product_attention`
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 6. 输出投影
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None # 返回与原始 LlamaAttention 兼容的元组


def patch_model(model: nn.Module):
    """
    遍历模型的所有子模块，并用我们自己的 LlamaAttentionWithPagedCache
    替换掉原始的 LlamaAttention 模块。
    """
    for name, module in model.named_children():
        if isinstance(module, LlamaAttention):
            # 找到一个 LlamaAttention 模块，进行替换
            # `model._modules[name]` 是访问和修改子模块的标准方式
            model._modules[name] = LlamaAttentionWithPagedCache(module)
        elif len(list(module.children())) > 0:
            # 如果模块还有子模块，则递归进入
            patch_model(module)