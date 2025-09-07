import torch
import torch.nn.functional as F
from forward_batch import SimplifiedForwardBatch

class SimpleAttentionBackend:

    @staticmethod
    def forward(q, k, v, attention_layer, forward_batch: SimplifiedForwardBatch, save_kv_cache: bool = True):
        layer_id = attention_layer.layer_id
        scaling = attention_layer.scaling
        
        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer_id, forward_batch.out_cache_loc, k, v)
        
        if forward_batch.is_prefill:
            return SimpleAttentionBackend._prefill_attention(q, k, v, scaling, forward_batch)
        else:
            return SimpleAttentionBackend._decode_attention(q, k, v, scaling, forward_batch, attention_layer.layer_id)

    @staticmethod
    def _prefill_attention(q, k, v, scaling, forward_batch: SimplifiedForwardBatch):
        outputs = []
        q_offset = 0
        kv_offset = 0
        
        for seq_len in forward_batch.seq_lens:
            seq_len = seq_len.item() if hasattr(seq_len, 'item') else seq_len
        
            per_req_q = q[q_offset:q_offset + seq_len]   # [t, n_q, d]
            per_req_k = k[kv_offset:kv_offset + seq_len] # [t, n_kv, d]
            per_req_v = v[kv_offset:kv_offset + seq_len] # [t, n_kv, d]
        
            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)
        
            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0) # [1, n_q, t, d]
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0) # [1, n_q, t, d]
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0) # [1, n_q, t, d]
        
            per_req_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
            
            per_req_out = per_req_out.squeeze(0).transpose(0, 1).flatten(-2) # [t, n_q*d]
            
            outputs.append(per_req_out)
            q_offset += seq_len
            kv_offset += seq_len

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _decode_attention(q, k, v, scaling, forward_batch, layer_id):
        outputs = []
        q_offset = 0
        
        for i, seq_len in enumerate(forward_batch.seq_lens):
            seq_len = int(seq_len.item()) if hasattr(seq_len, "item") else int(seq_len)
        
            per_req_q = q[q_offset:q_offset + 1] # [1, n_q, d]
        
            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            token_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len].long()
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
            per_req_k = k_buffer[token_indices]
            per_req_v = v_buffer[token_indices]
        
            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)
        
            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0)
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0)
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0)
        
            per_req_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        
            per_req_out = per_req_out.squeeze(0).transpose(0, 1).flatten(-2)
            outputs.append(per_req_out)
            q_offset += 1
        
        return torch.cat(outputs, dim=0)