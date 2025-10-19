import torch
import torch.nn.functional as F
from forward_batch import SimplifiedForwardBatch


class SimpleAttentionBackend:

    @staticmethod
    def forward(
        q,
        k,
        v,
        attention_layer,
        forward_batch: SimplifiedForwardBatch,
        save_kv_cache: bool = True,
    ):
        layer_id = attention_layer.layer_id
        scaling = attention_layer.scaling

        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer_id, forward_batch.out_cache_loc, k, v
            )

        if forward_batch.is_prefill:
            return SimpleAttentionBackend._prefill_attention(
                q, k, v, scaling, forward_batch, layer_id
            )
        else:
            return SimpleAttentionBackend._decode_attention(
                q, k, v, scaling, forward_batch, layer_id
            )

    @staticmethod
    def _prefill_attention(
        q, k, v, scaling, forward_batch: SimplifiedForwardBatch, layer_id: int
    ):
        outputs = []
        q_offset = 0

        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)

        for i in range(forward_batch.batch_size):
            extended_len = int(forward_batch.extended_lens[i].item())
            if extended_len == 0:
                continue  # nothing new to process

            prefix_len = int(forward_batch.prefix_lens[i].item())
            full_len = int(forward_batch.seq_lens[i].item())

            per_req_q = q[q_offset : q_offset + extended_len]
            q_offset += extended_len

            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            token_indices_full = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :full_len
            ].long()
            per_req_k = k_buffer[token_indices_full]  # [full_len, n_kv, d]
            per_req_v = v_buffer[token_indices_full]  # [full_len, n_kv, d]

            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)

            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0)  # [1, n_q, t_new, d]
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0)  # [1, n_q, full_len, d]
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0)  # [1, n_q, full_len, d]

            # Build boolean mask [1, 1, t_new, full_len] (True = masked)
            device = per_req_q.device
            row_positions = torch.arange(
                prefix_len, prefix_len + extended_len, device=device
            ).unsqueeze(
                1
            )  # [t_new, 1]
            col_positions = torch.arange(full_len, device=device).unsqueeze(
                0
            )  # [1, full_len]

            mask_2d = (
                col_positions > row_positions
            )  # [t_new, full_len] - True means masked
            attn_mask = mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, t_new, full_len]

            dtype = q_sdpa.dtype
            attn_bias = torch.zeros(
                (1, 1, extended_len, full_len), dtype=dtype, device=device
            )
            attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))

            per_req_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False,
            )

            per_req_out = (
                per_req_out.squeeze(0).transpose(0, 1).flatten(-2)
            )  # [t_new, n_q*d]
            outputs.append(per_req_out)

        if len(outputs) == 0:
            return q.new_zeros((0, q.shape[1] * q.shape[2]))  # empty
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _decode_attention(q, k, v, scaling, forward_batch, layer_id):
        outputs = []
        q_offset = 0

        for i, seq_len in enumerate(forward_batch.seq_lens):
            seq_len = int(seq_len.item()) if hasattr(seq_len, "item") else int(seq_len)

            per_req_q = q[q_offset : q_offset + 1]  # [1, n_q, d]

            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            token_indices = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :seq_len
            ].long()
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
                q_sdpa, k_sdpa, v_sdpa, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            per_req_out = per_req_out.squeeze(0).transpose(0, 1).flatten(-2)
            outputs.append(per_req_out)
            q_offset += 1

        return torch.cat(outputs, dim=0)
