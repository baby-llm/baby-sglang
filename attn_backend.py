import torch
import torch.nn.functional as F
from forward_batch import SimplifiedForwardBatch
import logging
import os

# Configure attention debug logger
_attn_logger = logging.getLogger("baby_sgl.attn")
if not _attn_logger.handlers:
    _attn_logger.setLevel(logging.DEBUG)
    os.makedirs("baby-sgl/debug", exist_ok=True)
    fh = logging.FileHandler("baby-sgl/debug/attn_debug.log")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    _attn_logger.addHandler(fh)


def _log_tensor_stats(name, t: torch.Tensor):
    try:
        _attn_logger.debug(
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
        )
        if t.numel() > 0 and t.dtype.is_floating_point:
            _attn_logger.debug(
                f"{name}: min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}"
            )
        elif t.numel() > 0 and t.dtype in (torch.int32, torch.int64):
            _attn_logger.debug(
                f"{name}: min={int(t.min().item())} max={int(t.max().item())}"
            )
    except Exception as e:
        _attn_logger.warning(f"_log_tensor_stats error for {name}: {e}")


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
        _attn_logger.debug(
            f"forward: layer_id={layer_id} is_prefill={forward_batch.is_prefill} save_kv_cache={save_kv_cache}"
        )
        _log_tensor_stats("q", q)
        if k is not None and v is not None:
            _log_tensor_stats("k", k)
            _log_tensor_stats("v", v)

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
            # positions slice for this request (new tokens)
            pos_slice = forward_batch.positions[q_offset : q_offset + extended_len]
            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            _attn_logger.debug(
                f"prefill[i={i}]: prefix_len={prefix_len} full_len={full_len} new_len={extended_len} req_pool_idx={req_pool_idx}"
            )
            _log_tensor_stats(f"prefill[i={i}] per_req_q", per_req_q)
            _log_tensor_stats(f"prefill[i={i}] positions_slice", pos_slice)
            # sanity: positions should be consecutive [prefix_len..prefix_len+new_len-1]
            try:
                if extended_len > 0:
                    expected = torch.arange(
                        prefix_len,
                        prefix_len + extended_len,
                        device=pos_slice.device,
                        dtype=pos_slice.dtype,
                    )
                    pos_ok = torch.equal(pos_slice, expected)
                    if not pos_ok:
                        _attn_logger.warning(
                            f"prefill[i={i}] positions mismatch: got={pos_slice.tolist()} expected={expected.tolist()}"
                        )
            except Exception as e:
                _attn_logger.warning(f"prefill[i={i}] positions check error: {e}")
            q_offset += extended_len

            token_indices_full = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :full_len
            ].long()
            _log_tensor_stats(f"prefill[i={i}] token_indices_full", token_indices_full)
            if (
                token_indices_full.numel() > 0
                and int(token_indices_full.min().item()) == 0
            ):
                _attn_logger.warning(
                    f"prefill[i={i}] token_indices_full contains 0 (padding) within full_len={full_len}"
                )

            per_req_k = k_buffer[token_indices_full]  # [full_len, n_kv, d]
            per_req_v = v_buffer[token_indices_full]  # [full_len, n_kv, d]

            num_heads = per_req_q.shape[1]
            num_kv_heads = per_req_k.shape[1]
            if num_kv_heads < num_heads:
                assert (
                    num_heads % num_kv_heads == 0
                ), f"Head mismatch: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
                repeat_factor = num_heads // num_kv_heads
                per_req_k = per_req_k.repeat_interleave(repeat_factor, dim=1)
                per_req_v = per_req_v.repeat_interleave(repeat_factor, dim=1)

            q_sdpa = per_req_q.transpose(0, 1).unsqueeze(0)  # [1, n_q, t_new, d]
            k_sdpa = per_req_k.transpose(0, 1).unsqueeze(0)  # [1, n_q, full_len, d]
            v_sdpa = per_req_v.transpose(0, 1).unsqueeze(0)  # [1, n_q, full_len, d]

            # Build boolean mask [1, 1, t_new, full_len] (True = masked)
            device = per_req_q.device
            full_idx = (
                torch.arange(full_len, device=device)
                .unsqueeze(0)
                .expand(extended_len, -1)
            )  # [t_new, full_len]
            allowed = (
                prefix_len + torch.arange(extended_len, device=device) + 1
            )  # [t_new]
            mask_2d = full_idx >= allowed.unsqueeze(1)  # [t_new, full_len]
            attn_mask = mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, t_new, full_len]

            # Enhanced mask debugging
            try:
                masked_counts = mask_2d.sum(dim=1)  # [t_new]
                unmasked_counts = full_len - masked_counts  # [t_new]

                # Log detailed mask analysis for first and last token
                if extended_len > 0:
                    first_token_mask_info = {
                        "position": prefix_len,
                        "allowed_until": allowed[0].item(),
                        "can_attend_count": unmasked_counts[0].item(),
                        "masked_count": masked_counts[0].item(),
                    }
                    last_token_mask_info = {
                        "position": prefix_len + extended_len - 1,
                        "allowed_until": allowed[-1].item(),
                        "can_attend_count": unmasked_counts[-1].item(),
                        "masked_count": masked_counts[-1].item(),
                    }

                    # Log actual attention pattern for first new token
                    first_token_attention_range = []
                    for j in range(full_len):
                        if not mask_2d[0, j]:
                            first_token_attention_range.append(j)

                    _attn_logger.debug(
                        f"prefill[i={i}] MASK_DETAILS: full_len={full_len}, "
                        f"first_token={first_token_mask_info}, "
                        f"last_token={last_token_mask_info}, "
                        f"first_token_attends_positions={first_token_attention_range[:10]}{'...' if len(first_token_attention_range) > 10 else ''}"
                    )

                    # Validate causal property
                    expected_unmasked = [
                        prefix_len + k + 1 for k in range(extended_len)
                    ]
                    actual_unmasked = unmasked_counts.tolist()
                    if expected_unmasked != actual_unmasked:
                        _attn_logger.warning(
                            f"prefill[i={i}] MASK_VALIDATION: expected_unmasked={expected_unmasked}, actual_unmasked={actual_unmasked}"
                        )

            except Exception as e:
                _attn_logger.warning(f"prefill[i={i}] enhanced mask logging error: {e}")

            per_req_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
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
            _attn_logger.debug(
                f"decode[i={i}]: seq_len={seq_len} req_pool_idx={req_pool_idx}"
            )
            _log_tensor_stats(f"decode[i={i}] token_indices", token_indices)
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
