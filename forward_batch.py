from dataclasses import dataclass
from typing import Any
import torch
import logging
import os

from memory_pool import ReqToTokenPool, MHATokenToKVPool

# Configure unified debug logger (shared with attention/scheduler)
_attn_logger = logging.getLogger("baby_sgl.attn")
if not _attn_logger.handlers:
    _attn_logger.setLevel(logging.DEBUG)
    os.makedirs("baby-sgl/debug", exist_ok=True)
    fh = logging.FileHandler("baby-sgl/debug/attn_debug.log")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    _attn_logger.addHandler(fh)


def _fb_log_tensor_stats(name: str, t: torch.Tensor):
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
        _attn_logger.warning(f"_fb_log_tensor_stats error for {name}: {e}")


@dataclass
class SimplifiedForwardBatch:
    batch_size: int
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool
    is_prefill: bool = True

    # Prefix cache metadata
    prefix_lens: torch.Tensor = None
    extended_lens: torch.Tensor = None

    @classmethod
    def create_prefill_batch(
        cls,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
        prefix_lens: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        extended_lens = (seq_lens - prefix_lens).to(
            dtype=torch.long, device=input_ids.device
        )

        positions = []
        for i in range(batch_size):
            pl = int(prefix_lens[i].item())
            nl = int(extended_lens[i].item())
            positions.extend(range(pl, pl + nl))
        positions = torch.tensor(positions, device=input_ids.device, dtype=torch.long)

        # Debug: forward_batch prefill construction
        _attn_logger.debug(f"create_prefill_batch: batch_size={batch_size}")
        _fb_log_tensor_stats("prefill.input_ids", input_ids)
        _fb_log_tensor_stats("prefill.req_pool_indices", req_pool_indices)
        _fb_log_tensor_stats("prefill.seq_lens", seq_lens)
        _fb_log_tensor_stats("prefill.prefix_lens", prefix_lens)
        _fb_log_tensor_stats("prefill.extended_lens", extended_lens)
        try:
            pos_cursor = 0
            for i in range(batch_size):
                pl = int(prefix_lens[i].item())
                nl = int(extended_lens[i].item())
                pos_slice = positions[pos_cursor : pos_cursor + nl]
                expected = torch.arange(
                    pl, pl + nl, device=positions.device, dtype=positions.dtype
                )
                if nl > 0 and not torch.equal(pos_slice, expected):
                    _attn_logger.warning(
                        f"create_prefill_batch positions mismatch for i={i}: got={pos_slice.tolist()} expected={expected.tolist()}"
                    )
                pos_cursor += nl
            _fb_log_tensor_stats("prefill.positions", positions)
        except Exception as e:
            _attn_logger.warning(f"create_prefill_batch positions debug error: {e}")

        return cls(
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            is_prefill=True,
            prefix_lens=prefix_lens,
            extended_lens=extended_lens,
        )

    @classmethod
    def create_decode_batch(
        cls,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
    ):
        batch_size = len(req_pool_indices)
        positions = seq_lens - 1  # logic index

        # Debug: forward_batch decode construction
        _attn_logger.debug(f"create_decode_batch: batch_size={batch_size}")
        _fb_log_tensor_stats("decode.input_ids", input_ids)
        _fb_log_tensor_stats("decode.req_pool_indices", req_pool_indices)
        _fb_log_tensor_stats("decode.seq_lens", seq_lens)
        _fb_log_tensor_stats("decode.positions", positions)

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
