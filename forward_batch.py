from dataclasses import dataclass
from typing import Any
import torch

from memory_pool import ReqToTokenPool, MHATokenToKVPool


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
