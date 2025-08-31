"""
Memory pool for baby-sglang.

SGLang has two levels of memory pool:
- ReqToTokenPool maps a request to its token locations  
- BaseTokenToKVPool maps a token location to its KV cache data

This implementation follows SGLang's exact design with flat storage,
NOT paged attention. SGLang uses token-level indirection, not block-level paging.

Based on sglang/srt/mem_cache/memory_pool.py
"""

import logging
from typing import List, Tuple, Union
import torch

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        
        # Following SGLang exactly: [batch_size, max_context_len] -> token_indices
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        """Write token indices to request slots."""
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        """Allocate request slots."""
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: Union[int, List[int]]):
        """Free request slots."""
        if isinstance(free_index, int):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # SGLang's float8 handling
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.free_slots = None
        self.clear()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        """Allocate token slots."""
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index.to(self.device, non_blocking=True) 
    
    def free(self, free_index: torch.Tensor):
        """Free token slots."""
        if free_index.numel() == 0:
            return
        self.free_slots = torch.concat((self.free_slots, free_index.cpu())) 

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        # Following SGLang's exact convention
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int32)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()


class MHATokenToKVPool(BaseTokenToKVPool):
    """
    SGLang's exact MHATokenToKVPool implementation.
    
    Uses FLAT storage: [size, head_num, head_dim] per layer.
    No paging, just token-level indirection.
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, dtype, device)

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()

        # Log memory usage like SGLang
        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. K size: {k_size / (1024**3):.2f} GB, "
            f"V size: {v_size / (1024**3):.2f} GB."
        )

    def _create_buffers(self):
        """Create flat KV buffers following SGLang's exact layout."""
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def get_kv_size_bytes(self):
        """Calculate memory usage."""
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += k_cache.numel() * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += v_cache.numel() * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    def get_key_buffer(self, layer_id: int):
        """Get key buffer for layer."""
        if self.store_dtype != self.dtype:
            raise NotImplementedError("float8 KV cache read is not supported in baby-sglang MVP")
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        """Get value buffer for layer."""
        if self.store_dtype != self.dtype:
            raise NotImplementedError("float8 KV cache read is not supported in baby-sglang MVP")
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        """Get K,V buffers for layer."""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """Set KV cache data at token locations."""
        # Ensure index dtype is correct for advanced indexing
        if loc.dtype != torch.long:
            loc = loc.long()

        # Handle dtype conversion like SGLang
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            raise NotImplementedError("float8 KV cache write is not supported in baby-sglang MVP")
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v