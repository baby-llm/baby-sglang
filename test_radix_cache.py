import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from memory_pool import ReqToTokenPool, MHATokenToKVPool
from radix_cache import RadixCache, TreeNode


# A lightweight request used only for testing RadixCache APIs
@dataclass
class DummyRequest:
    input_ids: torch.Tensor
    output_ids: List[int]
    max_new_tokens: int = 16
    eos_id: int = -1

    finished: bool = False
    req_pool_idx: Optional[int] = None
    seq_len: int = 0

    # Prefix cache fields expected by RadixCache
    prefix_indices: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32)
    )
    last_node: Optional[TreeNode] = None

    def reset(self):
        self.output_ids = []
        self.req_pool_idx = None
        self.seq_len = 0
        self.finished = False
        self.prefix_indices = torch.tensor([], dtype=torch.int32)
        self.last_node = None


def make_pools(req_pool_size=8, token_pool_size=1024, device="cpu"):
    # req_to_token_pool: maps requests to KV indices
    req_to_token_pool = ReqToTokenPool(
        size=req_pool_size,
        max_context_len=token_pool_size // max(1, req_pool_size),
        device=device,
    )
    # kv pool: simulates physical KV storage
    kv_pool = MHATokenToKVPool(
        size=token_pool_size,
        dtype=torch.float16,
        head_num=2,
        head_dim=4,
        layer_num=2,
        device=device,
    )
    return req_to_token_pool, kv_pool


def test_match_and_insert_basic():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    # Insert [1,2,3] -> [10,11,12]
    inserted = cache.insert([1, 2, 3], torch.tensor([10, 11, 12], dtype=torch.int32))
    assert inserted == 0  # no existing prefix

    # Exact match
    matched, node = cache.match_prefix([1, 2, 3])
    assert node is not None
    assert matched.dtype == torch.int32
    assert matched.tolist() == [10, 11, 12]

    # Partial match triggers split at prefix_len=2
    matched2, node2 = cache.match_prefix([1, 2])
    assert matched2.tolist() == [10, 11]
    assert isinstance(node2, TreeNode)


def test_insert_overlapping_sequences():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    cache.insert([1, 2, 3], torch.tensor([10, 11, 12], dtype=torch.int32))
    # Overlapping sequence shares [1,2]
    old_len = cache.insert(
        [1, 2, 4, 5], torch.tensor([10, 11, 20, 21], dtype=torch.int32)
    )
    assert old_len == 2

    # Check evictable_size accounts for leaf insertions
    # First insert added 3; second added 2 (for the new leaf [4,5])
    assert cache.evictable_size() == 5


def test_lock_ref_changes_evictable_size():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    cache.insert([1, 2, 3], torch.tensor([10, 11, 12], dtype=torch.int32))
    cache.insert([1, 2, 4, 5], torch.tensor([10, 11, 20, 21], dtype=torch.int32))
    init_evictable = cache.evictable_size()
    assert init_evictable == 5

    # Lock the exact node for [1,2,3]
    _, node = cache.match_prefix([1, 2, 3])
    assert isinstance(node, TreeNode)

    delta_dec = cache.inc_lock_ref(node)
    # Locking nodes should reduce evictable size by the locked path value lengths
    assert delta_dec < 0
    assert cache.evictable_size() == init_evictable + delta_dec

    # Unlock should restore evictable size
    delta_inc = cache.dec_lock_ref(node)
    assert delta_inc == -delta_dec
    assert cache.evictable_size() == init_evictable


def test_evict_lru_behavior():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    # Insert several leaves
    cache.insert([1, 2, 3], torch.tensor([10, 11, 12], dtype=torch.int32))
    cache.insert([4, 5], torch.tensor([40, 41], dtype=torch.int32))
    cache.insert([6], torch.tensor([60], dtype=torch.int32))

    # Touch [1,2,3] and [4,5] to set access times; leave [6] oldest (LRU)
    cache.match_prefix([1, 2, 3])
    time.sleep(0.001)
    cache.match_prefix([4, 5])

    freed_indices = []

    def evict_cb(t: torch.Tensor):
        freed_indices.append(t.clone())

    # Evict 1 token worth; should evict the LRU leaf first ([6] -> length 1)
    cache.evict(1, evict_cb)
    assert len(freed_indices) >= 1
    # First eviction must have at least size 1
    assert freed_indices[0].numel() >= 1
    # The evictable size should decrease
    assert cache.evictable_size() < 6  # original total leaf lengths were 3 + 2 + 1 = 6


def _prepare_req_with_mapping(
    cache: RadixCache, input_ids: List[int], output_ids: List[int]
) -> DummyRequest:
    req = DummyRequest(
        input_ids=torch.tensor(input_ids, dtype=torch.long), output_ids=list(output_ids)
    )
    # Allocate request index
    indices = cache.req_to_token_pool.alloc(1)
    assert indices is not None
    req.req_pool_idx = indices[0]

    # Build full sequence kv mapping for the current tokens
    full_seq = input_ids + output_ids
    needed = len(full_seq)
    kv_slots = cache.token_to_kv_pool.alloc(needed)
    assert kv_slots is not None

    # Write mapping to req_to_token_pool
    mapping = torch.zeros(
        (1, cache.req_to_token_pool.max_context_len), dtype=torch.int32
    )
    mapping[0, :needed] = kv_slots
    cache.req_to_token_pool.write(req.req_pool_idx, mapping[0])

    # Record current seq_len
    req.seq_len = needed

    # Prime prefix cache state
    req.prefix_indices, req.last_node = cache.match_prefix(full_seq)

    return req


def test_cache_finished_req_with_overlap():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    # First finished request caches [1,2,3]
    req1 = _prepare_req_with_mapping(cache, [1, 2, 3], [])
    # Simulate prefill-finished: output_ids contains next token, but we exclude last in caching
    req1.output_ids = [4]
    cache.cache_finished_req(req1)
    # Request slot freed
    assert req1.req_pool_idx is not None  # request object still has value
    # But pool should have this index back available
    assert (
        cache.req_to_token_pool.available_size() == cache.req_to_token_pool.size - 0
    )  # index recycled internally

    # Second finished request shares prefix [1,2,3] then adds [5]
    req2 = _prepare_req_with_mapping(cache, [1, 2, 3], [5])
    # Make sure prefix_indices reflects cache hit before caching finished
    req2.prefix_indices, req2.last_node = cache.match_prefix([1, 2, 3, 5])
    prev_free = kv_pool.available_size()
    cache.cache_finished_req(req2)
    # Some KV slots freed (duplicates of prefix part)
    assert kv_pool.available_size() >= prev_free


def test_cache_unfinished_req_updates_prefix():
    req_pool, kv_pool = make_pools()
    cache = RadixCache(req_pool, kv_pool)

    # Prepare request that will continue decoding
    req = _prepare_req_with_mapping(cache, [10, 11, 12], [13, 14])
    before_evictable = cache.evictable_size()

    # Cache the unfinished request (full sequence so far)
    prev_free = kv_pool.available_size()
    cache.cache_unfinished_req(req)

    # Prefix indices should cover full sequence so far
    assert isinstance(req.prefix_indices, torch.Tensor)
    assert req.prefix_indices.numel() == len([10, 11, 12, 13, 14])
    assert req.last_node is not None

    # KV duplicates for the cached prefix should be freed (if any)
    assert kv_pool.available_size() >= prev_free

    # last_node was locked by cache_unfinished_req; unlocking later should change evictable size
    delta = cache.dec_lock_ref(req.last_node)
    assert delta >= 0
    assert cache.evictable_size() == before_evictable + delta


def run_all_tests():
    test_match_and_insert_basic()
    test_insert_overlapping_sequences()
    test_lock_ref_changes_evictable_size()
    test_evict_lru_behavior()
    test_cache_finished_req_with_overlap()
    test_cache_unfinished_req_updates_prefix()
    print("All RadixCache tests passed.")


if __name__ == "__main__":
    run_all_tests()
