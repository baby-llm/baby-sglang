"""Test fixtures and utilities for baby-sglang tests."""

import torch
from typing import List, Dict, Any
from mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from managers.batch_scheduler import SimplifiedSamplingParams, SimplifiedRequest
from model_executor.qwen2 import SimplifiedForwardBatch

class TestConfig:
    """Test configuration for small-scale testing."""
    # Small model config for testing
    VOCAB_SIZE = 1000
    HIDDEN_SIZE = 128
    INTERMEDIATE_SIZE = 512
    NUM_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 32
    MAX_CONTEXT_LEN = 64
    
    # Memory pool configs
    REQ_POOL_SIZE = 8
    TOKEN_POOL_SIZE = 256
    
    # Test data
    TEST_PROMPTS = [
        "Hello world",
        "The quick brown fox",
        "In a galaxy far far away",
        "Once upon a time",
    ]

def create_test_memory_pools(device: str = "cpu") -> tuple:
    """Create small test memory pools."""
    req_pool = ReqToTokenPool(
        size=TestConfig.REQ_POOL_SIZE,
        max_context_len=TestConfig.MAX_CONTEXT_LEN,
        device=device,
    )
    
    token_pool = MHATokenToKVPool(
        size=TestConfig.TOKEN_POOL_SIZE,
        dtype=torch.float32,
        head_num=TestConfig.NUM_KV_HEADS,
        head_dim=TestConfig.HEAD_DIM,
        layer_num=TestConfig.NUM_LAYERS,
        device=device,
    )
    
    return req_pool, token_pool

def create_test_requests(num_requests: int = 2) -> List[SimplifiedRequest]:
    """Create test requests with simple token sequences."""
    requests = []
    
    for i in range(num_requests):
        # Simple test token sequences (mock tokenized)
        input_ids = list(range(i * 5 + 1, i * 5 + 6))  # [1,2,3,4,5], [6,7,8,9,10], etc
        
        req = SimplifiedRequest(
            request_id=f"test_req_{i}",
            input_text=f"Test prompt {i}",
            input_ids=input_ids,
            sampling_params=SimplifiedSamplingParams(
                max_new_tokens=3,
                temperature=1.0,
            ),
            eos_token_id=None,  # No EOS check for unit tests
        )
        requests.append(req)
    
    return requests

def create_test_forward_batch(
    input_ids: List[int],
    seq_lens: List[int],
    is_prefill: bool = True,
    device: str = "cpu"
) -> SimplifiedForwardBatch:
    """Create a test forward batch."""
    req_pool, token_pool = create_test_memory_pools(device)
    
    # Mock allocation for testing
    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)
    
    req_pool_indices = list(range(batch_size))
    out_cache_loc = list(range(1, total_tokens + 1))  # Start from 1 (slot 0 reserved)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
    req_pool_indices_tensor = torch.tensor(req_pool_indices, device=device, dtype=torch.long)
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
    out_cache_loc_tensor = torch.tensor(out_cache_loc, device=device, dtype=torch.long)
    
    if is_prefill:
        return SimplifiedForwardBatch.create_prefill_batch(
            input_ids=input_ids_tensor,
            req_pool_indices=req_pool_indices_tensor,
            seq_lens=seq_lens_tensor,
            out_cache_loc=out_cache_loc_tensor,
            req_to_token_pool=req_pool,
            token_to_kv_pool=token_pool,
        )
    else:
        return SimplifiedForwardBatch.create_decode_batch(
            input_ids=input_ids_tensor,
            req_pool_indices=req_pool_indices_tensor,
            seq_lens=seq_lens_tensor,
            out_cache_loc=out_cache_loc_tensor,
            req_to_token_pool=req_pool,
            token_to_kv_pool=token_pool,
        )

def assert_memory_pool_invariants(req_pool: ReqToTokenPool, token_pool: MHATokenToKVPool):
    """Assert memory pool invariants hold."""
    # Check that free slots don't overlap with size limits
    assert len(req_pool.free_slots) <= req_pool.size
    assert len(token_pool.free_slots) <= token_pool.size
    
    # Check that all free slots are valid indices
    for slot in req_pool.free_slots:
        assert 0 <= slot < req_pool.size
        
    for slot in token_pool.free_slots:
        assert 1 <= slot <= token_pool.size  # Token slots start from 1

def assert_no_cross_request_leakage(batch: SimplifiedForwardBatch):
    """Assert that requests can only access their own tokens."""
    # This is a simplified check - in real implementation we'd need more sophisticated validation
    for i, seq_len in enumerate(batch.seq_lens):
        req_idx = batch.req_pool_indices[i]
        # Check that this request's token indices are within expected range
        token_indices = batch.req_to_token_pool.req_to_token[req_idx, :seq_len]
        assert torch.all(token_indices > 0), f"Request {i} has invalid token indices"
        assert torch.all(token_indices <= batch.token_to_kv_pool.size), f"Request {i} token indices out of bounds"