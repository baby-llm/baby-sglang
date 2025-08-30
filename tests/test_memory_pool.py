"""
Unit tests for Memory Pool components.

Tests cover:
1. ReqToTokenPool - Request to token mapping functionality
2. MHATokenToKVPool - KV cache storage and retrieval
3. Memory allocation and deallocation correctness
4. SGLang compatibility and invariants
"""

import unittest
import torch
from typing import List

# Import components under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from tests.fixtures import TestConfig, create_test_memory_pools, assert_memory_pool_invariants


class TestReqToTokenPool(unittest.TestCase):
    """Test request to token pool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.req_pool = ReqToTokenPool(
            size=TestConfig.REQ_POOL_SIZE,
            max_context_len=TestConfig.MAX_CONTEXT_LEN,
            device=self.device,
        )
    
    def test_initialization(self):
        """Test proper initialization of ReqToTokenPool."""
        # Check initial state
        self.assertEqual(self.req_pool.size, TestConfig.REQ_POOL_SIZE)
        self.assertEqual(self.req_pool.max_context_len, TestConfig.MAX_CONTEXT_LEN)
        self.assertEqual(self.req_pool.device, self.device)
        
        # Check req_to_token tensor shape
        expected_shape = (TestConfig.REQ_POOL_SIZE, TestConfig.MAX_CONTEXT_LEN)
        self.assertEqual(self.req_pool.req_to_token.shape, expected_shape)
        self.assertEqual(self.req_pool.req_to_token.device.type, self.device)
        self.assertEqual(self.req_pool.req_to_token.dtype, torch.int32)
        
        # Check all slots are initially free
        self.assertEqual(len(self.req_pool.free_slots), TestConfig.REQ_POOL_SIZE)
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE)
    
    def test_allocation_success(self):
        """Test successful slot allocation."""
        # Allocate 2 slots
        allocated = self.req_pool.alloc(2)
        
        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 2)
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE - 2)
        
        # Check allocated slots are valid
        for slot in allocated:
            self.assertGreaterEqual(slot, 0)
            self.assertLess(slot, TestConfig.REQ_POOL_SIZE)
    
    def test_allocation_exhaustion(self):
        """Test allocation failure when pool is exhausted."""
        # Allocate all slots
        allocated = self.req_pool.alloc(TestConfig.REQ_POOL_SIZE)
        self.assertIsNotNone(allocated)
        self.assertEqual(self.req_pool.available_size(), 0)
        
        # Try to allocate one more - should fail
        failed_alloc = self.req_pool.alloc(1)
        self.assertIsNone(failed_alloc)
    
    def test_deallocation(self):
        """Test slot deallocation."""
        # Allocate some slots
        allocated = self.req_pool.alloc(3)
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE - 3)
        
        # Free one slot
        self.req_pool.free(allocated[0])
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE - 2)
        
        # Free multiple slots
        self.req_pool.free(allocated[1:])
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE)
    
    def test_write_operation(self):
        """Test writing token indices to request slots."""
        # Allocate slots
        allocated = self.req_pool.alloc(2)
        
        # Write token indices
        token_indices = torch.tensor([10, 20, 30], device=self.device, dtype=torch.int32)
        self.req_pool.write(allocated[0], token_indices)
        
        # Verify write
        written_tokens = self.req_pool.req_to_token[allocated[0], :len(token_indices)]
        torch.testing.assert_close(written_tokens, token_indices)
    
    def test_clear_pool(self):
        """Test clearing the pool."""
        # Allocate some slots
        self.req_pool.alloc(3)
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE - 3)
        
        # Clear pool
        self.req_pool.clear()
        self.assertEqual(self.req_pool.available_size(), TestConfig.REQ_POOL_SIZE)


class TestMHATokenToKVPool(unittest.TestCase):
    """Test MHA token-to-KV pool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.token_pool = MHATokenToKVPool(
            size=TestConfig.TOKEN_POOL_SIZE,
            dtype=torch.float32,
            head_num=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
            layer_num=TestConfig.NUM_LAYERS,
            device=self.device,
        )
    
    def test_initialization(self):
        """Test proper initialization of MHATokenToKVPool."""
        # Check basic properties
        self.assertEqual(self.token_pool.size, TestConfig.TOKEN_POOL_SIZE)
        self.assertEqual(self.token_pool.dtype, torch.float32)
        self.assertEqual(self.token_pool.head_num, TestConfig.NUM_KV_HEADS)
        self.assertEqual(self.token_pool.head_dim, TestConfig.HEAD_DIM)
        self.assertEqual(self.token_pool.layer_num, TestConfig.NUM_LAYERS)
        
        # Check buffers are created correctly
        self.assertEqual(len(self.token_pool.k_buffer), TestConfig.NUM_LAYERS)
        self.assertEqual(len(self.token_pool.v_buffer), TestConfig.NUM_LAYERS)
        
        for layer_id in range(TestConfig.NUM_LAYERS):
            k_buffer = self.token_pool.k_buffer[layer_id]
            v_buffer = self.token_pool.v_buffer[layer_id]
            
            # Check buffer shapes: [size + 1, head_num, head_dim]
            expected_shape = (TestConfig.TOKEN_POOL_SIZE + 1, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM)
            self.assertEqual(k_buffer.shape, expected_shape)
            self.assertEqual(v_buffer.shape, expected_shape)
            
            self.assertEqual(k_buffer.device.type, self.device)
            self.assertEqual(v_buffer.device.type, self.device)
        
        # Check free slots (start from 1, slot 0 reserved for padding)
        self.assertEqual(self.token_pool.available_size(), TestConfig.TOKEN_POOL_SIZE)
    
    def test_token_allocation(self):
        """Test token slot allocation."""
        # Allocate some tokens
        allocated = self.token_pool.alloc(10)
        
        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)
        self.assertEqual(self.token_pool.available_size(), TestConfig.TOKEN_POOL_SIZE - 10)
        
        # Check allocated indices are valid (should be >= 1)
        for idx in allocated:
            self.assertGreaterEqual(idx.item(), 1)
            self.assertLessEqual(idx.item(), TestConfig.TOKEN_POOL_SIZE)
    
    def test_token_deallocation(self):
        """Test token slot deallocation."""
        # Allocate tokens
        allocated = self.token_pool.alloc(5)
        self.assertEqual(self.token_pool.available_size(), TestConfig.TOKEN_POOL_SIZE - 5)
        
        # Free tokens
        self.token_pool.free(allocated)
        self.assertEqual(self.token_pool.available_size(), TestConfig.TOKEN_POOL_SIZE)
    
    def test_kv_buffer_access(self):
        """Test KV buffer getter methods."""
        for layer_id in range(TestConfig.NUM_LAYERS):
            # Test individual buffer access
            k_buffer = self.token_pool.get_key_buffer(layer_id)
            v_buffer = self.token_pool.get_value_buffer(layer_id)
            
            expected_shape = (TestConfig.TOKEN_POOL_SIZE + 1, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM)
            self.assertEqual(k_buffer.shape, expected_shape)
            self.assertEqual(v_buffer.shape, expected_shape)
            
            # Test combined buffer access
            k_buf, v_buf = self.token_pool.get_kv_buffer(layer_id)
            torch.testing.assert_close(k_buf, k_buffer)
            torch.testing.assert_close(v_buf, v_buffer)
    
    def test_kv_buffer_write(self):
        """Test writing to KV buffers."""
        # Allocate some token positions
        allocated = self.token_pool.alloc(3)
        
        # Create test KV data
        layer_id = 0
        cache_k = torch.randn(3, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM, device=self.device)
        cache_v = torch.randn(3, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM, device=self.device)
        
        # Write to cache
        self.token_pool.set_kv_buffer(layer_id, allocated, cache_k, cache_v)
        
        # Verify write
        k_buffer = self.token_pool.get_key_buffer(layer_id)
        v_buffer = self.token_pool.get_value_buffer(layer_id)
        
        retrieved_k = k_buffer[allocated]
        retrieved_v = v_buffer[allocated]
        
        torch.testing.assert_close(retrieved_k, cache_k)
        torch.testing.assert_close(retrieved_v, cache_v)
    
    def test_memory_size_calculation(self):
        """Test memory size calculation methods."""
        k_size, v_size = self.token_pool.get_kv_size_bytes()
        
        # Calculate expected size
        single_buffer_size = (TestConfig.TOKEN_POOL_SIZE + 1) * TestConfig.NUM_KV_HEADS * TestConfig.HEAD_DIM
        expected_size_per_layer = single_buffer_size * 4  # float32 = 4 bytes
        expected_total_size = expected_size_per_layer * TestConfig.NUM_LAYERS
        
        self.assertEqual(k_size, expected_total_size)
        self.assertEqual(v_size, expected_total_size)


class TestMemoryPoolIntegration(unittest.TestCase):
    """Test integration between ReqToTokenPool and MHATokenToKVPool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.req_pool, self.token_pool = create_test_memory_pools(self.device)
    
    def test_coordinated_allocation(self):
        """Test coordinated allocation between req and token pools."""
        # Simulate allocating for 2 requests with different sequence lengths
        num_requests = 2
        seq_lens = [5, 8]  # Different sequence lengths
        
        # Allocate request slots
        req_indices = self.req_pool.alloc(num_requests)
        self.assertIsNotNone(req_indices)
        
        # Allocate token slots for total tokens
        total_tokens = sum(seq_lens)
        token_indices = self.token_pool.alloc(total_tokens)
        self.assertIsNotNone(token_indices)
        
        # Map requests to tokens (SGLang pattern)
        token_offset = 0
        for i, seq_len in enumerate(seq_lens):
            req_tokens = token_indices[token_offset:token_offset + seq_len]
            self.req_pool.req_to_token[req_indices[i], :seq_len] = req_tokens
            token_offset += seq_len
        
        # Verify mappings
        for i, seq_len in enumerate(seq_lens):
            mapped_tokens = self.req_pool.req_to_token[req_indices[i], :seq_len]
            self.assertEqual(len(mapped_tokens), seq_len)
            # All tokens should be valid (>= 1)
            self.assertTrue(torch.all(mapped_tokens >= 1))
    
    def test_memory_pool_invariants(self):
        """Test that memory pool invariants hold after operations."""
        # Initial state
        assert_memory_pool_invariants(self.req_pool, self.token_pool)
        
        # After allocation
        req_slots = self.req_pool.alloc(2)
        token_slots = self.token_pool.alloc(10)
        assert_memory_pool_invariants(self.req_pool, self.token_pool)
        
        # After deallocation
        self.req_pool.free(req_slots)
        self.token_pool.free(token_slots)
        assert_memory_pool_invariants(self.req_pool, self.token_pool)
        
        # After clear
        self.req_pool.clear()
        self.token_pool.clear()
        assert_memory_pool_invariants(self.req_pool, self.token_pool)


if __name__ == '__main__':
    unittest.main()