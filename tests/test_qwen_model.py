"""
Unit tests for Qwen2 Model components.

Tests cover:
1. SimplifiedForwardBatch - Batch construction and data integrity
2. Model components - RMSNorm, RoPE, QKVLinear, MLP
3. Attention mechanisms - Prefill and decode attention with SGLang security
4. Full model forward pass functionality
5. Cross-request token leakage prevention
"""

import unittest
import torch
import torch.nn as nn
import math
from typing import List, Tuple

# Import components under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_executor.qwen2 import (
    SimplifiedForwardBatch, RMSNorm, RotaryEmbedding, 
    QKVLinear, MLP, QwenAttention, QwenDecoderLayer, QwenModel,
    SimpleAttentionBackend, SimplifiedModelArgs
)
from tests.fixtures import TestConfig, create_test_memory_pools, create_test_forward_batch, assert_no_cross_request_leakage


class TestSimplifiedForwardBatch(unittest.TestCase):
    """Test SimplifiedForwardBatch functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
    
    def test_prefill_batch_creation(self):
        """Test creating prefill batches."""
        input_ids = [1, 2, 3, 4, 5, 6]  # Two sequences: [1,2,3] and [4,5,6]
        seq_lens = [3, 3]
        
        batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device=self.device)
        
        # Check basic properties
        self.assertEqual(batch.batch_size, 2)
        self.assertTrue(batch.is_prefill)
        self.assertEqual(batch.input_ids.shape[0], 6)
        
        # Check positions are computed correctly
        expected_positions = torch.tensor([0, 1, 2, 0, 1, 2], device=self.device)
        torch.testing.assert_close(batch.positions, expected_positions)
        
        # Check sequence lengths
        expected_seq_lens = torch.tensor(seq_lens, device=self.device)
        torch.testing.assert_close(batch.seq_lens, expected_seq_lens)
    
    def test_decode_batch_creation(self):
        """Test creating decode batches."""
        input_ids = [10, 20]  # Single token per request
        seq_lens = [5, 7]     # Current sequence lengths
        
        batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=False, device=self.device)
        
        # Check decode properties
        self.assertEqual(batch.batch_size, 2)
        self.assertFalse(batch.is_prefill)
        self.assertEqual(batch.input_ids.shape[0], 2)  # One token per request
        
        # Check positions (should be current sequence lengths for new tokens)
        expected_positions = torch.tensor(seq_lens, device=self.device)
        torch.testing.assert_close(batch.positions, expected_positions)


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.hidden_size = TestConfig.HIDDEN_SIZE
        self.vocab_size = TestConfig.VOCAB_SIZE
        self.batch_size = 2
        self.seq_len = 4
    
    def test_rms_norm(self):
        """Test RMSNorm functionality."""
        norm = RMSNorm(self.hidden_size)
        
        # Test forward pass
        input_tensor = torch.randn(self.seq_len, self.hidden_size, device=self.device)
        output = norm(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Test with residual
        residual = torch.randn_like(input_tensor)
        output_with_residual = norm(input_tensor, residual)
        
        # Should include residual connection
        self.assertEqual(output_with_residual.shape, input_tensor.shape)
    
    def test_rotary_embedding(self):
        """Test RotaryEmbedding functionality."""
        rope = RotaryEmbedding(
            head_dim=TestConfig.HEAD_DIM,
            max_position_embeddings=TestConfig.MAX_CONTEXT_LEN
        )
        
        # Test forward pass
        query = torch.randn(self.seq_len, TestConfig.NUM_ATTENTION_HEADS, TestConfig.HEAD_DIM, device=self.device)
        key = torch.randn(self.seq_len, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM, device=self.device)
        positions = torch.arange(self.seq_len, device=self.device)
        
        q_rot, k_rot = rope(query, key, positions)
        
        # Check output shapes
        self.assertEqual(q_rot.shape, query.shape)
        self.assertEqual(k_rot.shape, key.shape)
        
        # Check that rotation actually changes the values (not identity)
        self.assertFalse(torch.allclose(q_rot, query))
        self.assertFalse(torch.allclose(k_rot, key))
    
    def test_qkv_linear(self):
        """Test QKVLinear merged projection."""
        qkv_proj = QKVLinear(
            input_size=self.hidden_size,
            num_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_kv_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM
        )
        
        # Test forward pass
        hidden_states = torch.randn(self.seq_len, self.hidden_size, device=self.device)
        query, key, value = qkv_proj(hidden_states)
        
        # Check output shapes
        expected_q_shape = (self.seq_len, TestConfig.NUM_ATTENTION_HEADS, TestConfig.HEAD_DIM)
        expected_kv_shape = (self.seq_len, TestConfig.NUM_KV_HEADS, TestConfig.HEAD_DIM)
        
        self.assertEqual(query.shape, expected_q_shape)
        self.assertEqual(key.shape, expected_kv_shape)
        self.assertEqual(value.shape, expected_kv_shape)
    
    def test_mlp(self):
        """Test MLP functionality."""
        mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=TestConfig.INTERMEDIATE_SIZE
        )
        
        # Test forward pass
        hidden_states = torch.randn(self.seq_len, self.hidden_size, device=self.device)
        output = mlp(hidden_states)
        
        self.assertEqual(output.shape, hidden_states.shape)


class TestAttentionMechanism(unittest.TestCase):
    """Test attention mechanisms and security."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.req_pool, self.token_pool = create_test_memory_pools(self.device)
        
        # Create test attention layer
        self.attention = QwenAttention(
            layer_id=0,
            hidden_size=TestConfig.HIDDEN_SIZE,
            num_attention_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_key_value_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
            max_position_embeddings=TestConfig.MAX_CONTEXT_LEN,
        )
    
    def test_prefill_attention_security(self):
        """Test that prefill attention prevents cross-request token leakage."""
        # Create batch with two different requests
        input_ids = [1, 2, 3, 10, 11]  # First req: [1,2,3], Second req: [10,11]
        seq_lens = [3, 2]
        
        batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device=self.device)
        
        # Create hidden states (mock embeddings)
        hidden_states = torch.randn(5, TestConfig.HIDDEN_SIZE, device=self.device)
        
        # Run attention
        with torch.no_grad():
            output = self.attention(hidden_states, batch.positions, batch)
        
        # Check output shape
        expected_shape = (5, TestConfig.HIDDEN_SIZE)
        self.assertEqual(output.shape, expected_shape)
        
        # Security check: ensure no cross-request leakage
        assert_no_cross_request_leakage(batch)
    
    def test_decode_attention_security(self):
        """Test that decode attention prevents cross-request token leakage."""
        # Set up decode scenario: each request has one new token
        input_ids = [99, 88]  # New tokens for two requests
        seq_lens = [4, 3]     # Current sequence lengths
        
        # First simulate prefill to populate KV cache
        prefill_input_ids = [1, 2, 3, 4, 10, 11, 12]
        prefill_seq_lens = [4, 3]
        prefill_batch = create_test_forward_batch(prefill_input_ids, prefill_seq_lens, is_prefill=True, device=self.device)
        
        # Mock KV cache population
        prefill_hidden = torch.randn(7, TestConfig.HIDDEN_SIZE, device=self.device)
        with torch.no_grad():
            _ = self.attention(prefill_hidden, prefill_batch.positions, prefill_batch)
        
        # Now test decode
        decode_batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=False, device=self.device)
        decode_hidden = torch.randn(2, TestConfig.HIDDEN_SIZE, device=self.device)
        
        with torch.no_grad():
            output = self.attention(decode_hidden, decode_batch.positions, decode_batch)
        
        # Check output shape (one token per request)
        expected_shape = (2, TestConfig.HIDDEN_SIZE)
        self.assertEqual(output.shape, expected_shape)
        
        # Security check
        assert_no_cross_request_leakage(decode_batch)
    
    def test_attention_backend_per_request_processing(self):
        """Test that SimpleAttentionBackend processes requests separately."""
        # Create test data
        seq_lens = [3, 2]
        total_tokens = sum(seq_lens)
        
        # Mock Q, K, V tensors
        num_heads = TestConfig.NUM_KV_HEADS  # Simplified: use KV heads for all
        head_dim = TestConfig.HEAD_DIM
        
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device)
        k = torch.randn(total_tokens, num_heads, head_dim, device=self.device) 
        v = torch.randn(total_tokens, num_heads, head_dim, device=self.device)
        
        # Create forward batch
        input_ids = list(range(total_tokens))
        batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device=self.device)
        
        # Test per-request prefill attention
        with torch.no_grad():
            output = SimpleAttentionBackend._prefill_attention_per_request(
                q, k, v, 1.0 / math.sqrt(head_dim), batch
            )
        
        # Check output shape matches input
        expected_shape = (total_tokens, num_heads * head_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # The output should be different from input (attention computation applied)
        self.assertFalse(torch.allclose(output, q.flatten(-2)))


class TestFullModelForward(unittest.TestCase):
    """Test full model forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        
        # Create small model for testing
        self.model_args = SimplifiedModelArgs(
            vocab_size=TestConfig.VOCAB_SIZE,
            hidden_size=TestConfig.HIDDEN_SIZE,
            intermediate_size=TestConfig.INTERMEDIATE_SIZE,
            num_hidden_layers=TestConfig.NUM_LAYERS,
            num_attention_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_key_value_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
            max_position_embeddings=TestConfig.MAX_CONTEXT_LEN,
        )
        
        self.model = QwenModel(self.model_args, device=self.device)
    
    def test_prefill_forward_pass(self):
        """Test full model prefill forward pass."""
        # Create test batch
        input_ids = [1, 2, 3, 10, 11, 12]  # Two requests
        seq_lens = [3, 3]
        
        batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device=self.device)
        
        # Run forward pass
        with torch.no_grad():
            logits = self.model(
                input_ids=batch.input_ids,
                positions=batch.positions,
                forward_batch=batch
            )
        
        # Check output shape: [total_tokens, vocab_size]
        expected_shape = (6, TestConfig.VOCAB_SIZE)
        self.assertEqual(logits.shape, expected_shape)
        
        # Check that logits are reasonable (not NaN, not all zeros)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.allclose(logits, torch.zeros_like(logits)))
        
        # Security check
        assert_no_cross_request_leakage(batch)
    
    def test_decode_forward_pass(self):
        """Test full model decode forward pass."""
        # First run prefill to populate KV cache
        prefill_input_ids = [1, 2, 3, 10, 11]
        prefill_seq_lens = [3, 2]
        prefill_batch = create_test_forward_batch(prefill_input_ids, prefill_seq_lens, is_prefill=True, device=self.device)
        
        with torch.no_grad():
            _ = self.model(
                input_ids=prefill_batch.input_ids,
                positions=prefill_batch.positions,
                forward_batch=prefill_batch
            )
        
        # Now test decode
        decode_input_ids = [99, 88]  # One new token per request
        decode_seq_lens = [3, 2]     # Current lengths
        decode_batch = create_test_forward_batch(decode_input_ids, decode_seq_lens, is_prefill=False, device=self.device)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=decode_batch.input_ids,
                positions=decode_batch.positions,
                forward_batch=decode_batch
            )
        
        # Check output shape: [batch_size, vocab_size] for decode
        expected_shape = (2, TestConfig.VOCAB_SIZE)
        self.assertEqual(logits.shape, expected_shape)
        
        # Quality checks
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.allclose(logits, torch.zeros_like(logits)))
        
        # Security check
        assert_no_cross_request_leakage(decode_batch)
    
    def test_consistency_across_devices(self):
        """Test model consistency across different devices (if available)."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create identical batches for CPU and CUDA
        input_ids = [1, 2, 3]
        seq_lens = [3]
        
        cpu_batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device="cpu")
        cuda_batch = create_test_forward_batch(input_ids, seq_lens, is_prefill=True, device="cuda")
        
        # Create models on different devices
        cpu_model = QwenModel(self.model_args, device="cpu")
        cuda_model = QwenModel(self.model_args, device="cuda")
        
        # Copy weights to ensure they're identical
        cuda_model.load_state_dict(cpu_model.state_dict())
        
        # Run forward passes
        with torch.no_grad():
            cpu_logits = cpu_model(
                input_ids=cpu_batch.input_ids,
                positions=cpu_batch.positions,
                forward_batch=cpu_batch
            )
            
            cuda_logits = cuda_model(
                input_ids=cuda_batch.input_ids,
                positions=cuda_batch.positions,
                forward_batch=cuda_batch
            ).cpu()
        
        # Results should be approximately equal
        torch.testing.assert_close(cpu_logits, cuda_logits, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()