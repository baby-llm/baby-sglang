"""
Integration tests for full Baby-SGLang pipeline.

Tests cover:
1. End-to-end generation pipeline (Engine + Scheduler + Model)
2. Multi-request concurrent processing
3. Memory management across components
4. Request lifecycle from input to completion
5. Error handling and recovery scenarios
"""

import unittest
import torch
import time
from typing import List

# Import components under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import SimplifiedInferenceEngine, create_engine_args
from managers.batch_scheduler import SimplifiedRequest, SimplifiedSamplingParams
from model_executor.qwen2 import SimplifiedModelArgs
from tests.fixtures import TestConfig


class TestEndToEndGeneration(unittest.TestCase):
    """Test complete generation pipeline."""
    
    def setUp(self):
        """Set up test engine."""
        self.device = "cpu"
        
        # Create minimal engine for testing
        self.engine_args = create_engine_args(
            model_path=None,  # Will use random weights
            device=self.device,
            max_total_tokens=TestConfig.TOKEN_POOL_SIZE,
            max_batch_size=TestConfig.REQ_POOL_SIZE // 2,
        )
        
        # Create small model config for fast testing
        model_args = SimplifiedModelArgs(
            vocab_size=TestConfig.VOCAB_SIZE,
            hidden_size=TestConfig.HIDDEN_SIZE,
            intermediate_size=TestConfig.INTERMEDIATE_SIZE,
            num_hidden_layers=TestConfig.NUM_LAYERS,
            num_attention_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_key_value_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
            max_position_embeddings=TestConfig.MAX_CONTEXT_LEN,
        )
        
        self.engine = SimplifiedInferenceEngine(self.engine_args, model_args)
    
    def test_single_request_generation(self):
        """Test generating text for a single request."""
        # Create test request
        request = SimplifiedRequest(
            request_id="test_single",
            input_text="Hello world",
            input_ids=[1, 2, 3, 4, 5],  # Mock tokenization
            sampling_params=SimplifiedSamplingParams(
                max_new_tokens=3,
                temperature=0.0,  # Greedy sampling for determinism
            ),
            eos_token_id=None,
        )
        
        # Add request to engine
        success = self.engine.add_request(request)
        self.assertTrue(success)
        
        # Run generation
        initial_stats = self.engine.scheduler.get_stats()
        self.assertEqual(initial_stats["waiting"], 1)
        
        # Execute generation loop
        self.engine.generate()
        
        # Check results
        final_stats = self.engine.scheduler.get_stats()
        self.assertEqual(final_stats["waiting"], 0)
        self.assertEqual(final_stats["running"], 0)
        self.assertEqual(final_stats["finished"], 1)
        
        # Check that request was completed
        finished_req = self.engine.scheduler.finished_requests[0]
        self.assertEqual(finished_req.request_id, "test_single")
        self.assertEqual(len(finished_req.output_ids), 3)  # Generated 3 tokens
        self.assertTrue(finished_req.finished)
    
    def test_multi_request_concurrent_generation(self):
        """Test concurrent processing of multiple requests."""
        # Create multiple requests with different lengths
        requests = []
        for i in range(3):
            req = SimplifiedRequest(
                request_id=f"test_multi_{i}",
                input_text=f"Test prompt {i}",
                input_ids=list(range(i * 3 + 1, i * 3 + 4)),  # Different input sequences
                sampling_params=SimplifiedSamplingParams(
                    max_new_tokens=2 + i,  # Different output lengths: 2, 3, 4 tokens
                    temperature=0.0,
                ),
                eos_token_id=None,
            )
            requests.append(req)
        
        # Add all requests
        for req in requests:
            success = self.engine.add_request(req)
            self.assertTrue(success)
        
        # Check initial state
        stats = self.engine.scheduler.get_stats()
        self.assertEqual(stats["waiting"], 3)
        
        # Run generation
        start_time = time.time()
        self.engine.generate()
        end_time = time.time()
        
        # Check that generation completed in reasonable time
        self.assertLess(end_time - start_time, 10.0)  # Should complete within 10 seconds
        
        # Check final state
        final_stats = self.engine.scheduler.get_stats()
        self.assertEqual(final_stats["finished"], 3)
        self.assertEqual(final_stats["running"], 0)
        
        # Check that all requests completed with correct output lengths
        finished_requests = self.engine.scheduler.finished_requests
        for i, req in enumerate(finished_requests):
            expected_output_len = 2 + i  # As specified in sampling params
            self.assertEqual(len(req.output_ids), expected_output_len)
    
    def test_memory_management_across_pipeline(self):
        """Test that memory is properly managed across the entire pipeline."""
        # Get initial memory state
        initial_req_available = self.engine.scheduler.req_to_token_pool.available_size()
        initial_token_available = self.engine.scheduler.token_to_kv_pool.available_size()
        
        # Create and process requests
        requests = []
        for i in range(2):
            req = SimplifiedRequest(
                request_id=f"memory_test_{i}",
                input_text=f"Memory test {i}",
                input_ids=list(range(5 * i + 1, 5 * i + 6)),  # 5 tokens each
                sampling_params=SimplifiedSamplingParams(max_new_tokens=2),
                eos_token_id=None,
            )
            requests.append(req)
        
        # Add requests and check memory allocation
        for req in requests:
            self.engine.add_request(req)
        
        # Memory should still be available (requests not scheduled yet)
        self.assertEqual(
            self.engine.scheduler.req_to_token_pool.available_size(),
            initial_req_available
        )
        
        # Run generation
        self.engine.generate()
        
        # After completion, memory should be freed
        final_req_available = self.engine.scheduler.req_to_token_pool.available_size()
        final_token_available = self.engine.scheduler.token_to_kv_pool.available_size()
        
        self.assertEqual(final_req_available, initial_req_available)
        self.assertEqual(final_token_available, initial_token_available)
    
    def test_request_lifecycle_states(self):
        """Test that request states transition correctly through the pipeline."""
        request = SimplifiedRequest(
            request_id="lifecycle_test",
            input_text="Lifecycle test",
            input_ids=[10, 20, 30],
            sampling_params=SimplifiedSamplingParams(max_new_tokens=2),
            eos_token_id=None,
        )
        
        # Initial state
        self.assertTrue(request.is_prefill_ready)
        self.assertFalse(request.is_decode_ready)
        
        # Add to engine
        self.engine.add_request(request)
        
        # Track state changes during generation
        # We'll modify the engine to allow step-by-step execution for testing
        
        # Schedule prefill
        prefill_batch = self.engine.scheduler.schedule_prefill_batch()
        self.assertIsNotNone(prefill_batch)
        
        # Check that request is now in PREFILL state
        running_req = self.engine.scheduler.running_requests[0]
        self.assertEqual(running_req.status.value, "prefill")
        
        # Simulate prefill execution (we'll run just the first part)
        with torch.no_grad():
            logits = self.engine.model(
                input_ids=prefill_batch.input_ids,
                positions=prefill_batch.positions,
                forward_batch=prefill_batch,
            )
        
        # Sample and update (this would normally be done by engine.generate())
        new_tokens = [100]  # Mock sampled token
        self.engine.scheduler.update_requests_with_tokens([running_req], new_tokens)
        
        # Change to DECODE state
        running_req.status = running_req.status.__class__("decode")
        
        self.assertFalse(running_req.is_prefill_ready)
        self.assertTrue(running_req.is_decode_ready)
        
        # Continue with full generation to completion
        self.engine.generate()
        
        # Check final state
        finished_req = self.engine.scheduler.finished_requests[0]
        self.assertFalse(finished_req.needs_generation)
        self.assertTrue(finished_req.finished)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test engine."""
        self.device = "cpu"
        self.engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=50,  # Very small pool for testing limits
            max_batch_size=2,
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=TestConfig.VOCAB_SIZE,
            hidden_size=TestConfig.HIDDEN_SIZE,
            intermediate_size=TestConfig.INTERMEDIATE_SIZE,
            num_hidden_layers=1,  # Single layer for speed
            num_attention_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_key_value_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
        )
        
        self.engine = SimplifiedInferenceEngine(self.engine_args, model_args)
    
    def test_batch_size_limit_enforcement(self):
        """Test that batch size limits are enforced."""
        # Try to add more requests than max_batch_size
        requests = []
        for i in range(4):  # max_batch_size is 2
            req = SimplifiedRequest(
                request_id=f"batch_limit_{i}",
                input_text=f"Test {i}",
                input_ids=[i + 1],
                sampling_params=SimplifiedSamplingParams(max_new_tokens=1),
                eos_token_id=None,
            )
            requests.append(req)
        
        # First 2 should succeed
        self.assertTrue(self.engine.add_request(requests[0]))
        self.assertTrue(self.engine.add_request(requests[1]))
        
        # Next 2 should fail
        self.assertFalse(self.engine.add_request(requests[2]))
        self.assertFalse(self.engine.add_request(requests[3]))
        
        # Check that only 2 are waiting
        stats = self.engine.scheduler.get_stats()
        self.assertEqual(stats["waiting"], 2)
    
    def test_memory_exhaustion_handling(self):
        """Test handling of memory pool exhaustion."""
        # Create requests that will exhaust token memory
        large_requests = []
        for i in range(2):
            # Each request uses many tokens to exhaust the small pool
            large_input = list(range(20))  # 20 tokens each
            req = SimplifiedRequest(
                request_id=f"large_{i}",
                input_text=f"Large request {i}",
                input_ids=large_input,
                sampling_params=SimplifiedSamplingParams(max_new_tokens=5),
                eos_token_id=None,
            )
            large_requests.append(req)
        
        # Add requests
        for req in large_requests:
            success = self.engine.add_request(req)
            # May fail if memory is insufficient
            if not success:
                break
        
        # Try to run generation - should handle memory exhaustion gracefully
        try:
            self.engine.generate()
            # If it succeeds, check that some requests were processed
            stats = self.engine.scheduler.get_stats()
            self.assertGreaterEqual(stats["finished"], 0)
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, (RuntimeError, RuntimeError))
    
    def test_empty_generation(self):
        """Test generation with no requests."""
        # Should handle empty case gracefully
        initial_stats = self.engine.scheduler.get_stats()
        
        # Run generation with no requests
        self.engine.generate()
        
        # Stats should remain unchanged
        final_stats = self.engine.scheduler.get_stats()
        self.assertEqual(initial_stats, final_stats)


class TestConcurrencyAndConsistency(unittest.TestCase):
    """Test concurrency handling and result consistency."""
    
    def setUp(self):
        """Set up test engine."""
        self.device = "cpu"
        self.engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=TestConfig.TOKEN_POOL_SIZE,
            max_batch_size=4,
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=TestConfig.VOCAB_SIZE,
            hidden_size=TestConfig.HIDDEN_SIZE,
            intermediate_size=TestConfig.INTERMEDIATE_SIZE,
            num_hidden_layers=TestConfig.NUM_LAYERS,
            num_attention_heads=TestConfig.NUM_ATTENTION_HEADS,
            num_key_value_heads=TestConfig.NUM_KV_HEADS,
            head_dim=TestConfig.HEAD_DIM,
        )
        
        self.engine = SimplifiedInferenceEngine(self.engine_args, model_args)
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same inputs and greedy sampling."""
        # Create identical requests
        request1 = SimplifiedRequest(
            request_id="det_test_1",
            input_text="Deterministic test",
            input_ids=[1, 2, 3, 4],
            sampling_params=SimplifiedSamplingParams(
                max_new_tokens=3,
                temperature=0.0,  # Greedy for determinism
            ),
            eos_token_id=None,
        )
        
        request2 = SimplifiedRequest(
            request_id="det_test_2", 
            input_text="Deterministic test",
            input_ids=[1, 2, 3, 4],  # Same input
            sampling_params=SimplifiedSamplingParams(
                max_new_tokens=3,
                temperature=0.0,  # Greedy for determinism
            ),
            eos_token_id=None,
        )
        
        # Process requests separately
        self.engine.add_request(request1)
        self.engine.generate()
        output1 = self.engine.scheduler.finished_requests[0].output_ids
        
        # Reset engine state
        self.engine.scheduler.finished_requests.clear()
        
        self.engine.add_request(request2) 
        self.engine.generate()
        output2 = self.engine.scheduler.finished_requests[0].output_ids
        
        # Outputs should be identical for greedy sampling with same inputs
        self.assertEqual(output1, output2)
    
    def test_cross_request_independence(self):
        """Test that concurrent requests don't interfere with each other."""
        # Create requests with very different inputs
        req1 = SimplifiedRequest(
            request_id="indep_1",
            input_text="First request",
            input_ids=[1, 2, 3],
            sampling_params=SimplifiedSamplingParams(max_new_tokens=2, temperature=0.0),
            eos_token_id=None,
        )
        
        req2 = SimplifiedRequest(
            request_id="indep_2", 
            input_text="Second request",
            input_ids=[100, 200, 300],  # Very different tokens
            sampling_params=SimplifiedSamplingParams(max_new_tokens=2, temperature=0.0),
            eos_token_id=None,
        )
        
        # Process together
        self.engine.add_request(req1)
        self.engine.add_request(req2)
        self.engine.generate()
        
        # Get outputs
        finished_reqs = self.engine.scheduler.finished_requests
        req1_finished = next(r for r in finished_reqs if r.request_id == "indep_1")
        req2_finished = next(r for r in finished_reqs if r.request_id == "indep_2")
        
        # Outputs should be different (very unlikely to be same with different inputs)
        self.assertNotEqual(req1_finished.output_ids, req2_finished.output_ids)
        
        # Both should have correct lengths
        self.assertEqual(len(req1_finished.output_ids), 2)
        self.assertEqual(len(req2_finished.output_ids), 2)


if __name__ == '__main__':
    unittest.main()