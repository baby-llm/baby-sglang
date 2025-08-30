"""
Unit tests for Static Scheduler components.

Tests cover:
1. SimplifiedRequest - Request lifecycle and state management
2. StaticBatchScheduler - Batch scheduling and memory management  
3. Request status transitions (WAITING -> PREFILL -> DECODE -> FINISHED)
4. Memory allocation coordination
5. Batch construction for prefill and decode phases
"""

import unittest
import torch
from typing import List

# Import components under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers.batch_scheduler import (
    SimplifiedRequest, SimplifiedSamplingParams, RequestStatus, StaticBatchScheduler
)
from model_executor.qwen2 import SimplifiedForwardBatch
from tests.fixtures import TestConfig, create_test_memory_pools, create_test_requests


class TestSimplifiedRequest(unittest.TestCase):
    """Test SimplifiedRequest functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_params = SimplifiedSamplingParams(
            max_new_tokens=5,
            temperature=1.0,
            top_p=0.9
        )
        
        self.request = SimplifiedRequest(
            request_id="test_req_001",
            input_text="Hello world",
            input_ids=[1, 2, 3, 4, 5],
            sampling_params=self.sampling_params,
            eos_token_id=None
        )
    
    def test_initialization(self):
        """Test proper initialization of SimplifiedRequest."""
        self.assertEqual(self.request.request_id, "test_req_001")
        self.assertEqual(self.request.input_text, "Hello world")
        self.assertEqual(self.request.input_ids, [1, 2, 3, 4, 5])
        self.assertEqual(self.request.prompt_len, 5)
        self.assertEqual(self.request.seq_len, 5)  # Initially same as prompt_len
        
        # Initial state
        self.assertEqual(self.request.status, RequestStatus.WAITING)
        self.assertFalse(self.request.finished)
        self.assertFalse(self.request.aborted)
        self.assertEqual(self.request.output_ids, [])
        
        # Memory management fields
        self.assertIsNone(self.request.req_pool_idx)
        self.assertIsNone(self.request.allocated_tokens)
    
    def test_sampling_params(self):
        """Test sampling parameters functionality."""
        # Test greedy sampling detection
        greedy_params = SimplifiedSamplingParams(temperature=0.0)
        self.assertTrue(greedy_params.is_greedy)
        
        non_greedy_params = SimplifiedSamplingParams(temperature=1.0)
        self.assertFalse(non_greedy_params.is_greedy)
    
    def test_request_readiness_states(self):
        """Test request readiness state methods."""
        # Initially should be ready for prefill
        self.assertTrue(self.request.is_prefill_ready)
        self.assertFalse(self.request.is_decode_ready)
        
        # After changing to DECODE status
        self.request.status = RequestStatus.DECODE
        self.assertFalse(self.request.is_prefill_ready)
        self.assertTrue(self.request.is_decode_ready)
        
        # After finishing
        self.request.status = RequestStatus.FINISHED
        self.assertFalse(self.request.is_prefill_ready)
        self.assertFalse(self.request.is_decode_ready)
    
    def test_output_token_generation(self):
        """Test adding output tokens and state updates."""
        # Initially needs generation
        self.assertTrue(self.request.needs_generation)
        
        # Add some tokens
        for i in range(3):
            self.request.add_output_token(100 + i)
            self.assertEqual(len(self.request.output_ids), i + 1)
            self.assertEqual(self.request.seq_len, self.request.prompt_len + i + 1)
        
        # Still needs generation (haven't reached max)
        self.assertTrue(self.request.needs_generation)
        
        # Add remaining tokens to reach max
        for i in range(2):  # 3 + 2 = 5 (max_new_tokens)
            self.request.add_output_token(200 + i)
        
        # Should be finished now
        self.assertFalse(self.request.needs_generation)
        self.assertTrue(self.request.finished)
        self.assertEqual(self.request.status, RequestStatus.FINISHED)
    
    def test_get_all_tokens(self):
        """Test getting concatenated input + output tokens."""
        # Initially only input tokens
        all_tokens = self.request.get_all_token_ids()
        self.assertEqual(all_tokens, self.request.input_ids)
        
        # After adding output tokens
        self.request.add_output_token(10)
        self.request.add_output_token(20)
        
        all_tokens = self.request.get_all_token_ids()
        expected = self.request.input_ids + [10, 20]
        self.assertEqual(all_tokens, expected)


class TestStaticBatchScheduler(unittest.TestCase):
    """Test StaticBatchScheduler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.req_pool, self.token_pool = create_test_memory_pools(self.device)
        
        self.scheduler = StaticBatchScheduler(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool=self.token_pool,
            max_batch_size=4,
            device=self.device,
        )
    
    def test_initialization(self):
        """Test proper initialization of StaticBatchScheduler."""
        self.assertEqual(self.scheduler.max_batch_size, 4)
        self.assertEqual(self.scheduler.device, self.device)
        
        # Initially empty
        self.assertEqual(len(self.scheduler.waiting_requests), 0)
        self.assertEqual(len(self.scheduler.running_requests), 0)
        self.assertEqual(len(self.scheduler.finished_requests), 0)
        self.assertEqual(len(self.scheduler.request_map), 0)
    
    def test_add_requests(self):
        """Test adding requests to scheduler."""
        requests = create_test_requests(2)
        
        # Add requests successfully
        for req in requests:
            success = self.scheduler.add_request(req)
            self.assertTrue(success)
        
        self.assertEqual(len(self.scheduler.waiting_requests), 2)
        self.assertEqual(len(self.scheduler.request_map), 2)
        
        # Verify requests are in request_map
        for req in requests:
            self.assertIn(req.request_id, self.scheduler.request_map)
            self.assertEqual(self.scheduler.request_map[req.request_id], req)
    
    def test_add_requests_batch_size_limit(self):
        """Test batch size limit enforcement."""
        # Create more requests than max batch size
        requests = create_test_requests(6)  # max_batch_size is 4
        
        # First 4 should succeed
        for i in range(4):
            success = self.scheduler.add_request(requests[i])
            self.assertTrue(success)
        
        # Next 2 should fail
        for i in range(4, 6):
            success = self.scheduler.add_request(requests[i])
            self.assertFalse(success)
        
        self.assertEqual(len(self.scheduler.waiting_requests), 4)
    
    def test_schedule_prefill_batch_success(self):
        """Test successful prefill batch scheduling."""
        # Add test requests
        requests = create_test_requests(2)
        for req in requests:
            self.scheduler.add_request(req)
        
        # Schedule prefill batch
        forward_batch = self.scheduler.schedule_prefill_batch()
        
        # Should succeed
        self.assertIsNotNone(forward_batch)
        self.assertIsInstance(forward_batch, SimplifiedForwardBatch)
        
        # Check batch properties
        self.assertEqual(forward_batch.batch_size, 2)
        self.assertTrue(forward_batch.is_prefill)
        
        # Check request state transitions
        self.assertEqual(len(self.scheduler.waiting_requests), 0)
        self.assertEqual(len(self.scheduler.running_requests), 2)
        
        for req in self.scheduler.running_requests:
            self.assertEqual(req.status, RequestStatus.PREFILL)
            self.assertIsNotNone(req.req_pool_idx)
            self.assertIsNotNone(req.allocated_tokens)
    
    def test_schedule_prefill_batch_empty(self):
        """Test prefill batch scheduling with no waiting requests."""
        # No requests added
        forward_batch = self.scheduler.schedule_prefill_batch()
        self.assertIsNone(forward_batch)
    
    def test_schedule_decode_batch_success(self):
        """Test successful decode batch scheduling."""
        # Set up requests in DECODE state
        requests = create_test_requests(2)
        for req in requests:
            self.scheduler.add_request(req)
        
        # Schedule prefill first to get requests into running state
        prefill_batch = self.scheduler.schedule_prefill_batch()
        self.assertIsNotNone(prefill_batch)
        
        # Change status to DECODE (simulate completion of prefill)
        for req in self.scheduler.running_requests:
            req.status = RequestStatus.DECODE
            req.add_output_token(99)  # Add one token to simulate prefill output
        
        # Now schedule decode batch
        decode_batch = self.scheduler.schedule_decode_batch()
        
        self.assertIsNotNone(decode_batch)
        self.assertFalse(decode_batch.is_prefill)
        self.assertEqual(decode_batch.batch_size, 2)
        
        # In decode mode, input_ids should be single tokens
        self.assertEqual(decode_batch.input_ids.shape[0], 2)  # One token per request
    
    def test_update_requests_with_tokens(self):
        """Test updating requests with generated tokens."""
        requests = create_test_requests(2)
        new_tokens = [100, 200]
        
        initial_lens = [req.seq_len for req in requests]
        
        # Update with tokens
        self.scheduler.update_requests_with_tokens(requests, new_tokens)
        
        # Check updates
        for i, req in enumerate(requests):
            self.assertEqual(req.output_ids[-1], new_tokens[i])
            self.assertEqual(req.seq_len, initial_lens[i] + 1)
    
    def test_cleanup_finished_requests(self):
        """Test cleanup of finished requests."""
        # Add and schedule requests
        requests = create_test_requests(2)
        for req in requests:
            self.scheduler.add_request(req)
        
        prefill_batch = self.scheduler.schedule_prefill_batch()
        self.assertIsNotNone(prefill_batch)
        
        # Mark one request as finished
        running_req = self.scheduler.running_requests[0]
        running_req.finished = True
        running_req.status = RequestStatus.FINISHED
        
        initial_running = len(self.scheduler.running_requests)
        initial_finished = len(self.scheduler.finished_requests)
        
        # Cleanup
        self.scheduler.cleanup_finished_requests()
        
        # Check cleanup results
        self.assertEqual(len(self.scheduler.running_requests), initial_running - 1)
        self.assertEqual(len(self.scheduler.finished_requests), initial_finished + 1)
        
        # Verify memory was freed (basic check)
        stats = self.scheduler.get_stats()
        self.assertGreater(stats["req_pool_available"], 0)
        self.assertGreater(stats["token_pool_available"], 0)
    
    def test_scheduler_stats(self):
        """Test scheduler statistics reporting."""
        # Initial stats
        stats = self.scheduler.get_stats()
        expected_keys = ["waiting", "running", "finished", "req_pool_available", "token_pool_available"]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Initially should have full availability
        self.assertEqual(stats["waiting"], 0)
        self.assertEqual(stats["running"], 0) 
        self.assertEqual(stats["finished"], 0)
        self.assertEqual(stats["req_pool_available"], TestConfig.REQ_POOL_SIZE)
        self.assertEqual(stats["token_pool_available"], TestConfig.TOKEN_POOL_SIZE)
        
        # After adding requests
        requests = create_test_requests(2)
        for req in requests:
            self.scheduler.add_request(req)
        
        stats = self.scheduler.get_stats()
        self.assertEqual(stats["waiting"], 2)
        
        # After scheduling
        self.scheduler.schedule_prefill_batch()
        stats = self.scheduler.get_stats()
        self.assertEqual(stats["waiting"], 0)
        self.assertEqual(stats["running"], 2)
        self.assertLess(stats["req_pool_available"], TestConfig.REQ_POOL_SIZE)
        self.assertLess(stats["token_pool_available"], TestConfig.TOKEN_POOL_SIZE)


class TestBatchConstruction(unittest.TestCase):
    """Test batch construction for prefill and decode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.req_pool, self.token_pool = create_test_memory_pools(self.device)
        self.scheduler = StaticBatchScheduler(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool=self.token_pool,
            max_batch_size=4,
            device=self.device,
        )
    
    def test_prefill_batch_construction(self):
        """Test proper construction of prefill batches."""
        # Create requests with different sequence lengths
        request1 = SimplifiedRequest("req1", "Hello", [1, 2, 3], SimplifiedSamplingParams(), eos_token_id=None)
        request2 = SimplifiedRequest("req2", "World", [4, 5], SimplifiedSamplingParams(), eos_token_id=None)
        
        self.scheduler.add_request(request1)
        self.scheduler.add_request(request2)
        
        forward_batch = self.scheduler.schedule_prefill_batch()
        
        # Check batch structure
        self.assertEqual(forward_batch.batch_size, 2)
        self.assertTrue(forward_batch.is_prefill)
        
        # Input IDs should be flattened: [1, 2, 3, 4, 5]
        expected_input_ids = torch.tensor([1, 2, 3, 4, 5], device=self.device)
        torch.testing.assert_close(forward_batch.input_ids, expected_input_ids)
        
        # Sequence lengths should be preserved
        expected_seq_lens = torch.tensor([3, 2], device=self.device)
        torch.testing.assert_close(forward_batch.seq_lens, expected_seq_lens)
        
        # Positions should be computed correctly
        expected_positions = torch.tensor([0, 1, 2, 0, 1], device=self.device)
        torch.testing.assert_close(forward_batch.positions, expected_positions)
        
        # Memory pools should be properly referenced
        self.assertIs(forward_batch.req_to_token_pool, self.req_pool)
        self.assertIs(forward_batch.token_to_kv_pool, self.token_pool)
    
    def test_decode_batch_construction(self):
        """Test proper construction of decode batches."""
        # Set up requests in decode state
        requests = create_test_requests(2)
        for req in requests:
            self.scheduler.add_request(req)
        
        # Schedule prefill and transition to decode
        self.scheduler.schedule_prefill_batch()
        
        for req in self.scheduler.running_requests:
            req.status = RequestStatus.DECODE
            req.add_output_token(99)  # Simulate generated token
        
        decode_batch = self.scheduler.schedule_decode_batch()
        
        # Check decode batch properties
        self.assertEqual(decode_batch.batch_size, 2)
        self.assertFalse(decode_batch.is_prefill)
        
        # Input should be single tokens per request
        self.assertEqual(decode_batch.input_ids.shape[0], 2)
        
        # Positions should be current sequence positions
        for i, req in enumerate(self.scheduler.running_requests):
            expected_pos = req.seq_len
            actual_pos = decode_batch.positions[i].item()
            self.assertEqual(actual_pos, expected_pos)


if __name__ == '__main__':
    unittest.main()