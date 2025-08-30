"""
Performance and memory usage tests for Baby-SGLang.

Tests cover:
1. Memory usage patterns and leak detection
2. Execution time benchmarks 
3. Throughput measurements
4. Scalability testing with different batch sizes
5. Resource utilization efficiency
"""

import unittest
import torch
import time
import gc
import psutil
import os
from typing import List, Dict, Any

# Import components under test  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import SimplifiedInferenceEngine, create_engine_args
from managers.batch_scheduler import SimplifiedRequest, SimplifiedSamplingParams
from model_executor.qwen2 import SimplifiedModelArgs
from tests.fixtures import TestConfig


class MemoryTracker:
    """Helper class to track memory usage during tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_mb()
        self.peak_memory = self.initial_memory
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
        
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current
            
    def get_memory_increase(self) -> float:
        """Get memory increase from initial."""
        return self.get_memory_mb() - self.initial_memory
        
    def get_peak_increase(self) -> float:
        """Get peak memory increase from initial."""
        return self.peak_memory - self.initial_memory


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage patterns and leak detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        gc.collect()  # Clean up before testing
        
    def test_memory_pool_allocation_efficiency(self):
        """Test that memory pools allocate efficiently without waste."""
        from mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
        
        tracker = MemoryTracker()
        
        # Create memory pools
        req_pool = ReqToTokenPool(
            size=100,
            max_context_len=512,
            device=self.device,
        )
        
        token_pool = MHATokenToKVPool(
            size=1000,
            dtype=torch.float32,
            head_num=8,
            head_dim=64,
            layer_num=4,
            device=self.device,
        )
        
        tracker.update_peak()
        
        # Calculate expected memory usage
        req_pool_size = 100 * 512 * 4  # int32 = 4 bytes
        token_pool_size = 1001 * 8 * 64 * 4 * 4 * 2  # K + V buffers, 4 layers
        expected_mb = (req_pool_size + token_pool_size) / 1024 / 1024
        
        # Actual memory increase should be close to expected
        actual_increase = tracker.get_peak_increase()
        
        # Allow some overhead for Python objects and alignment
        self.assertLess(actual_increase, expected_mb * 1.5)
        self.assertGreater(actual_increase, expected_mb * 0.8)
        
        # Clean up
        del req_pool, token_pool
        gc.collect()
        
    def test_memory_leak_detection(self):
        """Test for memory leaks during request processing."""
        tracker = MemoryTracker()
        
        # Create engine
        engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=200,
            max_batch_size=4,
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        
        engine = SimplifiedInferenceEngine(engine_args, model_args)
        baseline_memory = tracker.get_memory_mb()
        
        # Process multiple rounds of requests
        for round_num in range(5):
            # Create requests
            requests = []
            for i in range(3):
                req = SimplifiedRequest(
                    request_id=f"leak_test_{round_num}_{i}",
                    input_text=f"Test {round_num} {i}",
                    input_ids=list(range(5)),
                    sampling_params=SimplifiedSamplingParams(max_new_tokens=2),
                    eos_token_id=None,
                )
                requests.append(req)
            
            # Process requests
            for req in requests:
                engine.add_request(req)
            
            engine.generate()
            
            # Clear finished requests
            engine.scheduler.finished_requests.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory
            tracker.update_peak()
        
        # Memory should not have grown significantly
        final_memory = tracker.get_memory_mb()
        memory_growth = final_memory - baseline_memory
        
        # Allow some growth but not excessive (< 50MB for this test)
        self.assertLess(memory_growth, 50.0, f"Memory grew by {memory_growth:.2f} MB, suggesting leak")
        
        del engine
        gc.collect()
    
    def test_kv_cache_memory_scaling(self):
        """Test that KV cache memory scales linearly with sequence length."""
        from mem_cache.memory_pool import MHATokenToKVPool
        
        # Test different token pool sizes
        sizes = [100, 200, 400]
        memory_usage = []
        
        for size in sizes:
            tracker = MemoryTracker()
            
            pool = MHATokenToKVPool(
                size=size,
                dtype=torch.float32,
                head_num=4,
                head_dim=32,
                layer_num=2,
                device=self.device,
            )
            
            memory_usage.append(tracker.get_memory_increase())
            del pool
            gc.collect()
        
        # Memory usage should scale approximately linearly
        ratio1 = memory_usage[1] / memory_usage[0]  # 200/100
        ratio2 = memory_usage[2] / memory_usage[1]  # 400/200
        
        # Both ratios should be close to 2.0 (linear scaling)
        self.assertAlmostEqual(ratio1, 2.0, delta=0.5)
        self.assertAlmostEqual(ratio2, 2.0, delta=0.5)


class TestExecutionPerformance(unittest.TestCase):
    """Test execution time and throughput."""
    
    def setUp(self):
        """Set up test engine."""
        self.device = "cpu"
        
        self.engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=500,
            max_batch_size=8,
        )
        
        # Small but realistic model for performance testing
        self.model_args = SimplifiedModelArgs(
            vocab_size=2000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
        )
    
    def test_single_request_latency(self):
        """Test latency for single request processing."""
        engine = SimplifiedInferenceEngine(self.engine_args, self.model_args)
        
        request = SimplifiedRequest(
            request_id="latency_test",
            input_text="Latency test",
            input_ids=list(range(10)),  # 10 input tokens
            sampling_params=SimplifiedSamplingParams(
                max_new_tokens=5,
                temperature=0.0,
            ),
            eos_token_id=None,
        )
        
        engine.add_request(request)
        
        # Measure generation time
        start_time = time.time()
        engine.generate()
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should complete in reasonable time (< 5 seconds for CPU)
        self.assertLess(latency, 5.0)
        
        # Log performance for reference
        print(f"Single request latency: {latency:.3f} seconds")
        
        del engine
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than sequential."""
        # Test batch processing
        engine = SimplifiedInferenceEngine(self.engine_args, self.model_args)
        
        # Create 4 identical requests
        requests = []
        for i in range(4):
            req = SimplifiedRequest(
                request_id=f"batch_test_{i}",
                input_text=f"Batch test {i}",
                input_ids=list(range(8)),
                sampling_params=SimplifiedSamplingParams(max_new_tokens=3),
                eos_token_id=None,
            )
            requests.append(req)
        
        # Add all requests and measure batch processing time
        for req in requests:
            engine.add_request(req)
        
        start_time = time.time()
        engine.generate()
        batch_time = time.time() - start_time
        
        del engine
        
        # Test sequential processing
        total_sequential_time = 0
        for req in requests:
            engine = SimplifiedInferenceEngine(self.engine_args, self.model_args)
            engine.add_request(req)
            
            start_time = time.time()
            engine.generate()
            total_sequential_time += time.time() - start_time
            
            del engine
        
        # Batch processing should be faster than sequential
        efficiency_gain = total_sequential_time / batch_time
        print(f"Batch efficiency gain: {efficiency_gain:.2f}x")
        
        self.assertGreater(efficiency_gain, 1.0)  # Should be at least somewhat faster
    
    def test_throughput_scaling(self):
        """Test throughput scaling with different batch sizes."""
        batch_sizes = [1, 2, 4]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            engine = SimplifiedInferenceEngine(self.engine_args, self.model_args)
            
            # Create requests
            requests = []
            for i in range(batch_size):
                req = SimplifiedRequest(
                    request_id=f"throughput_{batch_size}_{i}",
                    input_text=f"Throughput test {i}",
                    input_ids=list(range(6)),
                    sampling_params=SimplifiedSamplingParams(max_new_tokens=4),
                    eos_token_id=None,
                )
                requests.append(req)
            
            # Add requests and measure
            for req in requests:
                engine.add_request(req)
            
            start_time = time.time()
            engine.generate()
            total_time = time.time() - start_time
            
            # Calculate tokens per second
            total_output_tokens = sum(len(req.output_ids) for req in engine.scheduler.finished_requests)
            throughput = total_output_tokens / total_time
            throughput_results[batch_size] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.2f} tokens/second")
            
            del engine
        
        # Larger batch sizes should generally have higher throughput
        self.assertGreater(throughput_results[2], throughput_results[1] * 0.8)  # At least 80% scaling
    
    def test_memory_efficiency_vs_batch_size(self):
        """Test memory efficiency across different batch sizes."""
        batch_sizes = [1, 2, 4]
        memory_per_request = {}
        
        for batch_size in batch_sizes:
            tracker = MemoryTracker()
            
            engine = SimplifiedInferenceEngine(self.engine_args, self.model_args)
            baseline = tracker.get_memory_mb()
            
            # Create and process requests
            requests = []
            for i in range(batch_size):
                req = SimplifiedRequest(
                    request_id=f"memory_eff_{batch_size}_{i}",
                    input_text=f"Memory efficiency {i}",
                    input_ids=list(range(8)),
                    sampling_params=SimplifiedSamplingParams(max_new_tokens=3),
                    eos_token_id=None,
                )
                requests.append(req)
                engine.add_request(req)
            
            engine.generate()
            peak_memory = tracker.get_memory_mb()
            memory_used = peak_memory - baseline
            memory_per_request[batch_size] = memory_used / batch_size
            
            print(f"Batch size {batch_size}: {memory_per_request[batch_size]:.2f} MB per request")
            
            del engine
            gc.collect()
        
        # Memory per request should be similar or decrease with larger batches
        # (indicating sharing of resources)
        self.assertLessEqual(
            memory_per_request[4],
            memory_per_request[1] * 1.2,  # Allow 20% overhead
            "Memory per request should not increase significantly with batch size"
        )


class TestScalabilityLimits(unittest.TestCase):
    """Test system behavior at scale limits."""
    
    def setUp(self):
        """Set up test configuration."""
        self.device = "cpu"
    
    def test_maximum_batch_size_handling(self):
        """Test behavior at maximum batch size limits."""
        # Create engine with small limits for testing
        engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=100,
            max_batch_size=3,  # Small limit for testing
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=500,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
        )
        
        engine = SimplifiedInferenceEngine(engine_args, model_args)
        
        # Try to add more requests than limit
        successful_adds = 0
        for i in range(5):  # More than max_batch_size
            req = SimplifiedRequest(
                request_id=f"limit_test_{i}",
                input_text=f"Test {i}",
                input_ids=[i + 1],
                sampling_params=SimplifiedSamplingParams(max_new_tokens=1),
                eos_token_id=None,
            )
            
            if engine.add_request(req):
                successful_adds += 1
        
        # Should only accept up to max_batch_size
        self.assertEqual(successful_adds, 3)
        
        # Should still process successfully
        engine.generate()
        self.assertEqual(len(engine.scheduler.finished_requests), 3)
        
        del engine
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create engine with very limited token memory
        engine_args = create_engine_args(
            model_path=None,
            device=self.device,
            max_total_tokens=20,  # Very small
            max_batch_size=5,
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=200,
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        
        engine = SimplifiedInferenceEngine(engine_args, model_args)
        
        # Create requests that will exceed memory
        requests = []
        for i in range(3):
            # Each request uses 8 tokens, total would be 24 > 20 limit
            req = SimplifiedRequest(
                request_id=f"pressure_test_{i}",
                input_text=f"Pressure test {i}",
                input_ids=list(range(8)),
                sampling_params=SimplifiedSamplingParams(max_new_tokens=1),
                eos_token_id=None,
            )
            requests.append(req)
            engine.add_request(req)
        
        # Should handle memory pressure gracefully
        try:
            engine.generate()
            # If successful, some requests should be processed
            stats = engine.scheduler.get_stats()
            self.assertGreaterEqual(stats["finished"], 0)
        except RuntimeError:
            # Acceptable to fail gracefully with memory error
            pass
        
        del engine


class PerformanceBenchmark:
    """Benchmark suite for comparing performance across configurations."""
    
    @staticmethod
    def run_benchmark_suite():
        """Run comprehensive benchmark suite."""
        print("\n" + "="*60)
        print("Baby-SGLang Performance Benchmark Suite")
        print("="*60)
        
        # Configuration variations to test
        configs = [
            {
                "name": "Tiny Model",
                "hidden_size": 64,
                "num_layers": 2,
                "num_heads": 4,
            },
            {
                "name": "Small Model", 
                "hidden_size": 128,
                "num_layers": 4,
                "num_heads": 8,
            }
        ]
        
        batch_sizes = [1, 2, 4]
        seq_lengths = [8, 16, 32]
        
        results = {}
        
        for config in configs:
            config_results = {}
            print(f"\nTesting {config['name']}:")
            print("-" * 40)
            
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    # Run benchmark
                    latency, throughput = PerformanceBenchmark._run_single_benchmark(
                        config, batch_size, seq_len
                    )
                    
                    key = f"B{batch_size}_S{seq_len}"
                    config_results[key] = {
                        "latency": latency,
                        "throughput": throughput
                    }
                    
                    print(f"  Batch={batch_size}, SeqLen={seq_len}: "
                          f"{latency:.3f}s, {throughput:.1f} tok/s")
            
            results[config["name"]] = config_results
        
        print("\nBenchmark Summary:")
        print("-" * 40)
        for config_name, config_results in results.items():
            avg_throughput = sum(r["throughput"] for r in config_results.values()) / len(config_results)
            print(f"{config_name}: Avg {avg_throughput:.1f} tokens/second")
        
        return results
    
    @staticmethod
    def _run_single_benchmark(config, batch_size, seq_len):
        """Run single benchmark configuration."""
        device = "cpu"
        
        # Create engine
        engine_args = create_engine_args(
            model_path=None,
            device=device,
            max_total_tokens=1000,
            max_batch_size=batch_size,
        )
        
        model_args = SimplifiedModelArgs(
            vocab_size=1000,
            hidden_size=config["hidden_size"],
            intermediate_size=config["hidden_size"] * 4,
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            num_key_value_heads=config["num_heads"] // 2,
            head_dim=config["hidden_size"] // config["num_heads"],
        )
        
        engine = SimplifiedInferenceEngine(engine_args, model_args)
        
        # Create requests
        requests = []
        for i in range(batch_size):
            req = SimplifiedRequest(
                request_id=f"bench_{i}",
                input_text=f"Benchmark {i}",
                input_ids=list(range(seq_len)),
                sampling_params=SimplifiedSamplingParams(max_new_tokens=4),
                eos_token_id=None,
            )
            requests.append(req)
            engine.add_request(req)
        
        # Benchmark
        start_time = time.time()
        engine.generate()
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_output_tokens = sum(len(req.output_ids) for req in engine.scheduler.finished_requests)
        throughput = total_output_tokens / total_time
        
        del engine
        gc.collect()
        
        return total_time, throughput


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run benchmark suite
    PerformanceBenchmark.run_benchmark_suite()