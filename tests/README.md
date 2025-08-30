# Baby-SGLang Testing Framework

Comprehensive testing suite for Baby-SGLang, a simplified implementation of SGLang's core inference engine.

## Overview

This testing framework provides thorough validation of all Baby-SGLang components:

- **Unit Tests**: Individual component testing (memory pools, scheduler, model components)
- **Integration Tests**: End-to-end pipeline testing with real generation workflows
- **Performance Tests**: Benchmarking, memory usage, and scalability testing
- **Security Tests**: Cross-request isolation and attention leakage prevention

## Quick Start

### Prerequisites

```bash
# Install core dependencies
pip install torch tokenizers transformers safetensors psutil

# Install testing dependencies  
pip install coverage pytest pytest-cov
```

### Running Tests

#### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/run_tests.py --all --verbose

# Run specific test suites
python tests/run_tests.py --unit
python tests/run_tests.py --integration  
python tests/run_tests.py --performance
python tests/run_tests.py --security

# Run with coverage report
python tests/run_tests.py --all --coverage

# Run benchmarks
python tests/run_tests.py --benchmark

# Test on specific devices
python tests/run_tests.py --all --device cuda    # CUDA GPU
python tests/run_tests.py --all --device mps     # Apple MPS  
python tests/run_tests.py --all --device cpu     # CPU only
```

#### Using Makefile (Alternative)

```bash
# Quick commands
make test-all         # Run all test suites
make test-unit        # Unit tests only
make test-int         # Integration tests only
make benchmark        # Performance benchmarks
make coverage         # Generate coverage report

# Device-specific
make test-cpu         # CPU tests
make test-cuda        # CUDA tests (if available)
make test-mps         # Apple MPS tests (if available)

# Development
make setup-dev        # Set up dev environment
make lint             # Run linting
make format           # Format code
make clean            # Clean temp files
```

## Test Categories

### 1. Unit Tests (`test_*.py`)

Individual component validation:

- **Memory Pool Tests** (`test_memory_pool.py`)
  - ReqToTokenPool allocation/deallocation
  - MHATokenToKVPool KV cache management
  - Memory pool invariants and SGLang compatibility

- **Scheduler Tests** (`test_scheduler.py`)
  - Request lifecycle (WAITING → PREFILL → DECODE → FINISHED)
  - Batch construction for prefill/decode phases
  - Memory allocation coordination
  - Error handling and cleanup

- **Model Tests** (`test_qwen_model.py`)
  - Individual component functionality (RMSNorm, RoPE, Attention)
  - Forward batch creation and data integrity
  - Attention security (cross-request isolation)
  - Full model forward pass validation

### 2. Integration Tests (`test_integration.py`)

End-to-end pipeline testing:

- **Single/Multi-request Generation**: Complete generation workflows
- **Memory Management**: Resource allocation across full pipeline
- **Request Lifecycle**: State transitions through entire process
- **Error Handling**: Graceful failure and recovery
- **Concurrency**: Cross-request independence and determinism

### 3. Performance Tests (`test_performance.py`)

System performance and scalability:

- **Memory Usage**: Leak detection, scaling patterns, efficiency
- **Execution Performance**: Latency, throughput, batch efficiency
- **Scalability**: Behavior at maximum limits
- **Benchmark Suite**: Comprehensive performance profiling

### 4. Security Tests

Cross-request isolation and safety:

- **Attention Isolation**: Prevents token leakage between requests
- **Memory Boundaries**: Ensures requests can't access each other's data
- **SGLang Security Model**: Validates per-request processing approach

## Test Configuration

Configuration is managed through `test_config.toml`:

```toml
[test]
verbosity = 1
timeout_per_test = 30

[test.performance] 
benchmark_iterations = 3
memory_limit = "4GB"

[devices]
cpu = true
cuda = true  
mps = true

[benchmark]
batch_sizes = [1, 2, 4, 8]
sequence_lengths = [8, 16, 32, 64]
```

## Continuous Integration

GitHub Actions CI/CD pipeline (`.github/workflows/ci.yml`):

- **Multi-Platform**: Ubuntu, macOS
- **Multi-Python**: 3.8, 3.9, 3.10, 3.11
- **Multi-Device**: CPU, CUDA, Apple MPS
- **Coverage Reports**: Automated coverage tracking
- **Nightly Benchmarks**: Performance regression detection

## Performance Benchmarks

Built-in benchmark suite tests various configurations:

```bash
# Run comprehensive benchmarks
python tests/run_tests.py --benchmark

# Results include:
# - Latency measurements
# - Throughput (tokens/second)
# - Memory usage per request
# - Scalability across batch sizes
```

Example output:
```
Baby-SGLang Performance Benchmark Suite
========================================

Testing Tiny Model:
  Batch=1, SeqLen=8: 0.045s, 89.2 tok/s
  Batch=2, SeqLen=8: 0.067s, 119.4 tok/s
  Batch=4, SeqLen=8: 0.098s, 163.2 tok/s

Testing Small Model:
  Batch=1, SeqLen=8: 0.078s, 51.3 tok/s
  Batch=2, SeqLen=8: 0.112s, 71.4 tok/s
  Batch=4, SeqLen=8: 0.163s, 98.2 tok/s
```

## Development Workflow

1. **Set up environment**:
   ```bash
   make setup-dev
   ```

2. **Run tests during development**:
   ```bash
   make quick-test      # Fast unit tests
   make smoke-test      # Basic functionality check
   ```

3. **Before committing**:
   ```bash
   make format          # Format code
   make lint            # Check code quality  
   make test-all        # Run full test suite
   make coverage        # Generate coverage report
   ```

4. **Pre-release checks**:
   ```bash
   make pre-release     # Complete validation
   ```

## Test Fixtures and Utilities

### TestConfig (`tests/fixtures.py`)

Centralized test configuration:
- Small model dimensions for fast testing
- Mock data generation utilities
- Memory pool creation helpers
- Assertion utilities for SGLang compatibility

### Memory Tracking

Built-in memory usage monitoring:
```python
from tests.fixtures import MemoryTracker

tracker = MemoryTracker()
# ... run code ...
print(f"Memory used: {tracker.get_memory_increase():.2f} MB")
```

### Security Validation

Utilities to verify cross-request isolation:
```python
from tests.fixtures import assert_no_cross_request_leakage

assert_no_cross_request_leakage(forward_batch)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the project root
2. **CUDA Tests Fail**: Check CUDA installation and availability
3. **Memory Tests Fail**: May need to adjust limits for smaller systems
4. **Slow Performance**: Expected on CPU; use smaller test configurations

### Debug Mode

Run with maximum verbosity:
```bash
python tests/run_tests.py --all --verbose --device cpu
```

### Memory Debugging

For memory leak investigation:
```bash
make memory-profile  # Requires memory_profiler
```

## Contributing

When adding new tests:

1. Follow existing test patterns and naming conventions
2. Include both positive and negative test cases  
3. Add performance considerations for expensive tests
4. Update this documentation for new test categories
5. Ensure tests pass on all supported platforms

### Test Categories Guidelines

- **Unit tests**: Test single components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Include benchmarks and resource usage
- **Security tests**: Focus on isolation and safety

## SGLang Compatibility

This test suite validates compatibility with SGLang's design:

- **Memory Management**: Exact same pool structures and allocation patterns
- **Attention Security**: Per-request processing prevents token leakage  
- **API Compatibility**: Method signatures and behavior match SGLang
- **Performance Characteristics**: Similar resource usage and scaling

## License

Part of the Baby-SGLang project. Tests are designed to validate correctness and compatibility with the SGLang inference engine architecture.