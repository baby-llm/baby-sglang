# Baby-SGLang Testing Makefile
# ===========================
# 
# Quick commands for running tests and development tasks
# Usage: make <target>
#
# Main test targets:
#   test-all       - Run all test suites
#   test-unit      - Run unit tests only
#   test-int       - Run integration tests only  
#   test-perf      - Run performance tests only
#   test-security  - Run security tests only
#   benchmark      - Run performance benchmarks
#
# Development targets:
#   setup-dev      - Set up development environment
#   lint           - Run code linting
#   format         - Format code with black and isort
#   coverage       - Generate test coverage report
#   clean          - Clean up temporary files

.PHONY: help test-all test-unit test-int test-perf test-security benchmark setup-dev lint format coverage clean

# Default target
help:
	@echo "Baby-SGLang Testing Makefile"
	@echo "============================"
	@echo ""
	@echo "Test targets:"
	@echo "  test-all       Run all test suites"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo "  test-perf      Run performance tests only"
	@echo "  test-security  Run security tests only"
	@echo "  benchmark      Run performance benchmarks"
	@echo ""
	@echo "Development targets:"
	@echo "  setup-dev      Set up development environment"
	@echo "  lint           Run code linting"
	@echo "  format         Format code with black and isort"
	@echo "  coverage       Generate test coverage report"
	@echo "  clean          Clean up temporary files"
	@echo ""
	@echo "Device-specific tests:"
	@echo "  test-cpu       Run tests on CPU"
	@echo "  test-cuda      Run tests on CUDA (if available)"
	@echo "  test-mps       Run tests on Apple MPS (if available)"

# Test targets
test-all:
	@echo "Running all test suites..."
	cd tests && python run_tests.py --all --verbose

test-unit:
	@echo "Running unit tests..."
	cd tests && python run_tests.py --unit --verbose

test-int:
	@echo "Running integration tests..."
	cd tests && python run_tests.py --integration --verbose

test-perf:
	@echo "Running performance tests..."
	cd tests && python run_tests.py --performance --verbose

test-security:
	@echo "Running security tests..."
	cd tests && python run_tests.py --security --verbose

benchmark:
	@echo "Running performance benchmarks..."
	cd tests && python run_tests.py --benchmark

# Device-specific tests
test-cpu:
	@echo "Running tests on CPU..."
	cd tests && python run_tests.py --all --device cpu --verbose

test-cuda:
	@echo "Running tests on CUDA..."
	@python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || (echo "CUDA not available" && exit 1)
	cd tests && python run_tests.py --unit --integration --device cuda --verbose

test-mps:
	@echo "Running tests on Apple MPS..."
	@python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" || (echo "MPS not available, falling back to CPU" && make test-cpu)
	cd tests && python run_tests.py --unit --integration --device mps --verbose

# Development environment setup
setup-dev:
	@echo "Setting up development environment..."
	pip install --upgrade pip
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install tokenizers transformers safetensors
	pip install psutil coverage pytest pytest-cov pytest-timeout pytest-xdist
	pip install black isort flake8 mypy
	@echo "Development environment ready!"

# Code quality targets
lint:
	@echo "Running linting checks..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Code formatting complete!"

format-check:
	@echo "Checking code format..."
	black --check --diff .
	isort --check-only --diff .

# Coverage reporting
coverage:
	@echo "Generating test coverage report..."
	cd tests && python run_tests.py --coverage
	@echo "Coverage report generated in htmlcov/"

coverage-report:
	@echo "Displaying coverage report..."
	cd tests && coverage report

coverage-html:
	@echo "Generating HTML coverage report..."
	cd tests && coverage html
	@echo "HTML report available in htmlcov/index.html"

# Quick test commands for development
quick-test:
	@echo "Running quick unit tests..."
	cd tests && python run_tests.py --unit

smoke-test:
	@echo "Running smoke test (basic functionality)..."
	cd tests && python run_tests.py --unit --integration --device cpu

# Continuous integration simulation
ci-test:
	@echo "Running CI test suite..."
	make lint
	make test-all
	make coverage

# Performance profiling
profile:
	@echo "Running performance profiling..."
	cd tests && python -m cProfile -o profile_output.prof run_tests.py --performance
	@echo "Profile saved to tests/profile_output.prof"

# Memory profiling (requires memory_profiler)
memory-profile:
	@echo "Running memory profiling..."
	@pip show memory_profiler > /dev/null || pip install memory_profiler
	cd tests && python -m memory_profiler run_tests.py --performance

# Stress testing
stress-test:
	@echo "Running stress tests..."
	cd tests && for i in {1..5}; do \
		echo "Stress test iteration $$i..."; \
		python run_tests.py --all --device cpu || exit 1; \
	done
	@echo "Stress test completed successfully!"

# Cleanup targets
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.prof" -delete
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Deep cleaning (including build artifacts)..."
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "Documentation generation not yet implemented"

# Installation targets  
install-dev: setup-dev
	@echo "Installing in development mode..."
	pip install -e .

# Docker targets (if Docker is available)
docker-test:
	@echo "Running tests in Docker container..."
	@command -v docker >/dev/null 2>&1 || (echo "Docker not available" && exit 1)
	docker build -t baby-sglang-test .
	docker run --rm baby-sglang-test make test-all

# Release preparation
pre-release:
	@echo "Preparing for release..."
	make clean
	make format
	make lint
	make test-all
	make coverage
	@echo "Pre-release checks complete!"

# Show system information
info:
	@echo "System Information"
	@echo "=================="
	@echo "Python version:"
	@python --version
	@echo "PyTorch version:"
	@python -c "import torch; print(torch.__version__)"
	@echo "CUDA available:"
	@python -c "import torch; print(torch.cuda.is_available())"
	@echo "MPS available:"
	@python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "false"
	@echo "Device count:"
	@python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "0"