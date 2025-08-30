# Baby-SGLang Makefile
# ===================
# 
# Basic commands for development
# Usage: make <target>

.PHONY: help setup lint format clean run-demo

# Default target
help:
	@echo "Baby-SGLang Makefile"
	@echo "==================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup         Set up development environment"
	@echo "  lint          Run code linting"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean up temporary files"
	@echo "  run-demo      Run demo comparison with vLLM"

# Development environment setup
setup:
	@echo "Setting up development environment..."
	pip install --upgrade pip
	pip install torch transformers tokenizers safetensors
	pip install vllm  # For comparison
	pip install black isort flake8
	@echo "Development environment ready!"

# Code quality targets
lint:
	@echo "Running linting checks..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Code formatting complete!"

format-check:
	@echo "Checking code format..."
	black --check --diff .
	isort --check-only --diff .

# Demo
run-demo:
	@echo "Running baby-sglang demo..."
	python run_demo.py

run-demo-quick:
	@echo "Running quick demo..."
	python run_demo.py --demo

# Cleanup targets
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleanup complete!"

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