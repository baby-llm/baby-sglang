"""
Utility functions for baby-sglang.

Common helper functions and utilities used across the system.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        format_str: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory stats in GB
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            return {
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - reserved) / (1024**3)
            }
        else:
            return {"total_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    except ImportError:
        return {"total_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}


def validate_model_path(model_path: str) -> bool:
    """
    Validate that model path exists and contains required files.
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        True if valid model path
    """
    if not os.path.exists(model_path):
        return False
    
    # TODO: Add more specific validation
    # - Check for config.json
    # - Check for model weights (pytorch_model.bin, model.safetensors, etc.)
    # - Check for tokenizer files
    
    return True


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes value to human readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(bytes_value)
    
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    
    return f"{value:.1f} {units[-1]}"


def create_unique_id(prefix: str = "req") -> str:
    """
    Create a unique identifier.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Unique identifier string
    """
    import time
    import uuid
    
    timestamp = int(time.time() * 1000)  # milliseconds
    unique_suffix = str(uuid.uuid4())[:8]
    
    return f"{prefix}_{timestamp}_{unique_suffix}"


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module, returning None if import fails.
    
    Args:
        module_name: Name of module to import
        package: Package name for relative imports
        
    Returns:
        Imported module or None if import failed
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError:
        return None


def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the runtime environment.
    
    Returns:
        Dictionary with environment information
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
    }
    
    # Add GPU info if available
    torch = safe_import("torch")
    if torch and torch.cuda.is_available():
        info.update({
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        })
    else:
        info["cuda_available"] = False
    
    return info


class Timer:
    """Simple timer utility for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        import time
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop timing and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        import time
        self.end_time = time.time()
        
        if self.start_time is None:
            return 0.0
        
        return self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """
        Get elapsed time without stopping.
        
        Returns:
            Elapsed time in seconds
        """
        import time
        if self.start_time is None:
            return 0.0
        
        return time.time() - self.start_time