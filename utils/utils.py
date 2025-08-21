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