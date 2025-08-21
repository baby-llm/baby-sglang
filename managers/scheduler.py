"""
Simplified scheduler for baby-sglang.

Based on SGLang's scheduler architecture but simplified for single GPU.
Handles request scheduling, batching, and memory management.
"""

import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from managers.io_struct import GenerateRequest, GenerateResponse
from mem_cache.radix_cache import RadixCache  
from mem_cache.memory_pool import MemoryPool
from model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Simplified scheduler managing request batching and execution.
    
    Key responsibilities:
    - Dynamic request batching
    - Memory management via RadixCache and MemoryPool
    - Model execution orchestration
    - Request lifecycle management
    """