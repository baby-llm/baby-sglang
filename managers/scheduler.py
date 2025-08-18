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
    
    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 32,
        max_seq_len: int = 2048
    ):
        """
        Initialize the scheduler.
        
        Args:
            model_path: Path to model files
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length
        """
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Request queues for dynamic batching
        self.waiting_queue = deque()
        self.running_batch = []
        
        # TODO: Initialize RadixCache for prefix caching
        # TODO: Initialize MemoryPool for GPU memory management
        # TODO: Initialize ModelRunner for inference execution
        # TODO: Setup scheduling policies (e.g., FCFS, priority)
        
        self.radix_cache = None
        self.memory_pool = None  
        self.model_runner = None
        
        # Scheduling state
        self.is_running = False
        self.scheduler_thread = None
        
        logger.info("Scheduler initialized for baby-sglang")
    
    def start(self):
        """Start the scheduler loop."""
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler loop."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Scheduler stopped")
    
    def add_request(self, request: GenerateRequest):
        """
        Add a new generation request to the waiting queue.
        
        Args:
            request: Generation request to process
        """
        # TODO: Validate request parameters
        # TODO: Check sequence length limits
        # TODO: Add to waiting queue with proper ordering
        self.waiting_queue.append(request)
        logger.debug(f"Added request {request.request_id} to queue")
    
    def _scheduler_loop(self):
        """Main scheduler loop for continuous batching."""
        while self.is_running:
            try:
                # TODO: Implement dynamic batching logic
                # TODO: Select requests from waiting queue
                # TODO: Check memory availability via MemoryPool
                # TODO: Create execution batch
                # TODO: Execute batch via ModelRunner
                # TODO: Handle finished requests
                # TODO: Update RadixCache with new KV data
                
                self._process_requests()
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _process_requests(self):
        """Process pending requests and manage running batch."""
        # TODO: Implement request processing logic
        # 1. Check for completed requests in running batch
        # 2. Add new requests from waiting queue if space available
        # 3. Execute forward pass for current batch
        # 4. Update request states
        pass
    
    def _create_batch(self) -> List[GenerateRequest]:
        """
        Create a batch of requests for execution.
        
        Returns:
            List of requests to execute in batch
        """
        # TODO: Implement intelligent batching
        # TODO: Consider memory constraints
        # TODO: Handle variable sequence lengths
        # TODO: Implement continuous batching (add/remove requests)
        batch = []
        return batch
    
    def _execute_batch(self, batch: List[GenerateRequest]):
        """
        Execute a batch of requests via ModelRunner.
        
        Args:
            batch: Batch of requests to execute
        """
        # TODO: Prepare input tensors for batch
        # TODO: Call ModelRunner.forward()
        # TODO: Process output tokens
        # TODO: Update request states
        # TODO: Handle finished requests
        pass