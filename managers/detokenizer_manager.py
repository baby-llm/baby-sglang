"""
Detokenizer manager for baby-sglang.

Handles response detokenization and runs in a separate subprocess.
Receives output tokens from Scheduler and sends final text back.
"""

import logging
from typing import List

from managers.io_struct import GenerateResponse

logger = logging.getLogger(__name__)


class DetokenizerManager:
    """
    Simplified detokenizer manager for baby-sglang.
    
    Responsibilities:
    - Receive output tokens from Scheduler
    - Detokenize tokens to text
    - Send final responses back to TokenizerManager
    - Handle streaming responses
    """
    
    def __init__(self):
        """Initialize detokenizer manager."""
        # TODO: Setup ZMQ communication with Scheduler
        # TODO: Setup ZMQ communication with TokenizerManager
        # TODO: Initialize tokenizer for detokenization
        
        self.tokenizer = None
        self.is_running = False
        
        logger.info("DetokenizerManager initialized")
    
    def start(self):
        """Start the detokenizer process loop."""
        # TODO: Start main processing loop
        # TODO: Listen for messages from Scheduler
        # TODO: Process detokenization requests
        self.is_running = True
        logger.info("DetokenizerManager started")
    
    def stop(self):
        """Stop the detokenizer process."""
        self.is_running = False
        logger.info("DetokenizerManager stopped")
    
    def process_tokens(self, tokens: List[int], request_id: str) -> GenerateResponse:
        """
        Process output tokens and create response.
        
        Args:
            tokens: List of output token IDs
            request_id: ID of the originating request
            
        Returns:
            GenerateResponse with detokenized text
        """
        # TODO: Detokenize tokens to text
        # TODO: Handle special tokens and formatting
        # TODO: Create GenerateResponse object
        # TODO: Send response back via ZMQ
        
        text = self._detokenize(tokens)
        response = GenerateResponse(
            request_id=request_id,
            text=text,
            tokens=tokens
        )
        
        logger.debug(f"Processed response for request {request_id}")
        return response
    
    def _detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        # TODO: Use tokenizer to decode tokens
        # TODO: Handle proper text formatting
        logger.debug(f"Detokenizing {len(tokens)} tokens")
        return "TODO: Implement detokenization"