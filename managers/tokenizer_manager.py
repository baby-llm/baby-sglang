"""
Tokenizer manager for baby-sglang.

Handles request tokenization and communication with scheduler.
Runs in the main process alongside the Engine.
"""

import logging
from typing import List, Optional

from managers.io_struct import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Simplified tokenizer manager for baby-sglang.
    
    Responsibilities:
    - Tokenize input prompts
    - Send requests to Scheduler
    - Receive responses from DetokenizerManager
    - Handle request lifecycle
    """