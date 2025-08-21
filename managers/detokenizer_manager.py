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