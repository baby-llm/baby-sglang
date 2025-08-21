"""
Radix cache implementation for baby-sglang.

Implements prefix caching using a radix tree structure for efficient
memory reuse across requests with shared prefixes.
Based on SGLang's RadixCache but simplified.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class TreeNode:
    """
    Node in the radix tree for prefix caching.
    
    Each node represents a sequence of tokens and stores
    the corresponding key-value cache data.
    """

class RadixCache:
    """
    Radix tree-based cache for KV data sharing.
    
    Enables efficient prefix sharing across requests by storing
    key-value cache data in a tree structure organized by token sequences.
    """