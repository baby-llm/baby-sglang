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
    
    def __init__(self, key: Optional[int] = None):
        """
        Initialize tree node.
        
        Args:
            key: Token ID for this node (None for root)
        """
        self.key = key
        self.children: Dict[int, 'TreeNode'] = {}
        self.parent: Optional['TreeNode'] = None
        
        # Cache data stored at this node
        self.kv_data: Optional[Any] = None  # TODO: Define KV cache tensor format
        self.last_access_time: float = 0.0
        self.ref_count: int = 0
        
        # Metadata
        self.depth: int = 0  # Distance from root
        self.seq_len: int = 0  # Length of sequence to this node
        
    def add_child(self, token_id: int) -> 'TreeNode':
        """
        Add a child node for the given token.
        
        Args:
            token_id: Token ID for the new child
            
        Returns:
            The new child node
        """
        if token_id not in self.children:
            child = TreeNode(token_id)
            child.parent = self
            child.depth = self.depth + 1
            child.seq_len = self.seq_len + 1
            self.children[token_id] = child
        
        return self.children[token_id]
    
    def get_path_tokens(self) -> List[int]:
        """
        Get the token sequence from root to this node.
        
        Returns:
            List of token IDs representing the path
        """
        if self.parent is None:
            return []
        
        path = self.parent.get_path_tokens()
        if self.key is not None:
            path.append(self.key)
        return path


class RadixCache:
    """
    Radix tree-based cache for KV data sharing.
    
    Enables efficient prefix sharing across requests by storing
    key-value cache data in a tree structure organized by token sequences.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize radix cache.
        
        Args:
            max_cache_size: Maximum number of cached nodes
        """
        self.root = TreeNode()  # Root node with no key
        self.max_cache_size = max_cache_size
        self.current_size = 0
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(f"RadixCache initialized with max size: {max_cache_size}")
    
    def insert(self, tokens: List[int], kv_data: Any) -> TreeNode:
        """
        Insert KV data for a token sequence.
        
        Args:
            tokens: Token sequence
            kv_data: Key-value cache data to store
            
        Returns:
            The leaf node where data was inserted
        """
        # TODO: Implement radix tree insertion
        # TODO: Handle cache eviction when full
        # TODO: Update access times and reference counts
        
        current = self.root
        for token in tokens:
            current = current.add_child(token)
        
        current.kv_data = kv_data
        current.ref_count += 1
        current.last_access_time = self._get_current_time()
        
        logger.debug(f"Inserted cache for {len(tokens)} tokens")
        return current
    
    def lookup(self, tokens: List[int]) -> Tuple[Optional[TreeNode], int]:
        """
        Look up cached data for a token sequence.
        
        Args:
            tokens: Token sequence to look up
            
        Returns:
            Tuple of (matching_node, match_length)
            matching_node is None if no prefix match found
        """
        # TODO: Implement prefix matching lookup
        # TODO: Update cache statistics
        # TODO: Handle partial prefix matches
        
        current = self.root
        match_length = 0
        
        for i, token in enumerate(tokens):
            if token in current.children:
                current = current.children[token]
                match_length = i + 1
            else:
                break
        
        if match_length > 0:
            self.hits += 1
            current.last_access_time = self._get_current_time()
            logger.debug(f"Cache hit: {match_length}/{len(tokens)} tokens")
            return current, match_length
        else:
            self.misses += 1
            logger.debug(f"Cache miss for {len(tokens)} tokens")
            return None, 0
    
    def evict_lru(self):
        """
        Evict least recently used cache entries.
        
        TODO: Implement LRU eviction policy
        TODO: Respect reference counts (don't evict active nodes)
        TODO: Update cache size tracking
        """
        # TODO: Find nodes with ref_count == 0 and oldest access time
        # TODO: Remove from tree and free KV data
        logger.debug("Performing LRU eviction")
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate.
        
        Returns:
            Hit rate as a percentage
        """
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return (self.hits / total_requests) * 100.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        return {
            "current_size": self.current_size,
            "max_size": self.max_cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "utilization": (self.current_size / self.max_cache_size) * 100.0
        }
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()