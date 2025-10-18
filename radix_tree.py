import time
from typing import Optional, Dict, List
import torch


class TreeNode:
    def __init__(self):
        self.parent: Optional[TreeNode] = None
        self.children: Dict[int, TreeNode] = {}  # Token ID -> Child

        self.key: List[int] = []  # Token IDs
        self.value: Optional[torch.Tensor] = None  # KV indices

        self.lock_ref: int = 0
        self.last_access_time: float = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time  # for LRU
