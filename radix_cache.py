from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import time
from memory_pool import ReqToTokenPool, BaseTokenToKVPool
from request import Request


class TreeNode:
    def __init__(self):
        self.parent: Optional[TreeNode] = None
        self.children: Dict[int, TreeNode] = {}  # Token ID -> Child

        self.key: List[int] = []  # Token IDs
        self.value: Optional[torch.Tensor] = None  # KV indices

        self.lock_ref: int = 0
        self.last_access_time: float = time.time()


class RadixCache:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.reset()

    def reset(self):
        # TODO(huangyz): implement
        raise NotImplementedError

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    def insert(self, key: List, value=None):
        raise NotImplementedError

    def cache_finished_req(self, req: Request, token_ids: Optional[List[int]] = None):
        raise NotImplementedError

    def cache_unfinished_req(self, req: Request, token_ids: Optional[List[int]] = None):
        raise NotImplementedError

    def evict(self, num_tokens: int, evict_callback: Callable):
        raise NotImplementedError

    def inc_lock_ref(self, node: TreeNode):
        raise NotImplementedError

    def dec_lock_ref(self, node: TreeNode):
        raise NotImplementedError

    def evictable_size(self):
        raise NotImplementedError
