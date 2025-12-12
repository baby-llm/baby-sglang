from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import time
from memory_pool import ReqToTokenPool, BaseTokenToKVPool
from request import Request
import heapq
from radix_tree import TreeNode

# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.reset()  # set a dummy empty node

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = torch.tensor([], dtype=torch.int32)
        self.root_node.lock_ref = 1

        self.evictable_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, TreeNode]:
        value: List[torch.Tensor] = []  # matched prefix kv indices
        last_node_box: List[TreeNode] = [
            self.root_node
        ]  # mutable box to track last matched node

        self._match_prefix_helper(self.root_node, key, value, last_node_box)

        if value:
            out = torch.concat(value)
        else:
            out = torch.tensor([], dtype=torch.int32)
        return out, last_node_box[0]

    def _match_prefix_helper(
        self,
        parent_node: TreeNode,
        key: List[int],
        value: List[torch.Tensor],
        last_node_box: List[TreeNode],
    ):
        parent_node.last_access_time = time.time()

        if len(key) == 0:
            return

        if key[0] in parent_node.children:
            child = parent_node.children[key[0]]
            prefix_len = _key_match(
                child.key, key
            )  # assert prefix_len in [1, len(child.key)]

            if prefix_len < len(child.key):
                # case2 partial match
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node_box[0] = new_node

            else:
                # case3 total match
                value.append(child.value)
                last_node_box[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node_box)

        # case1 no match, do nothing

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def insert(self, key: List[int], value: Optional[torch.Tensor] = None) -> int:
        # Fallback when value (KV indices) not provided: use token IDs shape as placeholder
        # Real callers (cache_* methods) pass KV indices tensors.
        if value is None:
            value = torch.tensor(key, dtype=torch.int32)
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, parent_node: TreeNode, key: List, value):
        parent_node.last_access_time = time.time()

        if len(key) == 0:
            return 0

        if key[0] in parent_node.children:
            child = parent_node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                # case 3
                if prefix_len == len(key):
                    return prefix_len

                key = key[prefix_len:]
                value = value[prefix_len:]
                return prefix_len + self._insert_helper(child, key, value)

            else:
                # case 2
                new_node = self._split_node(child.key, child, prefix_len)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )
        else:
            # case 1
            if len(key):
                new_node = TreeNode()
                new_node.key = key
                new_node.value = value

                new_node.parent = parent_node
                parent_node.children[key[0]] = new_node

                self.evictable_size_ += len(value)

            return 0  # no match

    def cache_finished_req(self, req: Request, token_ids: Optional[List[int]] = None):
        # Derive token_ids when not provided:
        # Exclude last generated token as it may not have KV written yet (prefill-finished case).
        if token_ids is None:
            base_ids = req.input_ids.tolist()
            if len(req.output_ids) > 0:
                token_ids = (base_ids + req.output_ids)[:-1]
            else:
                token_ids = base_ids

        if len(token_ids) == 0:
            # Nothing to cache; just release locks and req slot
            self.req_to_token_pool.free(req.req_pool_idx)
            self.dec_lock_ref(req.last_node)
            return

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_prefix_len = self.insert(token_ids, kv_indices.clone())

        # Free duplicated KV slots that were already present in the tree
        if new_prefix_len > len(req.prefix_indices):
            self.token_to_kv_pool.free(
                kv_indices[len(req.prefix_indices) : new_prefix_len]
            )

        # Free request slot and release prefix lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Request, token_ids: Optional[List[int]] = None):
        # Derive token_ids when not provided:
        # For unfinished requests, use the full sequence available so far.
        if token_ids is None:
            token_ids = req.input_ids.tolist() + req.output_ids

        if len(token_ids) == 0:
            return

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_prefix_len = self.insert(token_ids, kv_indices.clone())

        # Free duplicated KV slots that were already present in the tree
        if new_prefix_len > len(req.prefix_indices):
            self.token_to_kv_pool.free(
                kv_indices[len(req.prefix_indices) : new_prefix_len]
            )

        # Re-match to get updated prefix indices after insertion
        new_indices, new_last_node = self.match_prefix(token_ids)
        assert len(new_indices) == len(token_ids)

        # Write updated indices for the newly cached part
        if len(new_indices) > len(req.prefix_indices):
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
                new_indices[len(req.prefix_indices) :],
            )

        # Update locks and request metadata
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def evict(self, num_tokens: int, evict_callback: Callable):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.value)

    def inc_lock_ref(self, node: TreeNode):
        if node is None:
            return 0
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if node is None:
            return 0
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_
