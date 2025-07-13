from typing import Dict, List, Optional
from .memory_pool import MemoryPool

class RadixTreeNode:
    """
    RadixCache 的节点。
    
    每个节点代表 token 序列中的一个位置。它存储了指向子节点的指针
    以及与该节点关联的 KV Cache 物理块的索引。
    """
    def __init__(self, parent=None, block_idx: Optional[int] = None):
        # 子节点的字典，key 是 token_id
        self.children: Dict[int, RadixTreeNode] = {}
        # 指向父节点的指针
        self.parent = parent
        # 引用计数，记录有多少个请求共享了这个节点（即这个前缀）
        self.references = 0
        # 该节点对应的 KV Cache 在 MemoryPool 中的物理块索引
        self.block_idx = block_idx

class RadixCache:
    """
    RadixCache (基数缓存) 使用 Radix 树 (前缀树) 来管理和共享 token 序列的 KV 缓存。
    
    其核心功能是：
    1.  **共享前缀**: 如果多个请求具有相同的前缀，它们可以共享存储该前缀 KV 值的
        物理内存块，从而避免重复计算。
    2.  **动态分配**: 对于每个请求中超出共享前缀的新 token，它会从 `MemoryPool`
        中动态分配新的内存块。
    3.  **引用计数**: 通过引用计数机制来管理共享内存的生命周期。当一个内存块不再
        被任何请求共享时，它将被释放回 `MemoryPool`。
        
    这个类的设计是 SGLang 高性能的关键之一，其实现直接参考了
    `sglang.srt.mem_cache.radix_cache.RadixCache`。
    """
    def __init__(self, memory_pool: MemoryPool):
        """
        初始化 RadixCache。
        
        Args:
            memory_pool (MemoryPool): 用于分配和释放物理内存块的内存池实例。
        """
        self.memory_pool = memory_pool
        # 根节点不代表任何 token，也不关联任何物理块
        self.root = RadixTreeNode()

    def match_prefix(self, seq: List[int]) -> int:
        """
        查找给定 token 序列在缓存中匹配的最长前缀。

        Args:
            seq (List[int]): 输入的 token ID 序列。

        Returns:
            int: 匹配的最长前缀的长度。
        """
        node = self.root
        match_len = 0
        for token in seq:
            if token in node.children:
                node = node.children[token]
                match_len += 1
            else:
                break
        return match_len
        
    def insert(self, seq: List[int]) -> List[int]:
        """
        将一个 token 序列插入到 Radix 树中。
        
        该方法会沿着树查找，复用已有的前缀节点，并为新的 token 创建新节点并分配内存。
        同时，它会增加路径上所有节点的引用计数。

        Args:
            seq (List[int]): 要插入的 token ID 序列。

        Returns:
            List[int]: 一个列表，包含了代表该序列的、从 `MemoryPool` 分配的所有
                       物理块的索引 (block indices)。
        """
        node = self.root
        node.references += 1
        block_indices = []

        for token in seq:
            if token in node.children:
                # 前缀已存在，移动到下一个节点
                node = node.children[token]
                block_indices.append(node.block_idx)
            else:
                # 前缀不存在，需要创建新节点并分配内存
                new_block_idx = self.memory_pool.alloc(1)[0]
                new_node = RadixTreeNode(parent=node, block_idx=new_block_idx)
                node.children[token] = new_node
                node = new_node
                block_indices.append(new_block_idx)
            
            # 增加新路径上节点的引用计数
            node.references += 1
            
        return block_indices

    def append_token(self, all_input_ids: List[int], new_token_id: int, new_block_idx: int):
        """
        在现有序列的末尾追加一个新 token，并关联一个新的物理块。
        这用于 `decode` 阶段的增量更新。

        Args:
            all_input_ids (List[int]): 完整的前缀序列（不包含新 token）。
            new_token_id (int): 要追加的新 token。
            new_block_idx (int): 从 MemoryPool 分配的、与新 token 关联的物理块索引。
        """
        # 1. 找到前缀序列的末端节点
        node = self.root
        for token in all_input_ids:
            # 假设 `all_input_ids` 已经存在于树中
            node = node.children[token]
        
        # 2. 创建并插入新节点
        new_node = RadixTreeNode(parent=node, block_idx=new_block_idx)
        node.children[new_token_id] = new_node
        new_node.references = 1 # 新节点只被当前请求引用

    def free(self, seq: List[int]):
        """
        释放一个 token 序列。
        
        这会减少序列路径上所有节点的引用计数。如果一个节点的引用计数降为零，
        则将其对应的物理块归还给 `MemoryPool`，并从树中删除该节点。
        
        Args:
            seq (List[int]): 要释放的 token ID 序列。
        """
        node = self.root
        node.references -= 1

        for token in seq:
            if token not in node.children:
                # 理论上不应发生，表示尝试释放一个不存在的序列
                # 在真实实现中可以添加日志或断言
                return

            child_node = node.children[token]
            child_node.references -= 1
            
            if child_node.references == 0:
                # 引用计数为零，释放内存块并从树中删除节点
                self.memory_pool.free([child_node.block_idx])
                del node.children[token]
            
            node = child_node