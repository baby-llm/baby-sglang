import torch
from typing import List

class MemoryPool:
    """
    MemoryPool 负责管理 KV Cache 的物理内存。
    
    这是 PagedAttention 机制的基础。它将预先分配的一大块 GPU 内存 (`gpu_mem_pool`)
    分割成固定大小的块 (blocks)，并维护一个空闲块列表 (`free_blocks`)。
    
    它的主要职责是：
    1.  初始化时，根据模型参数 (num_layers, num_heads, head_dim) 和批处理大小，
        预先分配一块连续的 GPU 张量作为 KV Cache 池。
    2.  将这块大张量视图化为一个个小的 "block"。
    3.  提供 `alloc` 方法，从空闲列表中取出一个或多个块的索引。
    4.  提供 `free` 方法，将使用完毕的块的索引归还到空闲列表中。
    
    这个类的设计参考了 `sglang.srt.mem_cache.memory_pool.MemoryPool`，
    但简化了对不同数据类型 (dtype) 和分布式环境 (TP) 的处理。
    """
    def __init__(self, k_cache_size: int, v_cache_size: int,
                 block_size: int, device: str, mem_fraction_static: float):
        """
        初始化内存池。

        Args:
            k_cache_size (int): 单个 K-cache 块的元素数量。
            v_cache_size (int): 单个 V-cache 块的元素数量。
            block_size (int): 每个内存块可以存储的 token 数量。
            device (str): 内存分配的目标设备 (e.g., "cuda")。
            mem_fraction_static (float): 用于 KV Cache 的静态 GPU 内存比例。
        """
        self.k_cache_size = k_cache_size
        self.v_cache_size = v_cache_size
        self.block_size = block_size
        self.device = device
        self.mem_fraction_static = mem_fraction_static

        # 计算总共需要多少个块 (num_blocks)
        # 这通常基于可用的 GPU 内存和每个块的大小来决定
        self.num_blocks = self._calculate_num_blocks()
        
        # 预分配 K 和 V 的物理内存池
        # 形状为 (num_blocks, k_cache_size_per_block)
        self.gpu_k_cache_pool = torch.empty(
            (self.num_blocks, self.k_cache_size), dtype=torch.float16, device=self.device
        )
        self.gpu_v_cache_pool = torch.empty(
            (self.num_blocks, self.v_cache_size), dtype=torch.float16, device=self.device
        )

        # 初始化空闲块列表，包含所有块的索引
        self.free_blocks: List[int] = list(range(self.num_blocks))

    def _calculate_num_blocks(self) -> int:
        """
        根据可用 GPU 内存和指定的比例计算可以分配的总块数。
        """
        if self.device != "cuda":
            # 如果不是在 CUDA 设备上，我们返回一个固定的默认值，因为没有 GPU 内存可查。
            return 2048

        # torch.cuda.mem_get_info() 返回一个元组 (free_memory, total_memory)，单位是字节。
        free_mem, total_mem = torch.cuda.mem_get_info()
        
        # 计算指定用于 KV Cache 的总内存量
        total_mem_for_kv_cache = int(total_mem * self.mem_fraction_static)
        
        # 计算单个块（包含 K 和 V）所需的内存大小
        # 假设使用 float16，每个元素占 2 个字节
        bytes_per_element = 2
        block_size_in_bytes = (self.k_cache_size + self.v_cache_size) * bytes_per_element
        
        # 计算可以容纳的总块数
        num_blocks = total_mem_for_kv_cache // block_size_in_bytes
        
        print(f"Total GPU mem: {total_mem / 1e9:.2f} GB, "
              f"Available mem for KV cache: {total_mem_for_kv_cache / 1e9:.2f} GB, "
              f"Allocating {num_blocks} blocks.")
              
        return num_blocks

    def alloc(self, num_blocks: int = 1) -> List[int]:
        """
        从空闲列表中分配指定数量的块。

        Args:
            num_blocks (int): 需要分配的块的数量。

        Returns:
            List[int]: 一个包含分配的块索引的列表。如果空闲块不足，
                       则返回一个空列表或抛出异常。
        """
        if self.get_freed_block_num() < num_blocks:
            # 在真实系统中，这里应该抛出一个特定的 OutOfMemory 异常
            raise RuntimeError("Out of memory in MemoryPool")
        
        allocated_blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return allocated_blocks

    def free(self, block_indices: List[int]):
        """
        将一个或多个块归还到空闲列表。

        Args:
            block_indices (List[int]): 要释放的块的索引列表。
        """
        # 在真实实现中，可能需要检查重复释放等问题
        self.free_blocks.extend(block_indices)

    def get_freed_block_num(self) -> int:
        """返回当前空闲块的数量。"""
        return len(self.free_blocks)

    def get_total_block_num(self) -> int:
        """返回总块数。"""
        return self.num_blocks