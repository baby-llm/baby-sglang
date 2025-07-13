import zmq
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoModelForCausalLM

from ..model_executor.patch import patch_model
from ..utils.utils import get_server_socket, Req, Finish, ModelConfig
from ..mem_cache.memory_pool import MemoryPool
from ..mem_cache.radix_cache import RadixCache

class ModelRunner(nn.Module):
    """
    一个简单的模型包装器，负责加载模型并提供一个统一的 `forward` 接口。
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        
        # 使用 transformers 加载模型权重
        # torch_dtype=torch.float16 用于在 GPU 上使用半精度浮点数，以节省内存和提高速度。
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.path,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        # `kv_cache_params` 是一个关键参数，它将 PagedAttention 的信息传递给模型。
        # 它通常包含 `block_tables`, `context_lens` 等。
        kv_cache_params: Optional[dict] = None,
    ):
        """
        模型的单步前向传播。
        
        Args:
            input_ids: 当前批次需要处理的 token ID。
            position_ids: 对应于 `input_ids` 的位置 ID。
            kv_cache_params: 包含 PagedAttention 信息的字典。
        
        Returns:
            模型的输出 logits。
        """
        # SGLang 的一个核心修改是，它会 hook 模型的 `forward` 方法，
        # 并将 `kv_cache_params` 传递给 Attention 层。
        # 在 `nano-sglang` 中，我们假设使用的模型（或其 Attention 实现）
        # 已经被修改为可以接收这些参数。
        # 例如，vLLM 和 SGLang 都对 LlamaAttention 等做了修改。
        # 为简单起见，我们在此直接传递这些参数。

        # TODO: 要满足这里的假设使得代码可以直接运行 需要做出什么样的修改
        # TODO: 优先选择规模比较小的开源模型 比如qwen gemma gpt2 ...
        # 打过补丁的模型 forward 方法返回的是一个元组 (attn_output, None, None)
        # 我们只关心第一个元素 attn_output，它在这里是模型的最终输出 logits
        model_output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache_params=kv_cache_params
        )
        return model_output[0]

class Scheduler:
    """
    Scheduler 是 nano-sglang 的调度核心，负责管理所有推理请求的生命周期。
    
    其主要职责包括：
    1.  **请求管理**: 从 Tokenizer 接收请求，并将其放入等待队列。
    2.  **调度循环**: 实现 Continuous Batching (持续批处理) 的核心逻辑。
        -   将等待队列中的请求，根据当前的内存和批处理容量，调度到运行批次中。
        -   管理 `waiting_queue`, `running_batch`, `finished_batch` 等状态。
    3.  **内存管理**: 与 `MemoryPool` (用于 PagedAttention) 和 `RadixCache`
        (用于前缀共享) 交互，为请求的 token 分配和释放 KV 缓存。
    4.  **模型执行**: 准备模型输入 (`input_ids`, `position_ids`, `attn_mask` 等)，
        调用模型执行一步前向推理，并获取 logits。
    5.  **采样与后处理**: 根据 logits 和采样参数，生成新的 token。
    6.  **结果分发**: 将生成的 token 或完成信号发送给 Detokenizer。
    
    这个类的设计大量借鉴了 `sglang.srt.managers.scheduler.Scheduler`，但简化了
    分布式通信、多副本同步 (all-gather) 等复杂功能。
    """
    def __init__(self, model_path: str, server_name: str,
                 detokenizer_server_name: str, mem_fraction_static: float, tp_size: int,
                 block_size: int = 16):
        self.model_path = model_path
        self.server_name = server_name
        self.detokenizer_server_name = detokenizer_server_name
        self.mem_fraction_static = mem_fraction_static
        self.tp_size = tp_size
        self.block_size = block_size
        self.device = "cuda"

        # 1. 初始化模型配置和模型本身
        self.model_config = ModelConfig(path=model_path, trust_remote_code=True)
        self.model = ModelRunner(self.model_config)
        # 应用猴子补丁，将模型中的 LlamaAttention 替换为我们自己的实现
        patch_model(self.model)
        self.model = self.model.to(self.device)
        
        # 2. 初始化内存管理器
        # 使用从 ModelConfig 获取的真实参数来计算 KV Cache 的大小
        layer_num = self.model_config.num_hidden_layers
        head_num = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim
        
        # 计算单个物理块（block）的 K 和 V 缓存大小
        # 每个 token 需要 head_num * head_dim 个浮点数来存储 K 或 V
        # 一个块有 block_size 个 token
        # SGLang 中，K 和 V 缓存的大小是分开计算的，但通常它们是相同的。
        self.k_cache_size_per_block = self.block_size * head_num * head_dim
        self.v_cache_size_per_block = self.block_size * head_num * head_dim

        self.memory_pool = MemoryPool(
            k_cache_size=self.k_cache_size_per_block,
            v_cache_size=self.v_cache_size_per_block,
            block_size=self.block_size,
            device=self.device,
            mem_fraction_static=self.mem_fraction_static
        )
        self.radix_cache = RadixCache(self.memory_pool)
        
        # 初始化请求队列
        self.waiting_queue: List[Req] = []
        self.running_batch: List[Req] = []
        self.finished_batch: List[Req] = []

        # 初始化 ZMQ sockets
        context = zmq.Context()
        self.receiver = get_server_socket(context, server_name, zmq.PULL, bind=True)
        self.detokenizer_sender = get_server_socket(context, detokenizer_server_name, zmq.PUSH, bind=False)

    def loop(self):
        """
        调度器的无限主循环。
        这个循环模仿了 SGLang 的两阶段调度策略：decode 优先。
        """
        while True:
            # 1. 接收新请求
            self._receive_new_requests()

            # 2. 调度以决定哪些请求进入 prefill 或 decode 批次
            prefill_reqs, decode_reqs = self._schedule()

            # 3. Decode 优先：首先执行 decode 批次
            if decode_reqs:
                self._step(decode_reqs)

            # 4. 然后执行 prefill 批次
            if prefill_reqs:
                self._step(prefill_reqs)

    def _receive_new_requests(self):
        """
        从 ZMQ PULL socket 非阻塞地接收新请求，并加入等待队列。
        """
        try:
            while True:
                req = self.receiver.recv_pyobj(flags=zmq.NOBLOCK)
                self.waiting_queue.append(req)
        except zmq.Again:
            # 当没有更多消息时，`recv_pyobj` 会抛出 `zmq.Again` 异常
            pass

    def _schedule(self) -> (List[Req], List[Req]):
        """
        调度逻辑的核心，此方法决定哪些请求被 prefill，哪些被 decode。
        它返回两个独立的批次。
        """
        # 1. 首先处理已完成的请求并从运行批次中移除
        new_running_batch = []
        for req in self.running_batch:
            if self._check_finish(req):
                all_output_ids = req.input_ids + req.output_ids
                self.detokenizer_sender.send_pyobj(Finish(request_id=req.request_id, req_output=all_output_ids))
                self.radix_cache.free(all_output_ids)
            else:
                new_running_batch.append(req)
        self.running_batch = new_running_batch

        # 2. 确定 decode 批次：所有仍在运行的请求都需要 decode
        decode_reqs = self.running_batch

        # 3. 确定 prefill 批次：从等待队列中调度新请求
        prefill_reqs = []
        temp_waiting_queue = []
        for req in self.waiting_queue:
            try:
                block_indices = self.radix_cache.insert(req.input_ids)
                req.token_to_kv_pool = {i: block for i, block in enumerate(block_indices)}
                prefill_reqs.append(req)
            except RuntimeError:
                # 内存不足，将此请求和所有后续请求保留在等待队列
                temp_waiting_queue.append(req)
        
        self.waiting_queue = temp_waiting_queue
        
        # 将新 prefill 的请求加入到总的运行队列中
        self.running_batch.extend(prefill_reqs)

        return prefill_reqs, decode_reqs


    def _prepare_model_input(self, batch: List[Req]) -> dict:
        """
        为给定的批次（prefill或decode）准备模型输入。
        """
        input_ids = []
        position_ids = []
        context_lens = []
        block_tables = []
        
        for req in batch:
            context_len = len(req.input_ids) + len(req.output_ids)
            context_lens.append(context_len)

            # `input_ids` 和 `position_ids`:
            # - Prefill 阶段 (req.output_ids is empty): 需要处理整个 prompt。
            #   `input_ids` 是 prompt 的所有 token, `position_ids` 是 [0, 1, ..., L-1]。
            # - Decode 阶段: 只需要处理最新生成的那个 token。
            #   `input_ids` 是单个 token, `position_ids` 是它的真实位置 (context_len - 1)。
            # 模型一次性处理所有这些 token，无论它们是来自 prefill 还是 decode。
            if not req.output_ids:
                tokens_to_process = req.input_ids
                start_pos = 0
            else:
                tokens_to_process = [req.output_ids[-1]]
                start_pos = context_len - 1
            
            input_ids.extend(tokens_to_process)
            position_ids.extend(list(range(start_pos, context_len)))
            
            # `block_tables`: 将逻辑 token 映射到物理内存块的核心。
            # 它是一个形状为 [num_requests, max_blocks_per_req] 的张量。
            # `block_tables[i][j]` 的值是在 `MemoryPool` 中第 j 个逻辑块对应的物理块索引。
            physical_blocks = list(req.token_to_kv_pool.values())
            block_tables.append(physical_blocks)

        # 对 block_tables 进行填充（padding），使其成为一个矩形张量
        max_blocks_per_req = max(len(bt) for bt in block_tables) if block_tables else 0
        for i in range(len(block_tables)):
            padding_len = max_blocks_per_req - len(block_tables[i])
            block_tables[i].extend([0] * padding_len) # 使用 0 作为填充值
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "position_ids": torch.tensor(position_ids, dtype=torch.long, device=self.device),
            "kv_cache_params": {
                "context_lens": torch.tensor(context_lens, dtype=torch.int, device=self.device),
                "block_tables": torch.tensor(block_tables, dtype=torch.long, device=self.device),
                "kv_cache_pool": (self.memory_pool.gpu_k_cache_pool, self.memory_pool.gpu_v_cache_pool),
            }
        }

    def _sample(self, logits: torch.Tensor) -> List[int]:
        """从 logits 中为批次中的每个请求进行采样。"""
        # 使用贪心采样（argmax）
        return torch.argmax(logits, dim=-1).tolist()

    def _step(self, batch: List[Req]):
        """
        为给定的批次（prefill或decode）执行一步完整的模型推理、采样和状态更新。
        """
        # 1. 准备模型输入
        model_input = self._prepare_model_input(batch)

        # 2. 执行模型前向传播
        logits = self.model.forward(**model_input)
        
        # 3. 采样
        # `logits` 的形状是 [num_tokens, vocab_size]，我们需要根据 context_lens 提取
        # 每个请求对应的最后一个 token 的 logit
        last_token_indices = torch.cumsum(
            torch.tensor([len(req) for req in batch], device=self.device), dim=0
        ) - 1
        last_token_logits = logits[last_token_indices]
        new_token_ids = self._sample(last_token_logits)

        # 4. 更新状态和分发结果
        for i, req in enumerate(batch):
            new_token_id = new_token_ids[i]
            
            # 为新 token 分配内存并更新 Radix 树
            try:
                new_block_idx = self.memory_pool.alloc(1)[0]
                full_sequence = req.input_ids + req.output_ids
                self.radix_cache.append_token(full_sequence, new_token_id, new_block_idx)
                
                # 更新请求状态
                req.output_ids.append(new_token_id)
                new_logic_pos = len(full_sequence)
                req.token_to_kv_pool[new_logic_pos] = new_block_idx
                
                # 发送新 token 给 Detokenizer
                self.detokenizer_sender.send_pyobj((req.request_id, new_token_id))

            except RuntimeError as e:
                # 如果在 decode 阶段内存不足，这是一个更严重的问题
                # 这里我们简化处理：直接标记为完成并释放资源
                print(f"Out of memory during decode for request {req.request_id}. Finishing early. Details: {e}")
                self._check_finish(req, force_finish=True)


    def _check_finish(self, req: Req, force_finish: bool = False) -> bool:
        """检查请求是否完成"""
        if force_finish:
            return True
            
        max_new = req.sampling_params.get("max_new_tokens", 32)
        # 假设 EOS token id 是 2 (这在很多模型中是常见的，但应从 tokenizer 配置中获取)
        eos_token_id = 2
        
        return (
            len(req.output_ids) >= max_new or
            (req.output_ids and req.output_ids[-1] == eos_token_id)
        )

def run_scheduler(model_path: str, server_name: str,
                  detokenizer_server_name: str, mem_fraction_static: float, tp_size: int):
    """
    Scheduler 进程的入口函数。
    """
    scheduler = Scheduler(
        model_path,
        server_name,
        detokenizer_server_name,
        mem_fraction_static,
        tp_size,
    )
    print(f"Scheduler is ready at {server_name}")
    scheduler.loop()