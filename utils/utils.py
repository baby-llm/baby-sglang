import os
import zmq
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from transformers import AutoConfig

# ============================================
#               Data Structures
# ============================================

@dataclass
class Req:
    """
    Req (Request) 对象是在 `nano-sglang` 系统内部流转的核心数据结构。
    它封装了一个推理请求的所有状态信息。
    """
    request_id: str
    input_ids: List[int]
    sampling_params: Dict[str, Any]
    
    # 动态变化的字段
    output_ids: List[int] = field(default_factory=list)
    
    # PagedAttention 相关字段
    # `token_to_kv_pool` 是一个从 token 在序列中的逻辑位置到其在
    # `MemoryPool` 中物理块索引的映射。
    # 这是一个实现 PagedAttention 的关键数据结构。
    token_to_kv_pool: Dict[int, int] = field(default_factory=dict)
    
    def __len__(self):
        """返回请求的总长度（输入 + 输出）。"""
        return len(self.input_ids) + len(self.output_ids)

@dataclass
class Finish:
    """
    一个简单的信号对象，由 Scheduler 发送给 Detokenizer，
    表示一个请求已经全部处理完成。
    """
    request_id: str
    req_output: List[int] # 完整的输出 token 序列

@dataclass
class ModelConfig:
    """
    存储从 HuggingFace Hub 或本地路径加载的模型配置信息。
    """
    def __init__(self, path: str, trust_remote_code: bool):
        self.path = path
        self.trust_remote_code = trust_remote_code
        
        # 使用 transformers.AutoConfig 加载模型的 config.json
        config = AutoConfig.from_pretrained(
            path, trust_remote_code=trust_remote_code
        )
        
        # 从加载的 config 中提取核心参数
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        
        # 计算 head_dim，这对于 KV Cache 的大小计算至关重要
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # 模型类型，用于可能的特殊处理
        self.model_type = config.model_type

# ============================================
#           Inter-Process Communication
# ============================================

def get_server_name(server_type: str) -> str:
    """根据服务类型生成 ZMQ 的 IPC (Inter-Process Communication) 地址。"""
    # IPC 使用文件系统路径作为地址
    return f"ipc:///tmp/nano_sglang_{server_type}.sock"

def get_all_server_names() -> List[str]:
    """返回所有服务类型的标准名称。"""
    return [get_server_name(st) for st in ["scheduler", "detokenizer"]]

def get_and_register_new_filename(server_name: str):
    """清理可能存在的旧 ZMQ socket 文件。"""
    filename = server_name.replace("ipc://", "")
    if os.path.exists(filename):
        os.remove(filename)

def get_server_socket(context: zmq.Context, server_name: str, 
                      socket_type: int, bind: bool) -> zmq.Socket:
    """
    创建一个 ZMQ socket 并根据 `bind` 参数决定是绑定地址还是连接地址。
    
    Args:
        context: ZMQ 上下文。
        server_name: 服务的 IPC 地址。
        socket_type: ZMQ socket 类型 (e.g., zmq.PULL, zmq.PUSH)。
        bind: 如果为 `True`，则 socket 绑定到地址 (服务器端)；
              如果为 `False`，则 socket 连接到地址 (客户端)。
    """
    socket = context.socket(socket_type)
    if bind:
        socket.bind(server_name)
    else:
        socket.connect(server_name)
    return socket

def send_req_to_scheduler(scheduler_server_name: str,
                          req: Req):
    """
    一个辅助函数，用于从外部 (如 SGLangEngine) 向 Scheduler 发送请求。
    这是一个一次性的 ZMQ 连接，发送后即关闭。
    """
    context = zmq.Context()
    sender = get_server_socket(context, scheduler_server_name, zmq.PUSH, bind=False)
    
    sender.send_pyobj(req)

    # 短暂等待以确保消息发送出去
    sender.close()
    context.term()