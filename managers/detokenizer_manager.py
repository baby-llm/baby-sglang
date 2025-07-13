import zmq
from transformers import AutoTokenizer
from typing import Dict, List

from ..utils.utils import get_server_socket, Finish

def run_detokenizer(tokenizer_path: str, server_name: str, req_queue):
    """
    DetokenizerManager 的主运行函数，在一个独立的进程中执行。

    它的核心职责是：
    1.  从 Scheduler 接收生成的 token ID 或完成信号。
    2.  对于每个请求，维护一个解码状态，将 token ID 序列实时解码为文本。
    3.  当收到一个请求的完成信号 (`Finish`) 时，将该请求的最终解码文本
        通过 `req_queue` 发回给主进程的 `SGLangEngine`。

    这个函数的设计模仿了 SGLang 中的 `srt.managers.detokenizer_manager.DetokenizerManager`。

    Args:
        tokenizer_path (str): Tokenizer 模型的路径，用于解码。
        server_name (str): 当前 Detokenizer 服务的 ZMQ 地址。
        req_queue (multiprocessing.Queue): 用于将最终结果返回给主进程的队列。
    """
    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 初始化 ZMQ socket，用于从 Scheduler 接收数据
    context = zmq.Context()
    receiver = get_server_socket(context, server_name, zmq.PULL, bind=True)

    # 用于缓存每个请求已经生成的 token IDs
    # key: request_id, value: List[int]
    request_outputs: Dict[str, List[int]] = {}

    print(f"Detokenizer manager is ready at {server_name}")

    while True:
        try:
            # 1. 从 Scheduler 接收数据
            # 数据可能是 `(request_id, token_id)` 或 `Finish` 对象
            data = receiver.recv_pyobj()

            if isinstance(data, Finish):
                # 2. 如果是 Finish 信号
                # a. 获取最终的 token 序列
                final_tokens = data.req_output
                
                # b. 解码成完整文本
                output_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
                
                # c. 将最终结果放入返回队列
                req_queue.put({
                    "request_id": data.request_id,
                    "text": output_text
                })
                
                # d. 清理缓存
                if data.request_id in request_outputs:
                    del request_outputs[data.request_id]

            else:
                # 3. 如果是单个 token
                request_id, new_token_id = data
                
                # a. 将新 token 添加到对应请求的缓存中
                if request_id not in request_outputs:
                    request_outputs[request_id] = []
                request_outputs[request_id].append(new_token_id)
                
                # 在流式（streaming）实现中，可以在这里就进行解码并返回部分结果。
                # 为简化，我们只在请求完成时返回最终结果。

        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break
            else:
                raise