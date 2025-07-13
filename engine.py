import multiprocessing as mp
from typing import Dict, List, Optional, Union

from .managers.detokenizer_manager import run_detokenizer
from .managers.scheduler import run_scheduler
from .utils.utils import (
    get_server_name,
    get_all_server_names,
    get_and_register_new_filename,
    send_req_to_scheduler,
    Req,
)
from transformers import AutoTokenizer

class SGLangEngine:
    """
    SGLangEngine 是 nano-sglang 推理服务的总入口。
    它负责：
    1.  初始化模型，包括加载权重和设置分布式环境（尽管在 nano-sglang 中被简化）。
    2.  启动并管理三个核心的后台服务进程：
        *   TokenizerManager: 负责接收外部请求，将其转换为 token IDs。
        *   SchedulerManager: 调度引擎的核心，负责管理请求队列、内存池，并执行批处理（batching）和模型推理。
        *   DetokenizerManager: 负责将推理完成的 token IDs 转换回文本，并返回给客户端。
    3.  提供一个简单的 API (`generate`)，用于向系统提交新的推理请求。
    4.  通过 ZeroMQ (ZMQ) 管理进程间的通信。

    这个类的设计受到了原始 SGLang `srt.entrypoints.engine.SglangEngine` 的启发，
    但在我们的 `nano-sglang` 中进行了大幅简化，以专注于核心功能。
    """ 
    def __init__(self, model_path: str, tokenizer_path: str, mem_fraction_static: float, 
                 tp_size: int = 1):
        """
        初始化引擎。

        Args:
            model_path (str): 存放模型权重的路径。
            tokenizer_path (str): 存放 tokenizer 文件的路径。
            mem_fraction_static (float): KV Cache 使用的静态内存比例。
            tp_size (int): Tensor Parallelism 的大小。在 nano-sglang 中，我们简化为 1，
                           不实现真正的张量并行。
        """
        # 在 nano-sglang 中，我们假设 tp_size 始终为 1。
        # 模型的加载和分布式设置将被简化，直接在 Scheduler 进程中处理。
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.mem_fraction_static = mem_fraction_static
        self.tp_size = tp_size

        # 在主进程中加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 初始化进程间通信所需的 ZMQ context 和 queues
        # 这里的 "spawn" 是 multiprocessing 的一种启动新进程的方式。
        # "spawn" 会通过启动一个全新的 Python 解释器进程，然后在新进程中导入主模块并运行指定的代码。
        # 这样可以避免父进程中的资源（如文件描述符、内存等）被子进程继承，提升跨平台兼容性（如在 Windows 上必须使用 "spawn"）。
        # 但与 "fork" 相比，"spawn" 启动速度略慢，且要求所有被子进程执行的对象都能被 pickle 序列化。
        self.context = mp.get_context("spawn")
        self.req_queue = self.context.Queue()
        self.init_server_name_and_queues()

        # 启动核心服务进程
        self.procs = [] 
        self.start_processes()

    def init_server_name_and_queues(self):
        """
        初始化用于进程间通信的服务名称和 ZMQ sockets。
        每个服务 (tokenizer, scheduler, detokenizer) 都有一个唯一的 ZMQ 地址。
        """
        # 在真实的 SGLang 中，这里会使用更复杂的机制来分配端口和地址。
        # 在 nano-sglang 中，我们使用固定的、基于文件的命名方案来简化。
        self.scheduler_server_name = get_server_name("scheduler")
        self.detokenizer_server_name = get_server_name("detokenizer")
        
        # 确保在启动前清理掉旧的 ZMQ socket 文件
        for name in get_all_server_names():
            get_and_register_new_filename(name)
        
    def start_processes(self):
        """
        根据配置，创建并启动 Tokenizer, Scheduler, 和 Detokenizer 进程。
        """
        # Tokenizer 功能已集成到主进程中，不再需要独立的子进程。

        # 2. Scheduler 进程
        proc = self.context.Process(
            target=run_scheduler,
            args=(
                self.model_path,
                self.scheduler_server_name,
                self.detokenizer_server_name,
                self.mem_fraction_static,
                self.tp_size,
            ),
        )
        proc.start()
        self.procs.append(proc)

        # 3. Detokenizer 进程
        proc = self.context.Process(
            target=run_detokenizer,
            args=(
                self.tokenizer_path,
                self.detokenizer_server_name,
                self.req_queue
            ),
        )
        proc.start()
        self.procs.append(proc)
        
    def generate(self,
                 prompt: str,
                 sampling_params: Dict,
                 request_id: str) -> Dict:
        """
        向推理引擎提交一个生成请求，并阻塞等待直到收到最终结果。

        Args:
            prompt (str): 输入的提示文本。
            sampling_params (Dict): 控制生成的采样参数，例如 temperature, top_p 等。
            request_id (str): 唯一标识此请求的 ID。

        Returns:
            Dict: 一个包含生成结果的字典, e.g., {"request_id": "...", "text": "..."}
        """
        # 1. 使用主进程中的 tokenizer 进行编码
        input_ids = self.tokenizer.encode(prompt)

        # 2. 创建 Req 对象
        req = Req(
            request_id=request_id,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )

        # 3. 将 Req 对象直接发送给 Scheduler 进程
        send_req_to_scheduler(self.scheduler_server_name, req)

        # 4. 阻塞等待，从返回队列中获取结果
        # 这个循环会一直等待，直到 Detokenizer 进程将此 request_id 的结果放入队列
        while True:
            # 使用 get() 方法阻塞地从队列中获取项目
            # 可以设置 timeout 参数以避免无限等待
            result = self.req_queue.get()
            if result["request_id"] == request_id:
                return result
    
    def shutdown(self):
        """
        优雅地关闭所有后台服务进程。
        """
        print("\nShutting down SGLangEngine...")
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
                # 在真实应用中，join() 可能会更好，但 terminate() 更直接
                # proc.join(timeout=5)
        print("All processes have been terminated.")