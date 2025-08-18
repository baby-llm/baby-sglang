"""
I/O structures for inter-process communication in baby-sglang.

Defines request and response objects passed between components.
Based on SGLang's io_struct but simplified.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class GenerateRequest:
    """
    Request for text generation.
    
    Simplified version of SGLang's GenerateReqInput.
    """
    request_id: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_strings: Optional[List[str]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass 
class GenerateResponse:
    """
    Response from text generation.
    
    Contains the generated text and metadata.
    """
    request_id: str
    text: str
    tokens: List[int]
    finish_reason: str = "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class BatchRequest:
    """
    Batch of requests for efficient processing.
    """
    requests: List[GenerateRequest]
    batch_id: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class BatchResponse:
    """
    Batch response containing multiple generations.
    """
    responses: List[GenerateResponse]
    batch_id: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SchedulerState:
    """
    Current state of the scheduler for monitoring.
    """
    waiting_queue_size: int
    running_batch_size: int
    memory_usage: float
    cache_hit_rate: float
    requests_per_second: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ModelConfig:
    """
    Model configuration parameters.
    """
    model_path: str
    max_seq_len: int = 2048
    max_batch_size: int = 32
    dtype: str = "float16"
    device: str = "cuda"
    # TODO: Add more model-specific configs
    # - attention type (MHA/GQA/MQA)
    # - vocab size
    # - hidden dimension
    # - number of layers
    additional_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_config is None:
            self.additional_config = {}


# Message types for inter-process communication
@dataclass
class ZMQMessage:
    """
    Base class for ZMQ messages between processes.
    """
    message_type: str
    payload: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()