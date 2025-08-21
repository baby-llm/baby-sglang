"""
I/O structures for inter-process communication in baby-sglang.

Defines request and response objects passed between components.
Based on SGLang's io_struct but simplified.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_path: str
    tokenizer_path: str
    trust_remote_code: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

@dataclass
class SamplingParams:
    """Parameters for sampling."""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    # Add other sampling params as needed

@dataclass
class BatchRequest:
    """A batch of requests for the model."""
    prompts: List[str]
    sampling_params: SamplingParams

@dataclass  
class GenerateRequest:
    """Single generation request."""
    prompt: str
    sampling_params: SamplingParams

@dataclass
class GenerateResponse:
    """Response for a single generation request."""
    output: str

@dataclass
class BatchResponse:
    """The response for a batch of requests."""
    outputs: List[str]
