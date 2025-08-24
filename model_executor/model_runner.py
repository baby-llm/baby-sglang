"""
Model runner for baby-sglang.

Handles model loading, forward passes, and inference execution.
Simplified version of SGLang's ModelRunner focusing on core functionality.
"""

import logging
from typing import List, Optional, Tuple, Any

import torch
from managers.io_struct import ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

class ModelRunner:
    """
    Simplified model runner for inference execution.
    
    Responsibilities:
    - Load and manage the language model
    - Execute forward passes for batched requests
    - Handle attention computations with KV caching
    - Manage GPU memory and computation
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = model_config.device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            torch_dtype=model_config.dtype,
            trust_remote_code=model_config.trust_remote_code,
        ).to(self.device).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_kv_cache(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initializes a basic KV cache for a batch."""
        # Return None to let transformers handle cache initialization
        return None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Performs a forward pass with a simplified KV cache.
        """
        if kv_cache is not None:
             # Decode phase - use existing cache
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
            )
        else:
            # Prefill phase - no cache
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )

        logits = model_outputs.logits
        new_kv_cache = model_outputs.past_key_values
        return logits, new_kv_cache