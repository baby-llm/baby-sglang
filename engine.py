"""
Main engine entry point for baby-sglang inference.

This is the simplified implementation of SGLang's Engine class,
focusing on core functionality.
"""

import logging
from typing import List, Union, Optional

import torch
from managers.io_struct import BatchRequest, BatchResponse, ModelConfig, SamplingParams
from model_executor.model_runner import ModelRunner
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

class Engine:
    """
    Simplified SGLang Engine for baby-sglang.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_runner = ModelRunner(model_config)
        self.tokenizer: PreTrainedTokenizerBase = self.model_runner.tokenizer
        self.device = self.model_runner.device

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> BatchResponse:
        """
        Generates text from a batch of prompts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        # 1. Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        batch_size, prompt_len = input_ids.shape

        # 2. Prefill step
        logits, kv_cache = self.model_runner.forward(
            input_ids=input_ids,
            kv_cache=None,
            attention_mask=attention_mask,
        )
        
        next_token_ids = self._sample_tokens(logits[:, -1, :], sampling_params)
        # Initialize 'finished' based on first sampled token
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        if eos_id is None:
            eos_id = int(self.tokenizer.vocab_size - 1)
        finished = (next_token_ids == eos_id)
        output_ids = torch.cat([input_ids, next_token_ids.unsqueeze(1)], dim=1)
 
        # 3. Decode loop
        current_length = prompt_len
        
        for step in range(sampling_params.max_new_tokens - 1):
            if torch.all(finished):
                break
                
            # Recreate attention mask for the decode step
            # The new mask should cover the prompt and all generated tokens
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device)], dim=1)
            
            logits, kv_cache = self.model_runner.forward(
                input_ids=next_token_ids.unsqueeze(1),
                kv_cache=kv_cache,
                attention_mask=attention_mask,
            )
            
            # Apply sampling
            sampled_next = self._sample_tokens(logits[:, -1, :], sampling_params)
            eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
            if eos_id is None:
                eos_id = int(self.tokenizer.vocab_size - 1)
            eos_tensor = torch.full_like(sampled_next, eos_id)
            # Force finished sequences to stay at EOS
            next_token_ids = torch.where(finished, eos_tensor, sampled_next)
            # Update finished mask
            finished = finished | (next_token_ids == eos_id)
            # Append to outputs
            output_ids = torch.cat([output_ids, next_token_ids.unsqueeze(1)], dim=1)
            current_length += 1
        
        # 4. Detokenize results
        outputs = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        return BatchResponse(outputs=outputs)
    
    def _sample_tokens(self, logits: torch.Tensor, sampling_params: SamplingParams) -> torch.Tensor:
        """
        Sample tokens from logits using temperature and top_p.
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            sampling_params: Sampling parameters
            
        Returns:
            Sampled token IDs [batch_size]
        """
        
        # Greedy decoding when temperature <= 0 or top_p == 0
        if getattr(sampling_params, "temperature", 1.0) is not None and sampling_params.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        if getattr(sampling_params, "top_p", 1.0) == 0:
            return torch.argmax(logits, dim=-1)
 
        # Apply temperature
        temperature = float(getattr(sampling_params, "temperature", 1.0))
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
            
        # Apply top_p (nucleus sampling)
        top_p = float(getattr(sampling_params, "top_p", 1.0))
        if top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
 
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
 
            # Map mask back to original indices
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
            indices_to_remove = indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens
