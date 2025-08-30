"""
Generation engine for baby-sglang.

Integrates all components into a complete generation loop:
- StaticBatchScheduler for request management
- BabyQwen2ForCausalLM for model inference
- Memory pools for KV caching
- Sampling and generation control

Based on SGLang's design but simplified for MVP static batching.
"""

import logging
import time
from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F

from managers.batch_scheduler import (
    StaticBatchScheduler, 
    SimplifiedRequest, 
    SimplifiedSamplingParams,
    RequestStatus
)
from model_executor.qwen2 import BabyQwen2ForCausalLM, SimplifiedForwardBatch
from mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from utils.tokenizer import BabyTokenizer

logger = logging.getLogger(__name__)


class GenerationEngine:
    """
    Complete generation engine integrating all baby-sglang components.
    
    Phase 6: Implements the full generation loop:
    1. Accept requests
    2. Schedule prefill batches  
    3. Run iterative decode until completion
    4. Handle sampling and termination
    5. Clean up resources
    
    Follows SGLang's architecture but simplified for MVP.
    """
    
    def __init__(
        self,
        model: BabyQwen2ForCausalLM,
        model_config: Dict[str, Any],
        tokenizer_path: Optional[str] = None,
        max_batch_size: int = 32,
        max_total_tokens: int = 32768,
        max_context_len: int = 2048,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval() # Sets model to evaluation mode: disables dropout, fixes batchnorm for inference
        self.model_config = model_config
        self.device = device
        
        # Initialize tokenizer (Real tokenizer only)
        if tokenizer_path is None:
            tokenizer_path = "Qwen/Qwen2-0.5B-Instruct"  # Default model
        
        self.tokenizer = BabyTokenizer(model_path_or_name=tokenizer_path)
        logger.info(f"Tokenizer loaded: {tokenizer_path}")
        
        # Initialize memory pools (SGLang pattern)
        self.req_to_token_pool = ReqToTokenPool(
            size=max_batch_size,
            max_context_len=max_context_len, 
            device=device
        )
        
        # Extract model dimensions for KV pool
        hidden_size = model_config['hidden_size']
        num_attention_heads = model_config['num_attention_heads'] 
        num_key_value_heads = model_config.get('num_key_value_heads', num_attention_heads)
        num_hidden_layers = model_config['num_hidden_layers']
        head_dim = hidden_size // num_attention_heads
        
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_tokens,
            dtype=torch.float16,  # Match model dtype
            head_num=num_key_value_heads,
            head_dim=head_dim,
            layer_num=num_hidden_layers,
            device=device
        )
        
        # Initialize scheduler
        self.scheduler = StaticBatchScheduler(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            max_batch_size=max_batch_size,
            device=device
        )
        
        # Generation state
        self.is_running = False
        self.request_counter = 0
        
        logger.info(f"GenerationEngine initialized on {device}")
        logger.info(f"Model config: {num_hidden_layers} layers, {num_attention_heads} heads, {hidden_size} hidden_size")
        logger.info(f"Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SimplifiedSamplingParams] = None,
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.
        
        Uses real tokenizer for encoding/decoding.
        """
        if sampling_params is None:
            sampling_params = SimplifiedSamplingParams()
        
        # Step 1: Create and add requests
        requests = []
        for prompt in prompts:
            # Real tokenization
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            logger.debug(f"Tokenization: '{prompt[:50]}...' -> {len(input_ids)} tokens")
            
            request = SimplifiedRequest(
                request_id=f"req_{self.request_counter}",
                input_text=prompt,
                input_ids=input_ids,
                sampling_params=sampling_params,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            success = self.scheduler.add_request(request)
            if success:
                requests.append(request)
                self.request_counter += 1
            else:
                logger.warning(f"Failed to add request for prompt: {prompt[:50]}...")
        
        if not requests:
            logger.error("No requests could be scheduled")
            return []
        
        try:
            # Step 2: Run generation loop
            self._run_generation_loop()
            
            # Step 3: Extract results
            results = []
            for request in requests:
                if request.status == RequestStatus.FINISHED:
                    # Real detokenization
                    output_text = self.tokenizer.decode(
                        request.output_ids, 
                        skip_special_tokens=True
                    )
                    full_text = request.input_text + output_text
                    logger.debug(f"Detokenization: {len(request.output_ids)} tokens -> '{output_text[:50]}...'")
                    results.append(full_text)
                else:
                    logger.warning(f"Request {request.request_id} did not finish properly")
                    results.append(request.input_text + " [INCOMPLETE]")
            
            return results
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _run_generation_loop(self):
        """
        Main generation loop: prefill -> decode until all requests finish.
        
        Phase 6: Implements SGLang's prefill-decode cycle.
        """
        logger.info("Starting generation loop")
        
        # Step 1: Prefill phase
        prefill_batch = self.scheduler.schedule_prefill_batch()
        if prefill_batch is None:
            logger.error("Failed to schedule prefill batch")
            return
        
        # Run prefill forward pass
        with torch.no_grad():
            logits = self.model(
                input_ids=prefill_batch.input_ids,
                positions=prefill_batch.positions,
                forward_batch=prefill_batch,
            )
        
        # Sample tokens from prefill logits (only the last token of each sequence)
        running_requests = [req for req in self.scheduler.running_requests if req.needs_generation]
        prefill_new_tokens = self._sample_from_prefill_logits(logits, prefill_batch, running_requests)
        
        # Update requests with first generated tokens and change status to DECODE
        self.scheduler.update_requests_with_tokens(running_requests, prefill_new_tokens)
        
        # NOW change status to DECODE since prefill is actually completed
        for req in running_requests:
            if req.status == RequestStatus.PREFILL:
                req.status = RequestStatus.DECODE
        
        logger.info(f"Prefill completed for {len(running_requests)} requests")
        
        # Step 2: Decode loop
        max_decode_steps = max(req.sampling_params.max_new_tokens for req in running_requests)
        
        for step in range(max_decode_steps - 1):  # -1 because we already generated one token
            # Check if any requests still need generation
            active_requests = [req for req in self.scheduler.running_requests if req.needs_generation]
            if not active_requests:
                logger.info("All requests finished generation")
                break
            
            # Schedule decode batch
            decode_batch = self.scheduler.schedule_decode_batch()
            if decode_batch is None:
                logger.warning(f"Failed to schedule decode batch at step {step}")
                break
            
            # Run decode forward pass
            with torch.no_grad():
                logits = self.model(
                    input_ids=decode_batch.input_ids,
                    positions=decode_batch.positions,
                    forward_batch=decode_batch,
                )
            
            # Sample new tokens
            decode_requests = [req for req in active_requests if req.is_decode_ready]
            new_tokens = self._sample_from_decode_logits(logits, decode_batch, decode_requests)
            
            # Update requests
            self.scheduler.update_requests_with_tokens(decode_requests, new_tokens)
            
            logger.debug(f"Decode step {step+1}: generated tokens for {len(decode_requests)} requests")
        
        # Step 3: Cleanup
        self.scheduler.cleanup_finished_requests()
        
        finished_count = len([req for req in self.scheduler.running_requests 
                             if req.status == RequestStatus.FINISHED])
        logger.info(f"Generation loop completed: {finished_count} requests finished")
    
    def _sample_from_prefill_logits(
        self, 
        logits: torch.Tensor, 
        batch: SimplifiedForwardBatch,
        requests: List[SimplifiedRequest]
    ) -> List[int]:
        """
        Sample tokens from prefill logits with per-request sampling parameters.
        
        Phase 6: Extract logits for the last token of each sequence and sample.
        """
        batch_tokens = []
        offset = 0
        
        for i, seq_len in enumerate(batch.seq_lens):
            # Get logits for the last token of this sequence
            last_token_logits = logits[offset + seq_len - 1]  # [vocab_size]
            
            # Sample token with request-specific parameters
            request = requests[i]
            token_id = self._sample_token(last_token_logits, request.sampling_params)
            batch_tokens.append(token_id)
            
            offset += seq_len.item()
        
        return batch_tokens
    
    def _sample_from_decode_logits(
        self, 
        logits: torch.Tensor, 
        batch: SimplifiedForwardBatch,
        requests: List[SimplifiedRequest]
    ) -> List[int]:
        """
        Sample tokens from decode logits with per-request sampling parameters.
        
        Phase 6: Each position corresponds to one request's new token.
        """
        batch_tokens = []
        
        for i in range(batch.batch_size):
            token_logits = logits[i]  # [vocab_size]
            request = requests[i]
            token_id = self._sample_token(token_logits, request.sampling_params)
            batch_tokens.append(token_id)
        
        return batch_tokens
    
    def _sample_token(
        self, 
        logits: torch.Tensor, 
        sampling_params: SimplifiedSamplingParams
    ) -> int:
        """
        Sample a single token from logits with temperature and top_p sampling.
        
        Phase 6: Complete sampling implementation with temperature and nucleus sampling.
        """
        if sampling_params.is_greedy:
            # Greedy sampling (deterministic)
            token_id = torch.argmax(logits, dim=-1)
            return token_id.item()
        
        # Apply temperature scaling
        if sampling_params.temperature != 1.0:
            logits = logits / sampling_params.temperature
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top_p (nucleus) sampling if needed
        if sampling_params.top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find the cutoff index where cumulative probability exceeds top_p
            cutoff_idx = torch.searchsorted(cumulative_probs, sampling_params.top_p, right=False)
            cutoff_idx = max(1, cutoff_idx.item())  # Keep at least one token
            
            # Zero out probabilities beyond the cutoff
            sorted_probs[cutoff_idx:] = 0.0
            
            # Renormalize the kept probabilities
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            # Sample from the filtered distribution
            sampled_sorted_idx = torch.multinomial(sorted_probs, 1).item()
            token_id = sorted_indices[sampled_sorted_idx].item()
        else:
            # Sample from full probability distribution
            token_id = torch.multinomial(probs, 1).item()
            
        return token_id
    
    
    def get_tokenizer(self) -> BabyTokenizer:
        """Get the tokenizer instance (Phase 7)."""
        return self.tokenizer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        tokenizer_stats = self.tokenizer.get_special_tokens_dict()
        
        return {
            "scheduler": scheduler_stats,
            "model_device": str(self.device),
            "total_requests": self.request_counter,
            "tokenizer": {
                "vocab_size": self.tokenizer.get_vocab_size(),
                "special_tokens": tokenizer_stats,
            }
        }


# Convenience function for easy testing
def create_engine_from_hf(
    hf_model_path: str,
    device: str = "auto", 
    max_batch_size: int = 32,
    max_total_tokens: int = 32768,
    max_context_len: int = 2048,
) -> GenerationEngine:
    """
    Create GenerationEngine from HuggingFace Qwen2 model.
    
    Automatically loads model configuration and weights from HF checkpoint.
    Supports all Qwen2 model sizes (0.5B, 1.5B, 7B, etc.).
    
    Args:
        hf_model_path: Path to HF model directory
        device: Device to use ("auto", "cpu", "cuda", "mps")
        max_batch_size: Maximum batch size
        max_total_tokens: Maximum total tokens in memory
        max_context_len: Maximum context length per request
        
    Returns:
        Configured GenerationEngine with loaded weights
    """
    import json
    import os
    
    # Auto-detect device if needed
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps" 
        else:
            device = "cpu"
    
    logger.info(f"Creating engine from HF model: {hf_model_path}")
    logger.info(f"Using device: {device}")
    
    # Load HF model configuration
    config_path = os.path.join(hf_model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {hf_model_path}")
    
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
    
    # Create model config object
    class HFConfig:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
    
    config = HFConfig(hf_config)
    
    # Create model
    from model_executor.qwen2 import BabyQwen2ForCausalLM
    model = BabyQwen2ForCausalLM(config)
    
    # Load HF weights  
    model.load_weights_from_hf(hf_model_path, device)
    
    # Create tokenizer path (same as model path)
    tokenizer_path = hf_model_path
    
    # Create engine
    engine = GenerationEngine(
        model=model,
        model_config=hf_config,
        tokenizer_path=tokenizer_path,
        max_batch_size=max_batch_size,
        max_total_tokens=max_total_tokens,
        max_context_len=max_context_len,
        device=device,
    )
    
    logger.info(f"Successfully created engine from {hf_model_path}")
    return engine


def create_test_engine(
    device: str = "auto", 
    tokenizer_path: Optional[str] = None
) -> GenerationEngine:
    """
    Create a test GenerationEngine with minimal configuration.
    
    Uses mock model config for testing when no real model is available.
    Supports auto device detection including MPS.
    """
    # Auto-detect device if needed
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    # Mock model config (would come from real model in production)
    model_config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_key_value_heads': 8, 
        'num_hidden_layers': 6,
        'intermediate_size': 1024,
        'vocab_size': 1000,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'max_position_embeddings': 2048,
    }
    
    # Create mock model config object
    class MockConfig:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
    
    config = MockConfig(model_config)
    
    # Initialize model
    from model_executor.qwen2 import BabyQwen2ForCausalLM
    model = BabyQwen2ForCausalLM(config)
    
    # Create engine with tokenizer support (Phase 7)
    engine = GenerationEngine(
        model=model,
        model_config=model_config,
        tokenizer_path=tokenizer_path,
        max_batch_size=4,
        max_total_tokens=1024,
        max_context_len=256,
        device=device,
    )
    
    return engine


# Phase 7 Summary:
# ================
#
# Real tokenizer integration completed:
#
# 1. BabyTokenizer integration:
#    - Uses tokenizers library for real tokenization
#    - Automatic fallback to mock for testing
#    - Special tokens support (eos, pad, bos, unk)
#    - Local and HuggingFace model support
#
# 2. GenerationEngine updates:
#    - Real tokenizer as instance variable
#    - encode/decode using BabyTokenizer
#    - Backward compatibility with mock mode
#    - Enhanced stats with tokenizer info
#
# 3. Phase 7 features:
#    - Real tokenization: encode(text) -> List[int]
#    - Real detokenization: decode(ids) -> str
#    - Special token handling from config
#    - Graceful fallback if tokenizer fails
#    - Enhanced debugging and logging
#
# 4. Ready for Phase 8:
#    - Complete testing and validation
#    - Unit tests for all components
#    - Integration tests for full pipeline
#    - Performance sanity checks