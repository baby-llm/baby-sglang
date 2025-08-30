"""
Static batch scheduler for baby-sglang.

Manages request lifecycle, memory allocation, and batch construction.
Based on SGLang's ScheduleBatch design but simplified for static batching MVP.

References:
- sglang/srt/managers/schedule_batch.py::Req
- sglang/srt/managers/schedule_batch.py::ScheduleBatch  
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import torch

from mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from model_executor.qwen2 import SimplifiedForwardBatch

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Request processing status (SGLang-aligned)."""
    WAITING = "waiting"      # Waiting to be scheduled
    PREFILL = "prefill"      # In prefill phase  
    DECODE = "decode"        # In decode phase
    FINISHED = "finished"    # Generation completed
    ABORTED = "aborted"      # Request aborted


@dataclass
class SimplifiedSamplingParams:
    """Simplified sampling parameters (SGLang-compatible)."""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    # For MVP: only support greedy (temp=0) and basic sampling
    
    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0


class SimplifiedRequest:
    """
    Simplified version of SGLang's Req class for MVP.
    
    Maintains essential fields for static batching while
    preserving SGLang's request lifecycle pattern.
    """
    
    def __init__(
        self,
        request_id: str,
        input_text: str,
        input_ids: List[int],
        sampling_params: SimplifiedSamplingParams,
        eos_token_id: Optional[int] = None,
    ):
        # Basic request info (SGLang pattern)
        self.request_id = request_id
        self.input_text = input_text
        self.input_ids = input_ids
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id
        
        # Generation state
        self.output_ids: List[int] = []
        self.status = RequestStatus.WAITING
        
        # Memory management (SGLang-aligned)
        self.req_pool_idx: Optional[int] = None  # Index in req_to_token_pool
        self.allocated_tokens: Optional[torch.Tensor] = None  # Token indices in kv_pool
        
        # Position tracking
        self.prompt_len = len(input_ids)
        self.seq_len = self.prompt_len  # Current total length (prompt + generated)
        
        # Flags
        self.finished = False
        self.aborted = False
        
    @property
    def is_prefill_ready(self) -> bool:
        """Check if request is ready for prefill."""
        return self.status == RequestStatus.WAITING
        
    @property 
    def is_decode_ready(self) -> bool:
        """Check if request is ready for decode."""
        return self.status == RequestStatus.DECODE
    
    @property
    def needs_generation(self) -> bool:
        """Check if request needs more generation."""
        if self.finished or self.aborted:
            return False
        return len(self.output_ids) < self.sampling_params.max_new_tokens
    
    def add_output_token(self, token_id: int):
        """Add generated token and update state."""
        self.output_ids.append(token_id)
        self.seq_len += 1
        
        # Check finish conditions
        should_finish = False
        
        # Check max tokens limit
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            should_finish = True
        
        # Check EOS token (early termination)
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            should_finish = True
        
        if should_finish:
            self.finished = True
            self.status = RequestStatus.FINISHED
    
    def get_all_token_ids(self) -> List[int]:
        """Get concatenated input + output tokens."""
        return self.input_ids + self.output_ids


class StaticBatchScheduler:
    """
    Static batch scheduler for baby-sglang.
    
    Manages the complete lifecycle of requests:
    1. Accept new requests  
    2. Allocate memory resources
    3. Build batches for prefill/decode
    4. Release resources when done
    
    Based on SGLang's ScheduleBatch pattern but simplified for static batching.
    """
    
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
        max_batch_size: int = 32,
        device: str = "cuda",
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.max_batch_size = max_batch_size
        self.device = device
        
        # Request management
        self.waiting_requests: List[SimplifiedRequest] = []
        self.running_requests: List[SimplifiedRequest] = []  # prefill + decode
        self.finished_requests: List[SimplifiedRequest] = []
        
        # Request mapping for fast lookup
        self.request_map: Dict[str, SimplifiedRequest] = {}
        
        logger.info(f"StaticBatchScheduler initialized: max_batch_size={max_batch_size}")
    
    def add_request(self, request: SimplifiedRequest) -> bool:
        """Add new request to scheduler."""
        if len(self.waiting_requests) >= self.max_batch_size:
            logger.warning(f"Batch size limit reached, rejecting request {request.request_id}")
            return False
        
        self.waiting_requests.append(request)
        self.request_map[request.request_id] = request
        
        return True
    
    def schedule_prefill_batch(self) -> Optional[SimplifiedForwardBatch]:
        """
        Schedule a batch of requests for prefill phase.
        
        SGLang pattern: allocate memory, prepare batch tensors, update pools.
        """
        if not self.waiting_requests:
            return None
        
        # For static batching: process all waiting requests together
        batch_requests = self.waiting_requests[:]
        
        try:
            # Step 1: Allocate request slots (SGLang pattern)
            req_pool_indices = self.req_to_token_pool.alloc(len(batch_requests))
            if req_pool_indices is None:
                logger.error("No available request slots for prefill batch")
                return None
            
            # Step 2: Calculate total tokens needed
            total_tokens = sum(req.prompt_len for req in batch_requests)
            
            # Step 3: Allocate token slots
            out_cache_loc = self.token_to_kv_pool.alloc(total_tokens)
            if out_cache_loc is None:
                # Rollback request slots
                self.req_to_token_pool.free(req_pool_indices)
                logger.error("No available token slots for prefill batch")
                return None
            
            # Step 4: Update request states and memory assignments
            token_offset = 0
            for i, req in enumerate(batch_requests):
                req.req_pool_idx = req_pool_indices[i]
                req.status = RequestStatus.PREFILL
                
                # Allocate tokens for this request
                req_tokens = out_cache_loc[token_offset:token_offset + req.prompt_len]
                req.allocated_tokens = req_tokens
                
                # Update req_to_token_pool mapping (SGLang pattern)
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :req.prompt_len] = req_tokens
                
                token_offset += req.prompt_len
            
            # Step 5: Build batch tensors  
            forward_batch = self._build_prefill_batch(
                batch_requests, req_pool_indices, out_cache_loc
            )
            
            # Step 6: Move requests to running state  
            self.running_requests.extend(batch_requests)
            self.waiting_requests.clear()
            
            logger.info(f"Scheduled prefill batch: {len(batch_requests)} requests, {total_tokens} tokens")
            return forward_batch
            
        except Exception as e:
            logger.error(f"Failed to schedule prefill batch: {e}")
            # Cleanup on failure
            self._cleanup_failed_requests(batch_requests)
            return None
    
    def schedule_decode_batch(self) -> Optional[SimplifiedForwardBatch]:
        """
        Schedule decode batch for all running requests.
        
        SGLang pattern: each request generates one token, extend allocations if needed.
        """
        decode_ready = [req for req in self.running_requests 
                       if req.is_decode_ready and req.needs_generation]
        
        if not decode_ready:
            return None
            
        try:
            # Step 1: Extend token allocations for new tokens
            extended_tokens = []
            for req in decode_ready:
                # Allocate one token for this request
                new_token_loc = self.token_to_kv_pool.alloc(1)
                if new_token_loc is None:
                    logger.error(f"Cannot allocate token for request {req.request_id}")
                    return None
                    
                extended_tokens.append(new_token_loc[0])
                
                # Update req_to_token_pool for new position (SGLang pattern)
                current_pos = req.seq_len  # Position for new token
                self.req_to_token_pool.req_to_token[req.req_pool_idx, current_pos] = new_token_loc[0]
            
            # Step 2: Build decode batch
            forward_batch = self._build_decode_batch(decode_ready, extended_tokens)
            
            return forward_batch
            
        except Exception as e:
            logger.error(f"Failed to schedule decode batch: {e}")
            return None
    
    def _build_prefill_batch(
        self, 
        requests: List[SimplifiedRequest], 
        req_pool_indices: List[int], 
        out_cache_loc: torch.Tensor
    ) -> SimplifiedForwardBatch:
        """Build SimplifiedForwardBatch for prefill (SGLang pattern)."""
        
        # Flatten all input tokens
        input_ids = []
        seq_lens = []
        for req in requests:
            input_ids.extend(req.input_ids)
            seq_lens.append(req.prompt_len)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.long, device=self.device)
        
        # Use factory method for proper batch creation
        return SimplifiedForwardBatch.create_prefill_batch(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
    
    def _build_decode_batch(
        self,
        requests: List[SimplifiedRequest],
        extended_tokens: List[int]
    ) -> SimplifiedForwardBatch:
        """Build SimplifiedForwardBatch for decode (SGLang pattern)."""
        
        # In decode mode: each request contributes exactly one NEW token
        # The input_ids for decode should be the newly generated tokens from previous step
        # This is a simplified implementation - in real usage, these would come from sampling
        
        input_ids = []
        req_pool_indices = []
        seq_lens = []
        
        for req in requests:
            # For decode, we need the newly generated token as input
            # In the actual generation loop, this would be the token just sampled
            # For now, use the last generated token (placeholder logic)
            if req.output_ids:
                # Use the last generated token as input for next decode step
                new_input_token = req.output_ids[-1] 
            else:
                # This shouldn't happen in normal decode phase, but fallback to last input token
                new_input_token = req.input_ids[-1]
                
            input_ids.append(new_input_token)
            req_pool_indices.append(req.req_pool_idx)
            seq_lens.append(req.seq_len)  # Current sequence length (not including the new token yet)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        out_cache_loc = torch.tensor(extended_tokens, dtype=torch.long, device=self.device)
        
        # Use factory method
        return SimplifiedForwardBatch.create_decode_batch(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
    
    def update_requests_with_tokens(self, requests: List[SimplifiedRequest], new_tokens: List[int]):
        """Update requests with newly generated tokens."""
        for req, token in zip(requests, new_tokens):
            req.add_output_token(token)
    
    def cleanup_finished_requests(self):
        """Clean up finished requests and free memory (SGLang pattern)."""
        to_remove = []
        
        for req in self.running_requests:
            if not req.needs_generation:
                # Free memory resources
                if req.req_pool_idx is not None:
                    self.req_to_token_pool.free(req.req_pool_idx)
                if req.allocated_tokens is not None:
                    self.token_to_kv_pool.free(req.allocated_tokens)
                
                # Move to finished
                req.status = RequestStatus.FINISHED
                self.finished_requests.append(req)
                to_remove.append(req)
                
        
        # Remove from running
        for req in to_remove:
            self.running_requests.remove(req)
    
    def _cleanup_failed_requests(self, requests: List[SimplifiedRequest]):
        """Cleanup resources for failed requests."""
        for req in requests:
            if req.req_pool_idx is not None:
                self.req_to_token_pool.free(req.req_pool_idx)
                req.req_pool_idx = None
            if req.allocated_tokens is not None:
                self.token_to_kv_pool.free(req.allocated_tokens)
                req.allocated_tokens = None
            req.status = RequestStatus.ABORTED
    
    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics."""
        return {
            "waiting": len(self.waiting_requests),
            "running": len(self.running_requests),
            "finished": len(self.finished_requests),
            "req_pool_available": self.req_to_token_pool.available_size(),
            "token_pool_available": self.token_to_kv_pool.available_size(),
        }


