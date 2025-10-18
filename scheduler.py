from __future__ import annotations
from typing import List, Union, Optional, Tuple
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

from forward_batch import SimplifiedForwardBatch
from memory_pool import ReqToTokenPool, MHATokenToKVPool
from sample import SamplingParams
from sample import sample_next_ids
from request import Request
from radix_cache import RadixCache

# Configure unified debug logger (shared with attention)
_attn_logger = logging.getLogger("baby_sgl.attn")
if not _attn_logger.handlers:
    _attn_logger.setLevel(logging.DEBUG)
    os.makedirs("baby-sgl/debug", exist_ok=True)
    fh = logging.FileHandler("baby-sgl/debug/attn_debug.log")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    _attn_logger.addHandler(fh)


def _sched_log_tensor_stats(name: str, t: torch.Tensor):
    try:
        _attn_logger.debug(
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
        )
        if t.numel() > 0 and t.dtype.is_floating_point:
            _attn_logger.debug(
                f"{name}: min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}"
            )
        elif t.numel() > 0 and t.dtype in (torch.int32, torch.int64):
            _attn_logger.debug(
                f"{name}: min={int(t.min().item())} max={int(t.max().item())}"
            )
    except Exception as e:
        _attn_logger.warning(f"_sched_log_tensor_stats error for {name}: {e}")


class Scheduler:
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self.model = model

        self.max_total_tokens = token_pool_size = int(
            os.getenv("BABYSGL_MAX_TOTAL_TOKENS", "131072")
        )  # 128 K
        self.max_requests = req_pool_size = int(os.getenv("BABYSGL_MAX_REQUESTS", "32"))

        p = next(model.parameters())
        self.device = p.device
        dtype = p.dtype
        cfg = model.config

        self.req_to_token_pool = ReqToTokenPool(
            size=req_pool_size,
            max_context_len=token_pool_size // req_pool_size,
            device=str(self.device),
        )

        # TODO(@huangyz) replace MHATokenToKVPool with an abstract class to support different types of attention
        self.token_to_kv_pool = MHATokenToKVPool(
            size=token_pool_size,
            dtype=dtype,
            head_num=cfg.num_key_value_heads,  # using num_kv_heads as head_num for GQA
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            layer_num=cfg.num_hidden_layers,
            device=str(self.device),
        )

        self.waiting_queue: List[Request] = []
        # self.running_batch: List[Request] = []
        self.decoding_batch: List[Request] = []
        self.finished_requests: List[Request] = []

        self.r_init = 0.5
        self.min_r = 0.1
        self.decay_step = (self.r_init - self.min_r) / 50
        self.est_new_token_ratio = self.r_init
        self.retract_decode_steps = 20

        self.CLIP_MAX_NEW_TOKENS_ESTIMATION = 512  # max limit

        self.tree_cache = RadixCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
        # self.cache_metrics = {"total_tokens": 0, "hit_tokens": 0}

    def run_batch(
        self,
        requests: List[torch.Tensor],
        sampling: Optional[SamplingParams] = None,
    ) -> List[List[int]]:
        # Step1 Enqueue all requests
        original_order = []
        for i, req_input in enumerate(requests):
            input_ids = req_input.to(self.device)

            req = Request(
                input_ids=input_ids,
                output_ids=[],
                max_new_tokens=min(sampling.max_new_tokens, self.max_total_tokens),
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                do_sample=sampling.do_sample,
                repetition_penalty=getattr(sampling, "repetition_penalty", 1.0),
                eos_id=sampling.eos_id,
                prefix_indices=torch.tensor([], dtype=torch.int32),
                last_node=None,
                num_cached_tokens=0,
            )

            self.waiting_queue.append(req)
            original_order.append(req)

        while True:
            # Step7 Finish all requests and then exit loop
            if len(self.finished_requests) == len(original_order):
                break

            # Step2 Select running batch
            running_batch, mode = self._select_batch_to_run()

            # Handle error case - no available memory
            if mode == "error":
                # TODO: Add proper error handling strategy (e.g., wait, retry, etc.)
                raise RuntimeError("Insufficient memory to process any requests")

            # Step3 Prepare forward batch
            forward_batch = self._prepare_forward(running_batch, mode)

            # Step4 Model forward
            with torch.no_grad():
                logits = self.model(
                    forward_batch.input_ids, forward_batch.positions, forward_batch
                )

            # Step5 Sample next ids
            next_ids = self._sample_next_ids(logits, running_batch, mode)

            # Step6 Process results and update status
            self._process_results(next_ids, running_batch, mode)

        result = []
        for req in original_order:
            result.append(req.output_ids)

        return result

    def _select_batch_to_run(self) -> Tuple[List[Request], str]:
        # 1. Select prefill first
        if self.waiting_queue:
            can_run_list = self._try_select_prefill()
            if can_run_list:
                return can_run_list, "prefill"

        # 2. Else try decode
        if self.decoding_batch:
            decode_batch = self._try_select_decode()
            if decode_batch:
                return decode_batch, "decode"

        # 3. No enough mem to forward at least one request
        return [], "error"

    def _try_select_prefill(self) -> List[Request]:
        """
        1. len(can_run_list) <= req_to_token_pool.availabel_size()
        2. sum((len(input_ids) - cached_len) + max_new_tokens) in can_run_list
           <= token_to_kv_pool.available_size() + tree_cache.evictable_size()
              - sum(r * (max_new_tokens - len(output_ids))) in decoding_batch
        """
        # FIXME(prefix_cache): tree_cache.evictable_size() def add_one_req(self, req: Req) L447 和 L 462
        for req in self.waiting_queue:
            req.prefix_indices, req.last_node = self.tree_cache.match_prefix(
                req.input_ids.tolist()
            )
            req.num_cached_tokens = len(req.prefix_indices)

        self.waiting_queue.sort(key=lambda r: len(r.prefix_indices), reverse=True)

        num_req_available = self.req_to_token_pool.available_size()
        num_token_available = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        r = self.est_new_token_ratio

        num_reserved_tokens = 0
        for req in self.decoding_batch:
            num_reserved_tokens += min(
                int(r * (req.max_new_tokens - len(req.output_ids))),
                self.CLIP_MAX_NEW_TOKENS_ESTIMATION,
            )
        num_rem_tokens = num_token_available - num_reserved_tokens

        can_run_list: List[Request] = []
        for req in self.waiting_queue:
            new_tokens_needed = len(req.input_ids) - len(req.prefix_indices)
            total_tokens_needed = new_tokens_needed + req.max_new_tokens
            if (
                len(can_run_list) + 1 <= num_req_available
                and total_tokens_needed <= num_rem_tokens
            ):
                can_run_list.append(req)
                num_rem_tokens -= total_tokens_needed
            else:
                break

        return can_run_list

    def _try_select_decode(self) -> List[Request]:
        bs = len(self.decoding_batch)

        if bs <= self.token_to_kv_pool.available_size():
            self.est_new_token_ratio = max(
                self.min_r, self.est_new_token_ratio - self.decay_step
            )
            return self.decoding_batch[:]  # shallow copy

        self.tree_cache.evict(bs, self.token_to_kv_pool.free)  # try to free mem firstly

        if bs <= self.token_to_kv_pool.available_size():
            self.est_new_token_ratio = max(
                self.min_r, self.est_new_token_ratio - self.decay_step
            )
            return self.decoding_batch[:]

        # retract some reqs if still oom
        retracted_reqs = []
        while (
            len(self.decoding_batch) * self.retract_decode_steps
            > self.token_to_kv_pool.available_size()
        ):
            if len(self.decoding_batch) == 0:
                break

            # 1) remove req from decoding_batch
            req = self.decoding_batch.pop()

            # 2) free only the uncached suffix firstly
            token_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.num_cached_tokens : req.seq_len
            ]
            self.token_to_kv_pool.free(token_indices)
            self.req_to_token_pool.free(req.req_pool_idx)

            # free prefix lock
            if req.last_node is not None:
                self.tree_cache.dec_lock_ref(req.last_node)

            # 3) reset req
            req.reset()

            # 4) add req to waiting_queue
            retracted_reqs.append(req)
            self.waiting_queue.append(req)

            # 5) try to free shared prefix cache secondly
            residual = max(
                0,
                len(self.decoding_batch) * self.retract_decode_steps
                - self.token_to_kv_pool.available_size(),
            )
            if residual > 0:
                self.tree_cache.evict(residual, self.token_to_kv_pool.free)

        assert len(self.decoding_batch) > 0

        # 5) update est_new_token_ratio
        total_decoded_tokens = sum(len(req.output_ids) for req in self.decoding_batch)
        total_max_new_tokens = sum(req.max_new_tokens for req in self.decoding_batch)
        if total_max_new_tokens > 0:
            new_est_ratio = (
                total_decoded_tokens
                + len(self.decoding_batch) * self.retract_decode_steps
            ) / total_max_new_tokens
        else:
            new_est_ratio = self.est_new_token_ratio  # Keep current ratio

        self.est_new_token_ratio = min(1.0, new_est_ratio)
        return self.decoding_batch[:]  # shallow copy

    def _prepare_forward(
        self, batch_requests: List[Request], mode: str
    ) -> SimplifiedForwardBatch:
        if mode == "prefill":
            return self._prepare_prefill_batch(batch_requests)
        elif mode == "decode":
            return self._prepare_decode_batch(batch_requests)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _prepare_prefill_batch(
        self, batch_requests: List[Request]
    ) -> SimplifiedForwardBatch:
        B = len(batch_requests)

        req_pool_indices = self.req_to_token_pool.alloc(B)
        if req_pool_indices is None:
            raise RuntimeError("Failed to allocate request pool indices")
        req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.long, device=self.device
        )

        input_ids_list = []
        seq_lens = []
        prefix_lens = []

        for req in batch_requests:
            input_ids_list.append(req.input_ids[req.num_cached_tokens :])
            seq_lens.append(len(req.input_ids))
            prefix_lens.append(req.num_cached_tokens)

        concat_input_ids = torch.cat(input_ids_list, dim=0)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        prefix_lens_tensor = torch.tensor(
            prefix_lens, dtype=torch.long, device=self.device
        )
        total_tokens = concat_input_ids.shape[0]
        # Debug: prefill batch construction
        _attn_logger.debug(
            f"_prepare_prefill_batch: B={B} total_new_tokens={total_tokens} waiting_batch_size={len(batch_requests)}"
        )
        _sched_log_tensor_stats("prefill.concat_input_ids", concat_input_ids)
        _sched_log_tensor_stats("prefill.seq_lens_tensor", seq_lens_tensor)
        _sched_log_tensor_stats("prefill.prefix_lens_tensor", prefix_lens_tensor)
        try:
            _attn_logger.debug(
                f"prefill.seq_lens={seq_lens} prefix_lens={prefix_lens} new_token_counts={[int(len(req.input_ids) - req.num_cached_tokens) for req in batch_requests]}"
            )
        except Exception as e:
            _attn_logger.warning(f"prefill lens debug error: {e}")

        out_cache_loc = self.token_to_kv_pool.alloc(total_tokens)
        if out_cache_loc is None:
            # need eviction from radix cache
            # FIXME(prefix_cache): only evict necessary tokens
            self.tree_cache.evict(total_tokens, self.token_to_kv_pool.free)
            out_cache_loc = self.token_to_kv_pool.alloc(total_tokens)
            if out_cache_loc is None:
                avail = (
                    self.token_to_kv_pool.available_size()
                    + self.tree_cache.evictable_size()
                )
                _attn_logger.error(
                    f"_prepare_prefill_batch OOM: need={total_tokens} available={avail}"
                )
                raise RuntimeError(
                    f"Prefill OOM: need {total_tokens}, available {avail}"
                )
        _sched_log_tensor_stats(
            "prefill.out_cache_loc",
            (
                out_cache_loc
                if isinstance(out_cache_loc, torch.Tensor)
                else torch.tensor(out_cache_loc)
            ),
        )

        mapping = torch.zeros(
            (B, self.req_to_token_pool.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )

        token_offset = 0
        for i, seq_len in enumerate(seq_lens):
            req = batch_requests[i]

            # Write cached prefix indices first
            mapping[i, : req.num_cached_tokens] = req.prefix_indices
            # Then append new token KV locations
            new_cnt = seq_len - req.num_cached_tokens
            mapping[i, req.num_cached_tokens : seq_len] = out_cache_loc[
                token_offset : token_offset + new_cnt
            ]
            _attn_logger.debug(
                f"_prepare_prefill_batch map[i={i}]: req_pool_idx(pending) prefix_len={req.num_cached_tokens} full_seq_len={seq_len} new_cnt={new_cnt} cache_loc_offset_range=[{token_offset},{token_offset + new_cnt - 1}]"
            )
            token_offset += new_cnt

            req.req_pool_idx = req_pool_indices[i].item()
            req.seq_len = seq_len
            self.tree_cache.inc_lock_ref(req.last_node)

        _sched_log_tensor_stats("prefill.mapping", mapping)
        self.req_to_token_pool.write(req_pool_indices, mapping)

        return SimplifiedForwardBatch.create_prefill_batch(
            input_ids=concat_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_tensor,
            out_cache_loc=out_cache_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            prefix_lens=prefix_lens_tensor,
        )

    def _prepare_decode_batch(
        self, batch_requests: List[Request]
    ) -> SimplifiedForwardBatch:
        B = len(batch_requests)

        new_kv_slots = self.token_to_kv_pool.alloc(B)
        if new_kv_slots is None:
            raise RuntimeError("Failed to allocate KV slots for decode")

        req_pool_indices = []
        input_ids = []
        updated_seq_lens = []

        for i, req in enumerate(batch_requests):
            if req.req_pool_idx is None:
                raise RuntimeError(f"Request {i} missing req_pool_idx for decode")

            req_pool_indices.append(req.req_pool_idx)

            # For decode, we need the last generated token as input
            if not req.output_ids:
                raise RuntimeError(f"Request {i} has no output tokens for decode")
            input_ids.append(req.output_ids[-1])

            current_pos = req.seq_len  # logical position for the new token
            self.req_to_token_pool.req_to_token[req.req_pool_idx, current_pos] = (
                new_kv_slots[i]
            )
            req.seq_len += 1  # advance sequence length after allocating the slot
            updated_seq_lens.append(req.seq_len)

        req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.long, device=self.device
        )
        seq_lens_tensor = torch.tensor(
            updated_seq_lens, dtype=torch.long, device=self.device
        )
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        return SimplifiedForwardBatch.create_decode_batch(
            input_ids=input_ids_tensor,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_tensor,
            out_cache_loc=new_kv_slots,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )

    def _sample_next_ids(
        self, logits: torch.Tensor, batch_requests: List[Request], mode: str
    ) -> torch.Tensor:
        # 1. extract last token for prefill requests
        if mode == "prefill":
            seq_lens = [
                len(req.input_ids) - len(req.prefix_indices) for req in batch_requests
            ]
            ends = torch.cumsum(torch.tensor(seq_lens, device=logits.device), dim=0)
            last_indices = ends - 1
            try:
                _attn_logger.debug(
                    f"_sample_next_ids(prefill): seq_lens={seq_lens} ends={ends.tolist()} last_indices={last_indices.tolist()}"
                )
            except Exception as e:
                _attn_logger.warning(f"_sample_next_ids(prefill) debug error: {e}")
            logits = torch.stack(
                [logits[last_indices[i]] for i in range(len(batch_requests))], dim=0
            )
            _sched_log_tensor_stats("prefill.logits_last", logits)

        # 2. sample next token
        next_ids = []
        for i, req in enumerate(batch_requests):
            token_logits = logits[i : i + 1]  # [1, vocab_size]
            next_id = sample_next_ids(
                logits=token_logits,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                prev_ids=req.output_ids,
                repetition_penalty=req.repetition_penalty,
            )
            next_ids.append(next_id.item())
        _attn_logger.debug(f"_sample_next_ids: mode={mode} next_ids={next_ids}")
        return torch.tensor(next_ids, dtype=torch.long, device=logits.device)

    def _process_results(
        self, next_ids: torch.Tensor, batch_requests: List[Request], mode: str
    ):
        # 1) append sampled token id and set finish flags
        for i, req in enumerate(batch_requests):
            next_id = int(next_ids[i].item())
            req.output_ids.append(next_id)

            finished = False
            if len(req.output_ids) >= req.max_new_tokens:
                finished = True

            if req.eos_id != -1 and next_id == req.eos_id:
                finished = True

            req.finished = finished

        # 2) handle cache + queues
        if mode == "prefill":
            for req in batch_requests:
                if req.finished:
                    self.tree_cache.cache_finished_req(req)
                    self.finished_requests.append(req)
                else:
                    self.tree_cache.cache_unfinished_req(
                        req, token_ids=req.input_ids.tolist()
                    )
                    self.decoding_batch.append(req)

                # remove from waiting queue
                if req in self.waiting_queue:
                    self.waiting_queue.remove(req)

        elif mode == "decode":
            for req in batch_requests:
                if req.finished:
                    self.tree_cache.cache_finished_req(req)
                    self.finished_requests.append(req)

            self.decoding_batch = [
                req for req in self.decoding_batch if not req.finished
            ]
