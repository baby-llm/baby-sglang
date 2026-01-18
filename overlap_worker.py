from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional
from queue import Queue
import torch

from forward_batch import SimplifiedForwardBatch
from sample import SamplingParams, sample_next_ids


@dataclass
class _WorkItem:
    forward_batch: SimplifiedForwardBatch
    mode: str  # prefill or decode
    sampling: SamplingParams
    future_start: int
    batch_size: int


@dataclass
class _WorkResult:
    next_ids: torch.Tensor


class OverlapWorker:

    def __init__(
        self,
        model: torch.nn.Module,
        max_requests: int = 32,
        future_token_ids_limit: Optional[int] = None,
    ) -> None:
        self.model = model
        p = next(model.parameters())
        self.device = p.device

        # Future-token mapping (copy from sglang)
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = future_token_ids_limit or (max_requests * 3)
        self.future_token_ids_map = torch.empty(
            (max_requests * 5,), dtype=torch.int32, device=self.device
        )

        self.input_queue: Queue[_WorkItem] = Queue()
        self.output_queue: Queue[_WorkResult] = Queue()

        self.forward_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self.forward_stream = torch.cuda.Stream(device=self.device)
        else:
            raise NotImplementedError("non-CUDA is not supported yet")

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()  # Todo: disable it when normal mode

    def reset(self) -> None:
        if self.forward_stream is not None:
            self.forward_stream.synchronize()

        self.future_token_ids_ct = 0
        self.future_token_ids_map.zero_()

    def submit(
        self,
        forward_batch: SimplifiedForwardBatch,
        mode: str,
        sampling: SamplingParams,
    ) -> torch.Tensor:
        # 1. allocate future token ids from contigious range in the future map
        bs = int(forward_batch.batch_size)
        future_start = self.future_token_ids_ct
        future_next_ids = torch.arange(
            -(future_start + 1),
            -(future_start + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.future_token_ids_ct = (future_start + bs) % self.future_token_ids_limit

        # 2. sync the scheduler stream before enqueue like sglang
        torch.cuda.current_stream().synchronize()

        # 3. enqueue the forward req and return immediately
        self.input_queue.put(
            _WorkItem(
                forward_batch=forward_batch,
                mode=mode,
                sampling=sampling,
                future_start=future_start,  # [fs+1, ... fs+bs]
                batch_size=bs,
            )
        )
        return future_next_ids

    def resolve(self) -> torch.Tensor:
        copy_done, next_ids_cpu = self.output_queue.get()
        copy_done.synchronize()

        return next_ids_cpu

    def _thread_main(self) -> None:
        with torch.cuda.stream(self.forward_stream):
            self._thread_main_()

    @torch.no_grad()
    def _thread_main_(self):
        while True:
            # 1. Block until a new batch
            work = self.input_queue.get()

            # 2. Resolve negative placeholder
            input_ids = work.forward_batch.input_ids
            # Todo: enable @torch.compile
            input_ids[:] = torch.where(
                input_ids < 0,
                self.future_token_ids_map[torch.clamp(-input_ids, min=0)],
                input_ids,
            )
            # 3. Execute model forward
            logits = self.model(
                work.forward_batch.input_ids,
                work.forward_batch.positions,
                work.forward_batch,
            )
            # 4. Extract last logit for sampling
            if work.mode == "prefill":
                seq_lens = work.forward_batch.extended_lens.to(torch.long)
                ends = torch.cumsum(seq_lens, dim=0)
                last_indices = ends - 1
                logits = logits[last_indices]
            # 5. Sample
            next_ids_list = []
            for i in range(work.batch_size):
                token_logits = logits[i : i + 1]  # [1, vocab]
                next_id = sample_next_ids(
                    logits=token_logits,
                    do_sample=work.sampling.do_sample,
                    temperature=work.sampling.temperature,
                    top_k=work.sampling.top_k,
                    top_p=work.sampling.top_p,
                    prev_ids=[],  # Todo: repetition penalty is not supported now
                    repetition_penalty=1.0,
                )
                next_ids_list.append(int(next_id.item()))
            next_ids = torch.tensor(
                next_ids_list, dtype=torch.int32, device=self.device
            )
            # 6. Populate future map
            s = work.future_start
            b = work.batch_size
            self.future_token_ids_map[s + 1 : s + 1 + b] = next_ids
            # 7. Copy next_ids to cpu
            # 8. Put result to output queue
            next_ids_cpu = next_ids.to("cpu", non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(self.forward_stream)
            self.output_queue.put((copy_done, next_ids_cpu))
