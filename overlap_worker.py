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

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()  # Todo: disable it when normal mode

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
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        # 3.
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
        raise NotImplementedError

    def _thread_main(self) -> None:
        raise NotImplementedError
