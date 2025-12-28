from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

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
    scheduler_ready: Optional[torch.cuda.Event] = None


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
        raise NotImplementedError

    def submit(
        self,
        forward_batch: SimplifiedForwardBatch,
        mode: str,
        sampling: SamplingParams,
        scheduler_ready: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def resolve(self) -> torch.Tensor:
        raise NotImplementedError

    def _thread_main(self) -> None:
        raise NotImplementedError
