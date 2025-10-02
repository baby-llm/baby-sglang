from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer

from scheduler import Scheduler
from model_loader import build_model_from_hf
from sample import SamplingParams


class Engine:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device

        self.model = build_model_from_hf(model_id, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

        self.scheduler = Scheduler(self.model)

    def generate(
        self,
        requests: List[str],
        sampling: Optional[SamplingParams] = None,
    ) -> List[str]:
        if sampling is None:
            sampling = SamplingParams()

        if (
            sampling.eos_id == -1
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            sampling.eos_id = int(self.tokenizer.eos_token_id)

        p = next(self.model.parameters())
        device = p.device

        # 1. Tokenize to device tensors
        ids_list: List[torch.Tensor] = []
        for s in requests:
            # Use special tokens (e.g., BOS) to improve early decoding stability
            enc = self.tokenizer(s, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"].squeeze(0).to(device)
            ids_list.append(ids)

        # 2. Run scheduler
        out_token_ids: List[List[int]] = self.scheduler.run_batch(ids_list, sampling)

        # 3. Detokenize
        outputs: List[str] = []
        for ids in out_token_ids:
            outputs.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return outputs

    async def generate_async(
        self,
        requests: List[str],
        sampling: Optional[SamplingParams] = None,
    ):
        # TODO(@huangyz): implement the async and streaming output
        raise NotImplementedError("Not Implement")
