from typing import Dict
from dataclasses import dataclass
import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_id: int = -1
    do_sample: bool = False
    repetition_penalty: float = 1.0
    json_schema: Optional[Dict[str, any]] = None


from typing import Optional


def enforce_repetition_penalty_(
    logits: torch.Tensor, prev_ids: Optional[torch.Tensor], penalty: float
) -> torch.Tensor:
    """
    In-place-ish repetition penalty following common practice:
      - if logit > 0: logit /= penalty
      - else:         logit *= penalty
    """
    if penalty is None or penalty <= 1.0:
        return logits
    if prev_ids is None:
        return logits
    if isinstance(prev_ids, list):
        if len(prev_ids) == 0:
            return logits
        prev_ids = torch.tensor(prev_ids, dtype=torch.long, device=logits.device)
    if prev_ids.numel() == 0:
        return logits

    unique_ids = torch.unique(prev_ids)
    # Apply adjustment token-wise
    for tid in unique_ids.tolist():
        val = logits[..., tid]
        logits[..., tid] = torch.where(val > 0, val / penalty, val * penalty)
    return logits


def sample_next_ids(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    prev_ids: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    # 0) clone to avoid mutating caller's tensor
    logits = logits.clone()

    # 1) repetition penalty before filtering
    logits = enforce_repetition_penalty_(logits, prev_ids, repetition_penalty)

    # 2) greedy vs sampling
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    # 3) temperature scaling
    if temperature is None or temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature

    # 4) top-k filtering
    if top_k and top_k > 0:
        kth = (
            torch.topk(scaled, k=min(top_k, scaled.shape[-1]), dim=-1)
            .values[..., -1:]
            .expand_as(scaled)
        )
        scaled = torch.where(
            scaled < kth, torch.full_like(scaled, float("-inf")), scaled
        )

    # 5) top-p (nucleus) filtering
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[..., 0] = False
        sorted_logits = torch.where(
            cutoff, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(-1, sorted_indices, sorted_logits)

    # 6) sample
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
