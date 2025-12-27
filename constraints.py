from typing import Any, Dict, List, Optional
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
)
from lmformatenforcer.tokenenforcer import TokenEnforcer
import math


class ConstraintState:
    def process(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class JsonConstraintState(ConstraintState):
    def __init__(self, schema: Dict[str, Any], tokenizer) -> None:
        # ref: https://github.com/noamgat/lm-format-enforcer/blob/main/lmformatenforcer/integrations/vllm.py
        tok_data = build_token_enforcer_tokenizer_data(tokenizer, use_bitmask=False)
        self.token_enforcer = TokenEnforcer(tok_data, JsonSchemaParser(schema))
        self.mask: Optional[torch.Tensor] = None

    def process(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        token_sequence = input_ids
        allowed_tokens = self.token_enforcer.get_allowed_tokens(
            token_sequence
        ).allowed_tokens
        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            # We create it here because full_like() also copies the device and dtype
            self.mask = torch.full_like(scores, -math.inf)
        self.mask[..., allowed_tokens] = 0
        scores = scores + self.mask
        return scores

    def reset(self):
        self.token_enforcer.prefix_states.clear()
        self.mask = None
