from dataclasses import dataclass, field
from typing import List, Optional
import torch
from radix_tree import TreeNode
from constraints import ConstraintState


@dataclass
class Request:
    input_ids: torch.Tensor  # input prompts on the device
    output_ids: List[int]  # output ids
    max_new_tokens: int
    eos_id: int
    temperature: float
    top_k: int
    top_p: float
    do_sample: bool
    repetition_penalty: float

    finished: bool = False
    req_pool_idx: Optional[int] = None  # Index in req_to_token_pool when allocated
    seq_len: int = 0  # Current sequence length (input + output tokens)

    prefix_indices: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32)
    )
    last_node: Optional[TreeNode] = None
    num_cached_tokens: int = 0
    constraint_state: Optional[ConstraintState] = None

    def reset(self):
        self.output_ids = []
        self.req_pool_idx = None
        self.seq_len = 0
        self.finished = False

        self.prefix_indices = torch.tensor([], dtype=torch.int32)
        self.last_node = None
        self.num_cached_tokens = 0
        self.constraint_state = None
