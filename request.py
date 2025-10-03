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

    def reset(self):
        self.output_ids = []
        self.req_pool_idx = None
        self.seq_len = 0
        self.finished = False
