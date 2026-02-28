#!/usr/bin/env python3

# Adapted from https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
import torch
from engine import Engine
from sample import SamplingParams
import random


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B"
    num_seqs = 256
    input_len = 1024
    output_len = 1024
    seed = 0

    torch.manual_seed(seed)
    random.seed(seed)

    engine = Engine(model_id=model_id, device="cuda")
    device = engine.scheduler.device
    vocab_size = int(engine.model.config.vocab_size)
    gen = torch.Generator(device=device).manual_seed(seed)

    prompt_token_ids = [
        torch.randint(
            0,
            vocab_size,
            (random.randint(100, input_len),),
            generator=gen,
            dtype=torch.long,
            device=device,
        )
        for _ in range(num_seqs)
    ]

    sampling = SamplingParams(
        max_new_tokens=output_len,
        do_sample=True,
        temperature=0.6,
        eos_id=-1,  # disable EOS
    )

    engine.scheduler.run_batch_overlap([prompt_token_ids[-1]], sampling)
    engine.reset()

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = engine.scheduler.run_batch_overlap(prompt_token_ids, sampling)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    total_tokens = sum(len(x) for x in out)
    throughput = total_tokens / dt
    print(
        f"Total: {total_tokens} token, Time: {dt:.2f} s, Throughput: {throughput:.2f} token/s"
    )


if __name__ == "__main__":
    main()
