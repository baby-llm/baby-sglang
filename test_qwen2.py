import argparse
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from model_loader import build_model_from_hf
from forward_batch import SimplifiedForwardBatch
from memory_pool import ReqToTokenPool, MHATokenToKVPool


def get_builtin_prompts(preset: str) -> List[str]:
    if preset == "en":
        return [
            "Summarize the key benefits of unit testing in software engineering.",
            "Explain the concept of attention in transformer models.",
            "Draft a short professional email requesting a project status update.",
            "Give three tips to improve public speaking.",
        ]
    if preset == "cn":
        return [
            "用中文简要介绍大语言模型的工作原理。",
            "请解释A/B测试的基本流程和注意事项。",
            "写一段100字以内的自我介绍，语气自然。",
            "给出三条提高学习效率的建议。",
        ]
    if preset == "code":
        return [
            "Write a Python function to merge two sorted lists in O(n) time.",
            "Implement a BFS traversal on a graph in Python and return the visit order.",
            "Provide a Python snippet that debounces a function call using asyncio.",
            "Show a minimal example of context manager usage in Python.",
        ]
    if preset == "math":
        return [
            "Solve: If 3x + 2 = 20, what is x?",
            "Explain the difference between mean, median, and mode with an example.",
            "What is the derivative of x^3 + 2x w.r.t x?",
            "Prove by induction that the sum of the first n integers equals n(n+1)/2.",
        ]
    if preset == "qa":
        return [
            "Who wrote the novel '1984' and what is the central theme?",
            "What are the primary layers of the Earth's atmosphere?",
            "Describe the causes and effects of photosynthesis.",
            "What is the difference between RAM and ROM?",
        ]
    return [
        "Hello, introduce yourself briefly.",
        "用中文介绍一下量子计算的基本原理。",
        "Write a Python function to compute Fibonacci numbers efficiently.",
        "给出三条关于上海旅游的建议。",
    ]


def create_prefill_test(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    prompts: Optional[List[str]] = None,
    preset: str = "mix",
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    if prompts is None or len(prompts) == 0:
        prompts = get_builtin_prompts(preset)
    if seed is not None:
        torch.manual_seed(seed)
    model = build_model_from_hf(model_id, device="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    p = next(model.parameters())
    device = p.device
    ids_list: List[torch.Tensor] = []
    lens: List[int] = []
    for s in prompts:
        enc = tokenizer(s, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].squeeze(0).to(device)
        ids_list.append(ids)
        lens.append(int(ids.numel()))
    B = len(ids_list)
    L_max = max(lens)
    L_total = sum(lens)
    concat_ids = torch.cat(ids_list, dim=0)
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    kv_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    token_to_kv_pool = MHATokenToKVPool(
        size=L_total,
        dtype=kv_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=model.config.num_hidden_layers,
        device=str(device),
    )
    req_to_token_pool = ReqToTokenPool(
        size=B,
        max_context_len=L_max,
        device=str(device),
    )
    out_cache_loc = token_to_kv_pool.alloc(L_total)
    mapping = torch.zeros((B, L_max), dtype=torch.int32, device=device)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    for i in range(B):
        s, e = offsets[i], offsets[i + 1]
        mapping[i, : lens[i]] = out_cache_loc[s:e]
    req_indices = torch.arange(B, dtype=torch.long, device=device)
    req_to_token_pool.write(req_indices, mapping)
    seq_lens = torch.tensor(lens, dtype=torch.long, device=device)
    fwd_batch = SimplifiedForwardBatch.create_prefill_batch(
        input_ids=concat_ids,
        req_pool_indices=req_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
    )
    with torch.no_grad():
        logits = model(concat_ids, fwd_batch.positions, fwd_batch)
    ends = torch.cumsum(seq_lens, dim=0)
    last_indices = ends - 1
    last_logits = torch.stack([logits[last_indices[i]] for i in range(B)], dim=0)
    next_ids = sample_next_ids(last_logits, do_sample, temperature, top_k, top_p)
    next_texts = [
        tokenizer.decode(int(next_ids[i].item()), skip_special_tokens=True)
        for i in range(B)
    ]
    print("=" * 12 + " Prefill " + "=" * 12)
    for i, s in enumerate(prompts):
        print(f"[{i}] Prompt: {s}")
        print(f"Next token id: {int(next_ids[i].item())}")
        print(f"Next token text: {repr(next_texts[i])}")
    print("=" * 30)
    return model, tokenizer, [int(x.item()) for x in next_ids], next_texts


def create_decode_test(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    prompts: Optional[List[str]] = None,
    preset: str = "mix",
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    if prompts is None or len(prompts) == 0:
        prompts = get_builtin_prompts(preset)
    if seed is not None:
        torch.manual_seed(seed)
    model = build_model_from_hf(model_id, device="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    p = next(model.parameters())
    device = p.device
    ids_list: List[torch.Tensor] = []
    lens: List[int] = []
    for s in prompts:
        enc = tokenizer(s, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].squeeze(0).to(device)
        ids_list.append(ids)
        lens.append(int(ids.numel()))

    B = len(ids_list)
    L_max = max(lens)
    L_total = sum(lens)
    concat_ids = torch.cat(ids_list, dim=0)
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    kv_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    token_to_kv_pool = MHATokenToKVPool(
        size=L_total + max_new_tokens * B,
        dtype=kv_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=model.config.num_hidden_layers,
        device=str(device),
    )
    req_to_token_pool = ReqToTokenPool(
        size=B,
        max_context_len=L_max + max_new_tokens,
        device=str(device),
    )
    out_cache_loc_prefill = token_to_kv_pool.alloc(L_total)
    mapping = torch.zeros((B, L_max + max_new_tokens), dtype=torch.int32, device=device)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    for i in range(B):
        s, e = offsets[i], offsets[i + 1]
        mapping[i, : lens[i]] = out_cache_loc_prefill[s:e]
    req_indices = torch.arange(B, dtype=torch.long, device=device)
    req_to_token_pool.write(req_indices, mapping)
    seq_lens = torch.tensor(lens, dtype=torch.long, device=device)
    fwd_batch = SimplifiedForwardBatch.create_prefill_batch(
        input_ids=concat_ids,
        req_pool_indices=req_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc_prefill,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
    )

    with torch.no_grad():
        logits = model(concat_ids, fwd_batch.positions, fwd_batch)
    ends = torch.cumsum(seq_lens, dim=0)
    last_indices = ends - 1
    last_logits = torch.stack([logits[last_indices[i]] for i in range(B)], dim=0)
    next_ids = sample_next_ids(last_logits, do_sample, temperature, top_k, top_p).to(
        device
    )
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    generated: List[List[int]] = [[] for _ in range(B)]
    cur_lens = seq_lens.clone()
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        new_locs = token_to_kv_pool.alloc(B)
        rows = torch.arange(B, dtype=torch.long, device=device)
        req_to_token_pool.req_to_token[rows, cur_lens] = new_locs
        cur_lens = cur_lens + 1
        dec_input = next_ids.to(device=device, dtype=torch.long)
        for i in range(B):
            if not bool(finished[i].item()):
                generated[i].append(int(dec_input[i].item()))
        dec_fwd = SimplifiedForwardBatch.create_decode_batch(
            input_ids=dec_input,
            req_pool_indices=req_indices,
            seq_lens=cur_lens,
            out_cache_loc=new_locs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
        )
        dec_positions = dec_fwd.positions
        with torch.no_grad():
            dec_logits = model(dec_input, dec_positions, dec_fwd)
        next_ids = sample_next_ids(dec_logits, do_sample, temperature, top_k, top_p)
        if eos_id != -1:
            finished = finished | (next_ids.to(device) == eos_id)
            next_ids = torch.where(
                finished, torch.full_like(next_ids, eos_id), next_ids
            )
        if bool(finished.all().item()):
            break
    completions: List[str] = []
    for i in range(B):
        completions.append(tokenizer.decode(generated[i], skip_special_tokens=True))
    print("=" * 12 + " Decode " + "=" * 12)
    for i, s in enumerate(prompts):
        print(f"[{i}] Prompt: {s}")
        print(f"Completion: {completions[i]!r}")
    print("=" * 30)
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="prefill", choices=["prefill", "decode"]
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", type=str, action="append", default=None)
    parser.add_argument(
        "--preset",
        type=str,
        default="mix",
        choices=["mix", "en", "cn", "code", "math", "qa"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    prompts: Optional[List[str]] = (
        args.prompt if args.prompt else get_builtin_prompts(args.preset)
    )
    if args.mode == "prefill":
        create_prefill_test(
            model_id=args.model_id,
            prompts=prompts,
            preset=args.preset,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
    else:
        create_decode_test(
            model_id=args.model_id,
            prompts=prompts,
            preset=args.preset,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
