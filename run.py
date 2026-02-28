#!/usr/bin/env python3

import argparse
import time
import torch

from engine import Engine
from sample import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner for baby-sgl.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable sampling. Default is enabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument(
        "--enable-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable overlap scheduling. Default is enabled.",
    )
    return parser.parse_args()


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    torch.manual_seed(42)

    args = parse_args()

    user_prompt = input("Enter prompt: ").strip()

    print(f"Initing engine with model={args.model_id}, device={args.device} ...")
    engine = Engine(model_id=args.model_id, device=args.device)

    prompt = build_chat_prompt(engine.tokenizer, user_prompt)
    sampling = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print("Generating...")
    t0 = time.perf_counter()
    outputs = engine.generate([prompt], sampling, enable_overlap=args.enable_overlap)
    dt = time.perf_counter() - t0

    print(outputs[0])
    print(f"\nlatency={dt:.3f}s")


if __name__ == "__main__":
    main()
