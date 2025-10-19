#!/usr/bin/env python3
"""
Smoke test for Engine.generate() method.
This test file replicates the logic from test_qwen2.py but uses the high-level Engine interface.
"""

import argparse
from typing import List, Optional
import torch

from engine import Engine
from sample import SamplingParams


def get_builtin_prompts(preset: str) -> List[str]:
    """Get built-in test prompts - copied from test_qwen2.py"""
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


def test_engine_basic_generation(
    model_id: str = "Qwen/qwen2.5-1.5B",
    prompts: Optional[List[str]] = None,
    preset: str = "mix",
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    device: str = "auto",
):
    """
    Smoke test for basic generation functionality.
    Similar to create_decode_test in run_demo.py but uses Engine.generate().
    """
    if prompts is None or len(prompts) == 0:
        prompts = get_builtin_prompts(preset)

    if seed is not None:
        torch.manual_seed(seed)

    # Initialize Engine
    print(f"Initializing Engine with model: {model_id}")
    engine = Engine(model_id=model_id, device=device)

    # Create sampling parameters
    sampling = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    print("=" * 12 + " Engine Smoke Test " + "=" * 12)
    print(f"Testing with {len(prompts)} prompts")
    print(f"Sampling params: max_new_tokens={max_new_tokens}, do_sample={do_sample}")
    if do_sample:
        print(
            f"                temperature={temperature}, top_k={top_k}, top_p={top_p}"
        )

    # Test generation
    try:
        outputs = engine.generate(prompts, sampling)

        # Basic smoke test assertions
        assert isinstance(outputs, list), "Output should be a list"
        assert len(outputs) == len(
            prompts
        ), f"Expected {len(prompts)} outputs, got {len(outputs)}"

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            assert isinstance(output, str), f"Output {i} should be a string"
            assert len(output) > 0, f"Output {i} should not be empty"

            print(f"[{i}] Prompt: {prompt}")
            print(f"    Output: {output!r}")
            print()

        print("✅ Engine smoke test PASSED")
        return outputs

    except Exception as e:
        print(f"❌ Engine smoke test FAILED: {e}")
        raise

    finally:
        print("=" * 45)


def test_engine_single_prompt(
    model_id: str = "Qwen/qwen2.5-1.5B",
    seed: Optional[int] = None,
    device: str = "auto",
):
    """Test engine with a single simple prompt"""
    if seed is not None:
        torch.manual_seed(seed)

    engine = Engine(model_id=model_id, device=device)
    sampling = SamplingParams(max_new_tokens=16, do_sample=False)

    prompt = "Hello, how are you?"
    outputs = engine.generate([prompt], sampling)

    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    assert len(outputs[0]) > 0

    print("=" * 12 + " Single Prompt Test " + "=" * 12)
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs[0]!r}")
    print("✅ Single prompt test PASSED")
    print("=" * 45)

    return outputs[0]


def test_engine_empty_input(device: str = "auto"):
    """Test engine with edge cases"""
    engine = Engine(model_id="Qwen/qwen2.5-1.5B", device=device)
    sampling = SamplingParams(max_new_tokens=8, do_sample=False)

    # Test empty prompt list
    try:
        outputs = engine.generate([], sampling)
        assert len(outputs) == 0
        print("✅ Empty input test PASSED")
    except Exception as e:
        print(f"❌ Empty input test FAILED: {e}")
        raise


def run_comprehensive_smoke_test(
    model_id: str = "Qwen/qwen2.5-1.5B",
    seed: Optional[int] = None,
    device: str = "auto",
):
    """Run comprehensive smoke tests covering various scenarios"""
    print("🚀 Starting comprehensive Engine smoke tests...")
    print()

    # Test 1: Single prompt
    test_engine_single_prompt(model_id, seed, device)

    # Test 2: Empty input edge case
    test_engine_empty_input(device=device)

    # Test 3: Multiple prompts with greedy sampling
    test_engine_basic_generation(
        model_id=model_id,
        preset="en",
        max_new_tokens=24,
        do_sample=False,
        seed=seed,
        device=device,
    )

    # Test 4: Mixed language prompts
    test_engine_basic_generation(
        model_id=model_id,
        preset="mix",
        max_new_tokens=20,
        do_sample=False,
        seed=seed,
        device=device,
    )

    # Test 5: Sampling generation (if no seed to ensure reproducibility)
    if seed is not None:
        test_engine_basic_generation(
            model_id=model_id,
            preset="cn",
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            seed=seed,
            device=device,
        )

    print("🎉 All smoke tests completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Engine.generate()")
    parser.add_argument("--model-id", type=str, default="Qwen/qwen2.5-1.5B")
    parser.add_argument("--prompt", type=str, action="append", default=None)
    parser.add_argument(
        "--preset",
        type=str,
        default="mix",
        choices=["mix", "en", "cn", "code", "math", "qa"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument(
        "--comprehensive", action="store_true", help="Run comprehensive smoke tests"
    )

    args = parser.parse_args()

    if args.comprehensive:
        run_comprehensive_smoke_test(
            model_id=args.model_id, seed=args.seed, device=args.device
        )
    else:
        prompts = args.prompt if args.prompt else get_builtin_prompts(args.preset)
        test_engine_basic_generation(
            model_id=args.model_id,
            prompts=prompts,
            preset=args.preset,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
