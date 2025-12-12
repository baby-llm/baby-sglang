# baby-sgl

A lightweight sglang implementation built from scratch.
Clarity first, performance second.

## Inspiration

Inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial).

Focus on core idea of building sglang from scratch prioritizes caching, scheduling, parallelism, and asynchrony.

Friendly for SDEs without GPU programming background;

## Quick Start

```bash
python run_demo.py --model-id Qwen/Qwen2.5-0.5B --preset mix --max-new-tokens 1024 --do-sample --temperature 0.7 --top-k 20 --top-p 0.9
```

## Roadmap

- ✅ Qwen2.5 support
- ✅ Paged attention
- ✅ Dynamic batching
- ✅ Radix attention
- ✅ Constraint decoding
- 🚧 Async & Overlap
- 🚧 TP PP DP support
- 🚧 Speculative decoding
- 🚧 Kernel-level Optimization
- 🤔 Multiple level cache
- 🤔 PD Disaggregation
- 🤔 Semantic cache
- 🚀 Benchmark & Observability
