# baby-sgl

Minimal, hackable LLM inference playground inspired by SGLang. Clarity first, performance second.

## Quick Start

```bash
python run_demo.py --model-id Qwen/Qwen2.5-0.5B --preset json --max-new-tokens 1024 --do-sample --temperature 0.7 --top-k 20 --top-p 0.9
```

## Roadmap

- ✅ Qwen2 support
- ✅ Paged attention
- ✅ Dynamic batching
- ✅ Radix attention
- ✅ Constraint decoding
- 🚧 Asynchronous processing
- 🚧 Multiple level cache
- 🚧 TP PP DP support
