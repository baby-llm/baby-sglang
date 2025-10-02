# baby-sgl

Minimal, hackable LLM inference playground inspired by SGLang. Clarity first, performance second.

## Quick Start

```bash
python -m test_qwen2 --mode decode --model-id Qwen/Qwen2.5-0.5B --preset cn --max-new-tokens 64 --do-sample --temperature 0.7 --top-k 40 --top-p 0.9
```

## Roadmap

- ✅ Qwen2 baseline support
- ✅ Paged Attention
- ✅ Dynamic batching
- 🚧 Radix attention
- 🚧 Asynchronous processing
- 🚧 Multiple level cache
