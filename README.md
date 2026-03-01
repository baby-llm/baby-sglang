# Baby-SGL

A lightweight sglang implementation built from scratch step-by-step with only ~2,000 lines of code.

## Inspiration

Inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

Designed for SDEs without GPU programming experience.

Focuses on the core control plane of sglang (caching, scheduling, parallelism), excluding low-level complex GPU kernel optimizations.

## Quick Start

```bash
python run.py 
Enter prompt: 用中文简要介绍大语言模型的工作原理 。
Initing engine with model=Qwen/Qwen2.5-1.5B-Instruct, device=cuda ...
`torch_dtype` is deprecated! Use `dtype` instead!
Generating...
大语言模型（Large Language Models，LLM）是一种基于深度学习的机器学习模型，通过大规模的数据训练，能够理解并生成自然语言。它的工作原理主要包括以下几个步骤：
1. 数据收集：首先，模型需要大量的文本数据作为训练集，这些数据可以是书籍、文章、网页、社交媒体等。
2. 数据预处理：数据预处理是将原始文本数据转换为模型可以理解的形式，包括分词、去停用词、词性标注等。
3. 模型训练：模型通过大量训练数据进行学习，训练的目标是尽可能准确地预测文本中的语言和含义。训练过程中，模型会不断调整参数，以优化预测结果。
4. 模型评估：训练完成后，模型需要进行评估，以确定其性能。评估通常包括准确率、召回率、F1分数等指标。
5. 模型应用：经过训练和评估后的模型可以应用于各种任务，如文本生成、机器翻译、问答系统等。
大语言模型的工作原理是基于深度学习和自然语言处理技术，通过大规模的数据训练，能够理解并生成自然语言。
```

```bash
python run.py 
Enter prompt: Give three tips to improve public speaking.
Initing engine with model=Qwen/Qwen2.5-1.5B-Instruct, device=cuda ...
`torch_dtype` is deprecated! Use `dtype` instead!
Generating...
1. Practice: The more you practice, the more comfortable you will become with the material and the more confident you will feel. This will help you deliver your speech with ease and avoid nervousness.
2. Know your audience: Understanding your audience will help you tailor your speech to their interests and needs. This will make your speech more engaging and relevant, and will help you connect with your audience on a personal level.
3. Use body language: Your body language can say a lot about you. Make sure to use open, confident body language, such as facing the audience, maintaining eye contact, and standing up straight. This will help you appear more confident and professional, and will help you connect with your audience.
```

## Roadmap

- ✅ Qwen2.5 support
- ✅ Paged attention
- ✅ Dynamic batching
- ✅ Radix attention
- ✅ Constraint decoding
- ✅ Overlap schedule
- 🚀 Benchmark
- 🚧 TP support
- 🚧 Observability & Profile
- 🚧 Speculative decoding
- 🚧 Kernel-level Optimization
- 🤔 Multiple level cache
- 🤔 PD Disaggregation
- 🤔 Semantic cache
