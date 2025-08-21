"""
Demo script for baby-sglang.

Simple example showing how to use the baby-sglang engine
for text generation.
"""

import logging
import torch
import time
from engine import Engine
from managers.io_struct import ModelConfig, SamplingParams

try:
    from vllm import LLM, SamplingParams as VLLMSamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_vllm(model_name: str, prompts: list, sampling_params: SamplingParams):
    """Benchmark inference with vLLM for comparison."""
    if not VLLM_AVAILABLE:
        logger.warning("vLLM not available, skipping benchmark")
        return None, None
    
    logger.info("Running vLLM benchmark...")
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        vllm_sampling = VLLMSamplingParams(
            max_tokens=sampling_params.max_new_tokens,
            temperature=getattr(sampling_params, 'temperature', 1.0),
            top_p=getattr(sampling_params, 'top_p', 1.0)
        )
        
        start_time = time.time()
        vllm_outputs = llm.generate(prompts, vllm_sampling)
        vllm_time = time.time() - start_time
        
        vllm_results = [output.outputs[0].text for output in vllm_outputs]
        return vllm_results, vllm_time
        
    except Exception as e:
        logger.error(f"vLLM benchmark failed: {e}")
        return None, None

def main():
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device (this will be slow)")

    model_name = "Qwen/Qwen3-0.6B"
    logger.info(f"Loading model: {model_name}")

    model_config = ModelConfig(
        model_path=model_name,
        tokenizer_path=model_name,
        device=device,
        dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    )

    engine = Engine(model_config)

    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain what is machine learning in simple terms.",
        "Write a short poem about AI.",
    ]

    sampling_params = SamplingParams(max_new_tokens=50)

    logger.info(f"Running inference on {len(prompts)} prompts...")
    start_time = time.time()
    response = engine.generate(prompts, sampling_params)
    baby_sgl_time = time.time() - start_time
    
    logger.info(f"\n=== BABY-SGLANG RESULTS (Time: {baby_sgl_time:.2f}s) ===")
    for i, (prompt, output) in enumerate(zip(prompts, response.outputs)):
        logger.info(f"\n--- Prompt {i+1} ---")
        logger.info(f"Input: {prompt}")
        generated_text = output[len(prompt):]
        logger.info(f"Output: {generated_text.strip()}")
        logger.info("-" * 50)
    
    # Compare with vLLM if available
    vllm_results, vllm_time = benchmark_vllm(model_name, prompts, sampling_params)
    
    if vllm_results and vllm_time:
        logger.info(f"\n=== VLLM RESULTS (Time: {vllm_time:.2f}s) ===")
        for i, (prompt, output) in enumerate(zip(prompts, vllm_results)):
            logger.info(f"\n--- Prompt {i+1} ---")
            logger.info(f"Input: {prompt}")
            logger.info(f"Output: {output.strip()}")
            logger.info("-" * 50)
        
        logger.info(f"\n=== PERFORMANCE COMPARISON ===")
        logger.info(f"Baby-SGLang time: {baby_sgl_time:.2f}s")
        logger.info(f"vLLM time: {vllm_time:.2f}s")
        speedup = baby_sgl_time / vllm_time if vllm_time > 0 else float('inf')
        logger.info(f"Speed ratio (Baby-SGLang/vLLM): {speedup:.2f}x")
        if speedup > 1:
            logger.info("vLLM is faster")
        elif speedup < 1:
            logger.info("Baby-SGLang is faster")
        else:
            logger.info("Similar performance")


if __name__ == "__main__":
    main()