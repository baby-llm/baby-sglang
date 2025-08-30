"""
Demo script for baby-sglang with vLLM comparison.

Usage:
    python run_demo.py          # Full comparison
    python run_demo.py --demo   # Quick demo
"""

import logging
import time
from typing import List, Tuple
import torch
from engine import create_engine_from_hf, SimplifiedSamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2-1.5B"

# Simple test prompts
DEMO_PROMPTS = [
    "Hello, how are you?",
    "The capital of France is",
    "Explain machine learning:",
    "Write a Python function to add two numbers:",
]

def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"

def setup_baby_sglang_engine(model_path: str, device: str):
    """Setup baby-sglang engine."""
    logger.info(f"Setting up baby-sglang engine with {model_path} on {device}")
    
    engine = create_engine_from_hf(
        hf_model_path=model_path,
        device=device,
        max_batch_size=4,
        max_total_tokens=1024,
        max_context_len=256
    )
    
    logger.info("Baby-sglang engine setup completed")
    return engine

def setup_vllm_engine(model_path: str, device: str):
    """Setup vLLM engine."""
    try:
        from vllm import LLM
        logger.info(f"Setting up vLLM engine with {model_path} on {device}")
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=256,
            dtype="float16"
        )
        
        logger.info("vLLM engine setup completed")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to setup vLLM: {e}")
        return None

def run_baby_sglang_inference(engine, prompts: List[str]) -> Tuple[List[str], float]:
    """Run inference using baby-sglang."""
    sampling_params = SimplifiedSamplingParams(
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )
    
    start_time = time.time()
    results = engine.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_real_tokenizer=True
    )
    inference_time = time.time() - start_time
    
    return results, inference_time

def run_vllm_inference(llm_engine, prompts: List[str]) -> Tuple[List[str], float]:
    """Run inference using vLLM."""
    try:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
        )
        
        start_time = time.time()
        outputs = llm_engine.generate(prompts, sampling_params)
        inference_time = time.time() - start_time
        
        results = [output.prompt + output.outputs[0].text for output in outputs]
        return results, inference_time
        
    except Exception as e:
        logger.error(f"vLLM inference failed: {e}")
        return [], 0.0

def compare_engines(prompts: List[str]):
    """Compare baby-sglang and vLLM engines."""
    device = get_best_device()
    logger.info(f"Using device: {device}")
    
    # Setup engines
    try:
        baby_engine = setup_baby_sglang_engine(MODEL_NAME, device)
    except Exception as e:
        logger.error(f"Baby-sglang setup failed: {e}")
        baby_engine = None
    
    try:
        vllm_engine = setup_vllm_engine(MODEL_NAME, device) 
    except Exception as e:
        logger.error(f"vLLM setup failed: {e}")
        vllm_engine = None
    
    if not baby_engine and not vllm_engine:
        logger.error("Both engines failed to initialize")
        return
    
    # Run comparisons
    if baby_engine:
        logger.info("Running baby-sglang inference...")
        baby_results, baby_time = run_baby_sglang_inference(baby_engine, prompts)
        logger.info(f"Baby-sglang completed in {baby_time:.2f}s")
    else:
        baby_results, baby_time = [], 0.0
    
    if vllm_engine:
        logger.info("Running vLLM inference...")
        vllm_results, vllm_time = run_vllm_inference(vllm_engine, prompts)
        logger.info(f"vLLM completed in {vllm_time:.2f}s")
    else:
        vllm_results, vllm_time = [], 0.0
    
    # Display results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        if baby_results and i < len(baby_results):
            print(f"Baby-sglang: {baby_results[i]}")
        else:
            print("Baby-sglang: [Failed]")
            
        if vllm_results and i < len(vllm_results):
            print(f"vLLM:        {vllm_results[i]}")
        else:
            print("vLLM:        [Failed]")
    
    # Performance comparison
    print(f"\nPerformance:")
    if baby_time > 0 and vllm_time > 0:
        if baby_time < vllm_time:
            print(f"Baby-sglang is {vllm_time/baby_time:.1f}x faster than vLLM")
        else:
            print(f"vLLM is {baby_time/vllm_time:.1f}x faster than baby-sglang")
    else:
        print(f"Baby-sglang: {baby_time:.2f}s")
        print(f"vLLM:        {vllm_time:.2f}s")

def quick_demo():
    """Quick demo with sample outputs."""
    device = get_best_device()
    prompts = DEMO_PROMPTS[:2]  # Use only first 2 prompts
    
    try:
        engine = setup_baby_sglang_engine(MODEL_NAME, device)
        results, inf_time = run_baby_sglang_inference(engine, prompts)
        
        print("\n" + "="*40)
        print("BABY-SGLANG DEMO")
        print("="*40)
        
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"{i}. {prompt}")
            print(f"   -> {result}\n")
        
        print(f"Time: {inf_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

def main():
    """Main function."""
    import sys
    
    logger.info("Starting baby-sglang demo")
    logger.info(f"Model: {MODEL_NAME}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        quick_demo()
    else:
        compare_engines(DEMO_PROMPTS)

if __name__ == "__main__":
    main()