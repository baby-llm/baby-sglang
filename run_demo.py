"""
Demo script for baby-sglang with HuggingFace Qwen2-1.5B model loading and vLLM comparison.

This script demonstrates:
1. Loading HuggingFace Qwen2-1.5B model using baby-sglang
2. Running comprehensive demo prompts 
3. Comparing results and performance with vLLM
4. Detailed performance analysis and reporting

Usage:
    python run_demo.py
    
Requirements:
    - transformers>=4.30.0
    - vllm>=0.4.0
    - torch>=2.0.0
"""

import logging
import time
import statistics
from typing import List, Dict, Any, Tuple, Optional
import torch
import traceback
from engine import create_engine_from_hf, create_test_engine, GenerationEngine, SimplifiedSamplingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen2-1.5B"
MODEL_PATH = None  # Will be set after download

# Demo prompts for comprehensive testing
DEMO_PROMPTS = [
    # Simple completion
    "Hello, how are you?",
    "The capital of France is",
    "In machine learning,",
    
    # Reasoning and knowledge
    "Explain the concept of neural networks in simple terms:",
    "What are the main differences between Python and Java?",
    "Solve this step by step: What is 15% of 240?",
    
    # Creative writing
    "Write a short story about a robot learning to paint:",
    "Complete this poem: 'The sun rises high above the mountain tops,'",
    
    # Code generation
    "Write a Python function to calculate fibonacci numbers:",
    "How do you create a REST API in Python using Flask?",
    
    # Conversational
    "Tell me a joke about programming",
    "What advice would you give to someone learning to code?",
]

# Sampling configurations for testing
SAMPLING_CONFIGS = [
    {"name": "greedy", "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 50},
    {"name": "creative", "temperature": 0.8, "top_p": 0.9, "max_new_tokens": 100},
    {"name": "balanced", "temperature": 0.5, "top_p": 0.95, "max_new_tokens": 75},
]

class PerformanceMetrics:
    """
    Class to track and calculate performance metrics.
    """
    
    def __init__(self):
        self.setup_times: List[float] = []
        self.inference_times: List[float] = []
        self.tokens_per_second: List[float] = []
        self.memory_usage: List[float] = []
        
    def add_setup_time(self, time_seconds: float):
        self.setup_times.append(time_seconds)
        
    def add_inference_time(self, time_seconds: float):
        self.inference_times.append(time_seconds)
        
    def add_throughput(self, tokens_per_second: float):
        self.tokens_per_second.append(tokens_per_second)
        
    def add_memory_usage(self, memory_mb: float):
        self.memory_usage.append(memory_mb)
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "setup_time_avg": statistics.mean(self.setup_times) if self.setup_times else 0,
            "inference_time_avg": statistics.mean(self.inference_times) if self.inference_times else 0,
            "inference_time_std": statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0,
            "throughput_avg": statistics.mean(self.tokens_per_second) if self.tokens_per_second else 0,
            "throughput_std": statistics.stdev(self.tokens_per_second) if len(self.tokens_per_second) > 1 else 0,
            "memory_avg": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "memory_peak": max(self.memory_usage) if self.memory_usage else 0,
        }


def get_best_device() -> str:
    """
    Auto-detect the best available device.
    
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"


def get_gpu_memory_usage() -> float:
    """
    Get current GPU memory usage in MB.
    Returns 0 if no GPU available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, return 0
        return 0.0
    else:
        return 0.0


def download_model_if_needed(model_name: str) -> str:
    """
    Download HuggingFace model if needed and return local path.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2-1.5B")
        
    Returns:
        Local path to the model
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Downloading/checking model: {model_name}")
        
        # This will download to HF cache if not already present
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_info = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=None  # Don't load to device yet
        )
        
        # Get cache directory 
        cache_dir = tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else model_name
        
        logger.info(f"Model available at: {cache_dir}")
        return model_name  # Return model name for HF loading
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def setup_baby_sglang_engine(model_path: str, device: str) -> Tuple[GenerationEngine, float]:
    """
    Setup baby-sglang engine with HuggingFace model.
    
    Args:
        model_path: Path to HuggingFace model
        device: Device to use
        
    Returns:
        Tuple of (engine, setup_time)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Setting up baby-sglang engine with {model_path} on {device}")
        
        engine = create_engine_from_hf(
            hf_model_path=model_path,
            device=device,
            max_batch_size=8,
            max_total_tokens=2048,
            max_context_len=512
        )
        
        setup_time = time.time() - start_time
        logger.info(f"Baby-sglang engine setup completed in {setup_time:.2f}s")
        
        return engine, setup_time
        
    except Exception as e:
        logger.error(f"Failed to setup baby-sglang engine: {e}")
        raise


def setup_vllm_engine(model_path: str, device: str) -> Tuple[Any, float]:
    """
    Setup vLLM engine with HuggingFace model.
    
    Args:
        model_path: Path to HuggingFace model  
        device: Device to use
        
    Returns:
        Tuple of (llm_engine, setup_time)
    """
    start_time = time.time()
    
    try:
        from vllm import LLM, SamplingParams
        
        logger.info(f"Setting up vLLM engine with {model_path} on {device}")
        
        # Configure vLLM
        if device == "cuda":
            tensor_parallel_size = 1
            gpu_memory_utilization = 0.7
        else:
            tensor_parallel_size = 1 
            gpu_memory_utilization = 0.7
            
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=512,  # Match baby-sglang context length
            dtype="float16"
        )
        
        setup_time = time.time() - start_time
        logger.info(f"vLLM engine setup completed in {setup_time:.2f}s")
        
        return llm, setup_time
        
    except Exception as e:
        logger.error(f"Failed to setup vLLM engine: {e}")
        raise

def run_baby_sglang_inference(
    engine: GenerationEngine,
    prompts: List[str],
    sampling_config: Dict[str, Any]
) -> Tuple[List[str], float, float]:
    """
    Run inference using baby-sglang engine.
    
    Args:
        engine: Baby-sglang engine
        prompts: List of input prompts
        sampling_config: Sampling parameters
        
    Returns:
        Tuple of (outputs, inference_time, throughput)
    """
    sampling_params = SimplifiedSamplingParams(
        max_new_tokens=sampling_config["max_new_tokens"],
        temperature=sampling_config["temperature"],
        top_p=sampling_config["top_p"],
    )
    
    start_time = time.time()
    mem_before = get_gpu_memory_usage()
    
    try:
        results = engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_real_tokenizer=True
        )
        
        inference_time = time.time() - start_time
        
        # Calculate throughput (tokens per second)
        total_output_tokens = sum(len(result.split()) for result in results)  # Approximate
        throughput = total_output_tokens / inference_time if inference_time > 0 else 0
        
        mem_after = get_gpu_memory_usage()
        logger.info(f"Baby-sglang inference completed in {inference_time:.2f}s, throughput: {throughput:.1f} tok/s")
        logger.info(f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        
        return results, inference_time, throughput
        
    except Exception as e:
        logger.error(f"Baby-sglang inference failed: {e}")
        raise


def run_vllm_inference(
    llm_engine: Any,
    prompts: List[str], 
    sampling_config: Dict[str, Any]
) -> Tuple[List[str], float, float]:
    """
    Run inference using vLLM engine.
    
    Args:
        llm_engine: vLLM engine
        prompts: List of input prompts
        sampling_config: Sampling parameters
        
    Returns:
        Tuple of (outputs, inference_time, throughput)
    """
    try:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=sampling_config["max_new_tokens"],
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
        )
        
        start_time = time.time()
        mem_before = get_gpu_memory_usage()
        
        outputs = llm_engine.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            full_text = output.prompt + generated_text  # Combine prompt and generated
            results.append(full_text)
        
        inference_time = time.time() - start_time
        
        # Calculate throughput
        total_output_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
        throughput = total_output_tokens / inference_time if inference_time > 0 else 0
        
        mem_after = get_gpu_memory_usage()
        logger.info(f"vLLM inference completed in {inference_time:.2f}s, throughput: {throughput:.1f} tok/s")
        logger.info(f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        
        return results, inference_time, throughput
        
    except Exception as e:
        logger.error(f"vLLM inference failed: {e}")
        raise


def run_comparison_test(
    baby_sglang_engine: GenerationEngine,
    vllm_engine: Any,
    prompts: List[str],
    sampling_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a comprehensive comparison test between baby-sglang and vLLM.
    
    Args:
        baby_sglang_engine: Baby-sglang engine
        vllm_engine: vLLM engine
        prompts: Test prompts
        sampling_config: Sampling configuration
        
    Returns:
        Comparison results
    """
    logger.info(f"Running comparison test with {len(prompts)} prompts")
    logger.info(f"Sampling config: {sampling_config}")
    
    results = {
        "sampling_config": sampling_config,
        "num_prompts": len(prompts),
        "baby_sglang": {},
        "vllm": {},
        "comparison": {}
    }
    
    # Test baby-sglang
    logger.info("Running baby-sglang inference...")
    try:
        baby_outputs, baby_time, baby_throughput = run_baby_sglang_inference(
            baby_sglang_engine, prompts, sampling_config
        )
        results["baby_sglang"] = {
            "outputs": baby_outputs,
            "inference_time": baby_time,
            "throughput": baby_throughput,
            "success": True
        }
    except Exception as e:
        logger.error(f"Baby-sglang failed: {e}")
        results["baby_sglang"] = {
            "error": str(e),
            "success": False
        }
    
    # Test vLLM
    logger.info("Running vLLM inference...")
    try:
        vllm_outputs, vllm_time, vllm_throughput = run_vllm_inference(
            vllm_engine, prompts, sampling_config
        )
        results["vllm"] = {
            "outputs": vllm_outputs,
            "inference_time": vllm_time,
            "throughput": vllm_throughput,
            "success": True
        }
    except Exception as e:
        logger.error(f"vLLM failed: {e}")
        results["vllm"] = {
            "error": str(e),
            "success": False
        }
    
    # Compare results if both succeeded
    if results["baby_sglang"]["success"] and results["vllm"]["success"]:
        baby_time = results["baby_sglang"]["inference_time"]
        vllm_time = results["vllm"]["inference_time"]
        baby_throughput = results["baby_sglang"]["throughput"]
        vllm_throughput = results["vllm"]["throughput"]
        
        results["comparison"] = {
            "speed_ratio": vllm_time / baby_time if baby_time > 0 else float('inf'),
            "throughput_ratio": baby_throughput / vllm_throughput if vllm_throughput > 0 else float('inf'),
            "faster_engine": "baby-sglang" if baby_time < vllm_time else "vLLM",
            "higher_throughput": "baby-sglang" if baby_throughput > vllm_throughput else "vLLM"
        }
        
        # Compare output similarity (basic text similarity)
        baby_outputs = results["baby_sglang"]["outputs"]
        vllm_outputs = results["vllm"]["outputs"]
        
        similarities = []
        for baby_out, vllm_out in zip(baby_outputs, vllm_outputs):
            # Simple word overlap similarity
            baby_words = set(baby_out.lower().split())
            vllm_words = set(vllm_out.lower().split())
            if len(baby_words) == 0 and len(vllm_words) == 0:
                similarity = 1.0
            elif len(baby_words) == 0 or len(vllm_words) == 0:
                similarity = 0.0
            else:
                intersection = len(baby_words.intersection(vllm_words))
                union = len(baby_words.union(vllm_words))
                similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
            
        results["comparison"]["avg_similarity"] = statistics.mean(similarities) if similarities else 0.0
        
    return results


def format_comparison_results(all_results: List[Dict[str, Any]]) -> str:
    """
    Format comparison results into a readable report.
    
    Args:
        all_results: List of comparison results
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("\n" + "="*80)
    report.append("           BABY-SGLANG vs vLLM COMPARISON RESULTS")
    report.append("="*80)
    
    for i, result in enumerate(all_results):
        config = result["sampling_config"]
        report.append(f"\n### Test {i+1}: {config['name'].upper()} Sampling ###")
        report.append(f"Config: temp={config['temperature']}, top_p={config['top_p']}, max_tokens={config['max_new_tokens']}")
        report.append(f"Prompts: {result['num_prompts']}")
        
        # Performance comparison
        if result["baby_sglang"]["success"] and result["vllm"]["success"]:
            baby_time = result["baby_sglang"]["inference_time"]
            vllm_time = result["vllm"]["inference_time"]
            baby_throughput = result["baby_sglang"]["throughput"]
            vllm_throughput = result["vllm"]["throughput"]
            
            report.append("\nPerformance Results:")
            report.append(f"  Baby-sglang: {baby_time:.2f}s, {baby_throughput:.1f} tok/s")
            report.append(f"  vLLM:        {vllm_time:.2f}s, {vllm_throughput:.1f} tok/s")
            
            comparison = result["comparison"]
            speed_ratio = comparison["speed_ratio"]
            throughput_ratio = comparison["throughput_ratio"]
            
            if speed_ratio < 1.0:
                report.append(f"  🚀 Baby-sglang is {1/speed_ratio:.1f}x FASTER than vLLM")
            else:
                report.append(f"  🐌 vLLM is {speed_ratio:.1f}x faster than baby-sglang")
                
            if throughput_ratio > 1.0:
                report.append(f"  📈 Baby-sglang has {throughput_ratio:.1f}x HIGHER throughput than vLLM")
            else:
                report.append(f"  📉 vLLM has {1/throughput_ratio:.1f}x higher throughput than baby-sglang")
                
            report.append(f"  🎯 Output similarity: {comparison['avg_similarity']:.1%}")
            
        else:
            report.append("\nPerformance Results:")
            if not result["baby_sglang"]["success"]:
                report.append(f"  ❌ Baby-sglang failed: {result['baby_sglang'].get('error', 'Unknown error')}")
            if not result["vllm"]["success"]:  
                report.append(f"  ❌ vLLM failed: {result['vllm'].get('error', 'Unknown error')}")
    
    # Overall summary
    report.append("\n" + "-"*80)
    report.append("OVERALL SUMMARY")
    report.append("-"*80)
    
    successful_tests = [r for r in all_results if r["baby_sglang"]["success"] and r["vllm"]["success"]]
    if successful_tests:
        avg_baby_time = statistics.mean([r["baby_sglang"]["inference_time"] for r in successful_tests])
        avg_vllm_time = statistics.mean([r["vllm"]["inference_time"] for r in successful_tests])
        avg_baby_throughput = statistics.mean([r["baby_sglang"]["throughput"] for r in successful_tests])
        avg_vllm_throughput = statistics.mean([r["vllm"]["throughput"] for r in successful_tests])
        avg_similarity = statistics.mean([r["comparison"]["avg_similarity"] for r in successful_tests])
        
        report.append(f"Average Performance ({len(successful_tests)} successful tests):")
        report.append(f"  Baby-sglang: {avg_baby_time:.2f}s avg, {avg_baby_throughput:.1f} tok/s avg")
        report.append(f"  vLLM:        {avg_vllm_time:.2f}s avg, {avg_vllm_throughput:.1f} tok/s avg")
        report.append(f"  Average similarity: {avg_similarity:.1%}")
        
        if avg_baby_time < avg_vllm_time:
            report.append(f"  🏆 Baby-sglang is {avg_vllm_time/avg_baby_time:.1f}x faster on average")
        else:
            report.append(f"  🏆 vLLM is {avg_baby_time/avg_vllm_time:.1f}x faster on average")
            
    failed_tests = len(all_results) - len(successful_tests)
    if failed_tests > 0:
        report.append(f"  ⚠️ {failed_tests} tests failed")
        
    report.append("\n" + "="*80)
    
    return "\n".join(report)

def main():
    """
    Main function to run comprehensive baby-sglang vs vLLM comparison.
    """
    logger.info("Starting comprehensive baby-sglang vs vLLM comparison")
    logger.info(f"Model: {MODEL_NAME}")
    
    device = get_best_device()
    logger.info(f"Using device: {device}")
    
    # Step 1: Download/prepare model
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Model Preparation")
    logger.info("="*50)
    
    try:
        model_path = download_model_if_needed(MODEL_NAME)
        logger.info(f"✓ Model ready: {model_path}")
    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        return False
    
    # Step 2: Setup engines
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Engine Setup")
    logger.info("="*50)
    
    baby_sglang_engine = None
    vllm_engine = None
    
    # Setup baby-sglang
    try:
        baby_sglang_engine, baby_setup_time = setup_baby_sglang_engine(model_path, device)
        logger.info(f"✓ Baby-sglang engine ready ({baby_setup_time:.1f}s)")
    except Exception as e:
        logger.error(f"Failed to setup baby-sglang: {e}")
        logger.error("Continuing with vLLM-only testing...")
    
    # Setup vLLM
    try:
        vllm_engine, vllm_setup_time = setup_vllm_engine(model_path, device) 
        logger.info(f"✓ vLLM engine ready ({vllm_setup_time:.1f}s)")
    except Exception as e:
        logger.error(f"Failed to setup vLLM: {e}")
        if baby_sglang_engine is None:
            logger.error("Both engines failed to initialize. Exiting.")
            return False
        logger.error("Continuing with baby-sglang-only testing...")
    
    # Step 3: Run comparison tests
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Comparison Testing")
    logger.info("="*50)
    
    all_results = []
    
    for i, sampling_config in enumerate(SAMPLING_CONFIGS):
        logger.info(f"\nRunning test {i+1}/{len(SAMPLING_CONFIGS)}: {sampling_config['name']}")
        
        if baby_sglang_engine and vllm_engine:
            try:
                result = run_comparison_test(
                    baby_sglang_engine,
                    vllm_engine,
                    DEMO_PROMPTS[:5],  # Use subset for faster testing
                    sampling_config
                )
                all_results.append(result)
                logger.info(f"✓ Test {i+1} completed successfully")
            except Exception as e:
                logger.error(f"Test {i+1} failed: {e}")
                continue
        else:
            logger.warning(f"Skipping test {i+1} - engines not available")
    
    # Step 4: Display results
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Results")
    logger.info("="*50)
    
    if all_results:
        report = format_comparison_results(all_results)
        print(report)
        logger.info("Comparison completed successfully!")
        
        # Save results to file
        try:
            import json
            with open("comparison_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info("Results saved to comparison_results.json")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
            
        return True
    else:
        logger.error("No successful comparison tests completed")
        return False


def demo_sample_outputs():
    """
    Show sample outputs for a few prompts to demonstrate functionality.
    """
    logger.info("\n" + "="*50)
    logger.info("SAMPLE OUTPUT DEMONSTRATION")
    logger.info("="*50)
    
    device = get_best_device()
    
    # Use simple prompts
    sample_prompts = [
        "Hello, how are you?",
        "Explain machine learning in simple terms:",
        "Write a Python function to add two numbers:"
    ]
    
    sampling_config = {
        "name": "demo",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 80
    }
    
    try:
        model_path = download_model_if_needed(MODEL_NAME)
        baby_engine, _ = setup_baby_sglang_engine(model_path, device)
        
        logger.info("Running sample generation with baby-sglang:")
        
        results, inf_time, throughput = run_baby_sglang_inference(
            baby_engine, sample_prompts, sampling_config
        )
        
        print("\n" + "-"*60)
        print("SAMPLE OUTPUTS FROM BABY-SGLANG:")
        print("-"*60)
        
        for i, (prompt, result) in enumerate(zip(sample_prompts, results), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Output: {result}")
            print()
        
        print(f"Performance: {inf_time:.2f}s, {throughput:.1f} tok/s")
        print("-"*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Quick demo mode with sample outputs
        demo_sample_outputs()
    else:
        # Full comparison mode
        success = main()
        sys.exit(0 if success else 1)