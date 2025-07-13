import argparse
import time
from engine import SGLangEngine

def main(args):
    """
    nano-sglang 的演示运行程序。
    """
    # 1. 初始化 SGLangEngine
    # SGLangEngine 将在后台启动 Scheduler 和 Detokenizer 进程。
    print("Initializing SGLangEngine...")
    engine = SGLangEngine(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or args.model_path,
        mem_fraction_static=args.mem_fraction,
        tp_size=1, # nano-sglang 不支持张量并行
    )
    print("Engine initialized.")

    try:
        # 2. 定义示例请求
        prompts = [
            "The capital of France is",
            "The quick brown fox jumps over the",
        ]
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 32,
        }

        print("\n--- Running Prompts ---")
        
        # 3. 提交请求并等待结果
        results = []
        for i, prompt in enumerate(prompts):
            request_id = f"request_{i}"
            print(f"Submitting request '{request_id}': '{prompt}'")
            
            start_time = time.time()
            result = engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            end_time = time.time()
            
            results.append(result)
            print(f"Result for '{request_id}': {result['text']}")
            print(f"Time taken: {end_time - start_time:.2f} seconds\n")
            
        print("\n--- All prompts processed ---")
    finally:
        # 4. 确保在程序结束时关闭引擎
        engine.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo runner for nano-sglang.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="gpt2",
        help="The path to the Hugging Face model folder or the model name on the Hub."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="The path to the Hugging Face tokenizer folder. If not provided, it defaults to `model_path`."
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.8,
        help="The fraction of GPU memory to be used for the KV cache."
    )
    
    args = parser.parse_args()
    main(args)