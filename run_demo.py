"""
Demo script for baby-sglang.

Simple example showing how to use the baby-sglang engine
for text generation.
"""

import asyncio
import logging

from engine import Engine
from utils.utils import setup_logger


async def main():
    """Main demo function."""
    # Setup logging
    logger = setup_logger("baby-sglang-demo", level=logging.INFO)
    logger.info("Starting baby-sglang demo")
    
    # TODO: Add command line argument parsing
    # TODO: Allow configurable model path and parameters
    
    # Initialize engine
    model_path = "/path/to/model"  # TODO: Use actual model path
    engine = Engine(
        model_path=model_path,
        max_batch_size=8,
        max_seq_len=1024
    )
    
    try:
        # Start engine
        await engine.start_async()
        logger.info("Engine started successfully")
        
        # Example generation
        prompt = "The future of AI is"
        logger.info(f"Generating response for: '{prompt}'")
        
        response = await engine.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        logger.info(f"Generated response: {response}")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
    finally:
        # Cleanup
        await engine.shutdown()
        logger.info("Demo completed")


if __name__ == "__main__":
    # TODO: Add proper argument parsing
    # TODO: Add configuration file support
    # TODO: Add interactive mode
    
    asyncio.run(main())