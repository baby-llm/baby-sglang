"""
Real tokenizer implementation for baby-sglang.

Provides SGLang-compatible tokenization interface using tokenizers library.

References:
- HuggingFace tokenizers library  
- SGLang tokenizer patterns
"""

import logging
import os
from typing import List, Optional, Union, Dict, Any
import json

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.decoders import ByteLevel
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    Tokenizer = None

logger = logging.getLogger(__name__)


class BabyTokenizer:
    """
    Real tokenizer implementation for baby-sglang.
    
    Provides encode/decode interface compatible with SGLang patterns.
    """
    
    def __init__(
        self, 
        model_path_or_name: str = "Qwen/Qwen2-0.5B-Instruct",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path_or_name: HuggingFace model name or local path
            tokenizer_mode: Tokenizer mode (auto, slow, fast)
            trust_remote_code: Whether to trust remote code
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "tokenizers library not available. Install with: pip install tokenizers"
            )
        
        self.model_path_or_name = model_path_or_name
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        
        # Initialize tokenizer
        self.tokenizer = None
        self.vocab_size = 0
        self.eos_token_id = None
        self.pad_token_id = None
        self.bos_token_id = None
        self.unk_token_id = None
        
        # Special token mappings
        self._special_tokens = {}
        
        # Load tokenizer
        self._load_tokenizer()
        
        logger.info(f"BabyTokenizer initialized: {model_path_or_name}")
        logger.info(f"Vocab size: {self.vocab_size}")
        logger.info(f"Special tokens: eos={self.eos_token_id}, pad={self.pad_token_id}")
    
    def _load_tokenizer(self):
        """Load tokenizer from model path or HuggingFace hub."""
        if os.path.exists(self.model_path_or_name):
            # Local path
            self._load_from_local_path(self.model_path_or_name)
        else:
            # Try to load from HuggingFace hub  
            self._load_from_hf_hub(self.model_path_or_name)
    
    def _load_from_local_path(self, path: str):
        """Load tokenizer from local directory."""
        tokenizer_json_path = os.path.join(path, "tokenizer.json")
        
        if os.path.exists(tokenizer_json_path):
            # Load from tokenizer.json
            self.tokenizer = Tokenizer.from_file(tokenizer_json_path)
            logger.info(f"Loaded tokenizer from {tokenizer_json_path}")
        else:
            raise FileNotFoundError(f"tokenizer.json not found in {path}")
        
        # Load special tokens from tokenizer_config.json if available
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self._extract_special_tokens_from_config(config)
        
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def _load_from_hf_hub(self, model_name: str):
        """Load tokenizer from HuggingFace hub."""
        # For now, raise error since we need local tokenizer files
        raise NotImplementedError(
            f"HuggingFace hub loading not implemented. "
            f"Please download {model_name} locally and provide the path."
        )
    
    def _extract_special_tokens_from_config(self, config: Dict[str, Any]):
        """Extract special token IDs from tokenizer config."""
        # Common special token mappings
        special_token_map = {
            'eos_token': 'eos_token_id',
            'pad_token': 'pad_token_id', 
            'bos_token': 'bos_token_id',
            'unk_token': 'unk_token_id',
        }
        
        for token_name, attr_name in special_token_map.items():
            if token_name in config:
                token_info = config[token_name]
                if isinstance(token_info, dict) and 'content' in token_info:
                    token = token_info['content']
                elif isinstance(token_info, str):
                    token = token_info
                else:
                    continue
                
                # Get token ID
                token_id = self.tokenizer.token_to_id(token)
                if token_id is not None:
                    setattr(self, attr_name, token_id)
                    logger.debug(f"Set {attr_name} = {token_id} ('{token}')")
        
        # Set vocab size
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Use tokenizers library
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(
        self, 
        ids: Union[List[int], int], 
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token ID or list of token IDs
            skip_special_tokens: Whether to skip special tokens
            spaces_between_special_tokens: Whether to add spaces between special tokens
            
        Returns:
            Decoded text string
        """
        if isinstance(ids, int):
            ids = [ids]
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Use tokenizers library
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_tokens_dict(self) -> Dict[str, Optional[int]]:
        """Get dictionary of special token IDs."""
        return {
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'unk_token_id': self.unk_token_id,
        }


def create_tokenizer(
    model_path_or_name: str = "Qwen/Qwen2-0.5B-Instruct",
    **kwargs
) -> BabyTokenizer:
    """
    Factory function to create tokenizer instance.
    
    Convenient factory for creating real tokenizer.
    """
    return BabyTokenizer(model_path_or_name, **kwargs)


# Real Tokenizer Implementation Summary:
# ======================================
#
# Complete real tokenizer implementation using tokenizers library:
#
# 1. BabyTokenizer class:
#    - Loads tokenizer.json from local path 
#    - Extracts special tokens (eos, pad, bos, unk) from config
#    - Provides encode/decode interface compatible with SGLang
#    - Requires real tokenizer files - no fallback
#
# 2. Key methods:
#    - encode(text) -> List[int]: Tokenize text to IDs
#    - decode(ids) -> str: Detokenize IDs to text
#    - get_vocab_size() -> int: Get vocabulary size
#    - get_special_tokens_dict(): Get special token mappings
#
# 3. Features:
#    - Real tokenizers library integration
#    - Special token handling from tokenizer config
#    - SGLang-compatible interface
#    - Local model file support
#    - Proper error handling
#
# 4. Requirements:
#    - tokenizers library must be installed
#    - Local tokenizer files (tokenizer.json, tokenizer_config.json)
#    - No mock fallback - fails if real tokenizer unavailable