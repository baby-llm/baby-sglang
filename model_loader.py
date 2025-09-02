"""
Model loader for baby-sgl: load HuggingFace Qwen2 weights into simplified BabyQwen2 model.

This module mirrors the verified implementation's loading capability, extracting weights
from an HF checkpoint and mapping them into our simplified architecture with merged QKV
and merged MLP Gate/Up projections.

Public APIs:
- BabyQwenConfig: dataclass that adapts HF config to our simplified model
- build_model_from_hf(model_id: str, device: str = "auto") -> BabyQwen2ForCausalLM
- load_weights_from_hf(model: BabyQwen2ForCausalLM, hf_model_path: str, device: str = "auto") -> Tuple[str, torch.dtype]
- load_weights(model: BabyQwen2ForCausalLM, weights: Mapping[str, torch.Tensor]) -> None
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

# Make local imports work without requiring a package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen2 import BabyQwen2ForCausalLM  # noqa: E402

logger = logging.getLogger(__name__)


# ========= Config adapter =========

@dataclass
class BabyQwenConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, model_id: str):
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cls(
            hidden_size=getattr(hf_cfg, "hidden_size"),
            num_attention_heads=getattr(hf_cfg, "num_attention_heads"),
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", getattr(hf_cfg, "num_kv_heads", None)),
            intermediate_size=getattr(hf_cfg, "intermediate_size"),
            num_hidden_layers=getattr(hf_cfg, "num_hidden_layers"),
            vocab_size=getattr(hf_cfg, "vocab_size"),
            rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
            hidden_act=getattr(hf_cfg, "hidden_act", "silu"),
            max_position_embeddings=getattr(hf_cfg, "max_position_embeddings", 32768),
            rope_theta=getattr(hf_cfg, "rope_theta", 1000000.0),
            rope_scaling=getattr(hf_cfg, "rope_scaling", None),
            tie_word_embeddings=getattr(hf_cfg, "tie_word_embeddings", False),
        )


# ========= Device / dtype helpers =========

def _best_device_str(device: str = "auto") -> Tuple[str, torch.dtype]:
    if device != "auto":
        dt = torch.float16 if device in ("cuda", "mps") else torch.float32
        return device, dt
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ========= HF loader entrypoints =========

def build_model_from_hf(model_id: str, device: str = "auto") -> BabyQwen2ForCausalLM:
    """
    Create BabyQwen2ForCausalLM from a HuggingFace model id by adapting the HF config
    and then copying weights into the simplified architecture.
    """
    cfg = BabyQwenConfig.from_hf(model_id)
    if cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads

    model = BabyQwen2ForCausalLM(cfg)
    model.eval()

    # Load weights and move to device/dtype
    dev, dt = load_weights_from_hf(model, model_id, device=device)
    model.to(device=dev, dtype=dt)
    return model


def load_weights_from_hf(
    model: BabyQwen2ForCausalLM,
    hf_model_path: str,
    device: str = "auto",
) -> Tuple[str, torch.dtype]:
    """
    Load weights from a HuggingFace checkpoint into the simplified model.
    Chooses dtype by device: float16 for CUDA/MPS, float32 for CPU.
    Avoids requiring `accelerate` by not using `device_map`.
    """
    try:
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading Qwen2 weights from: {hf_model_path}")

        # Resolve device and dtype
        target_device, torch_dtype = _best_device_str(device)

        # Load HF model on CPU to read state dict (no accelerate needed)
        logger.info("Loading HuggingFace model (CPU) to fetch state_dict...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # ensure standard torch.load path without accelerate
        )
        state = hf_model.state_dict()
        logger.info(f"Loaded {len(state)} tensors from HuggingFace model")

        # Copy into our simplified model
        load_weights(model, state)

        # Cleanup big ref
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Qwen2 weights from {hf_model_path} onto {target_device} with dtype={torch_dtype}")
        return target_device, torch_dtype

    except Exception as e:
        logger.error(f"Failed to load HF weights: {e}")
        raise


# ========= State dict mapping into simplified structure =========

def load_weights(model: BabyQwen2ForCausalLM, weights: Mapping[str, torch.Tensor]) -> None:
    """
    Load weights from a HuggingFace state_dict into the simplified model.
    Supports merged QKV and merged Gate/Up projections.
    """
    state = dict(weights) if not isinstance(weights, dict) else weights
    params = dict(model.named_parameters())

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    q_size = cfg.num_attention_heads * head_dim
    kv_size = cfg.num_key_value_heads * head_dim
    inter = cfg.intermediate_size

    def copy_param(dst_name: str, src_tensor: torch.Tensor):
        p = params.get(dst_name, None)
        if p is None:
            # Some weights may not exist (e.g., if tied head)
            return
        if p.data.shape != src_tensor.shape:
            logger.warning(f"Size mismatch for {dst_name}: {tuple(p.data.shape)} vs {tuple(src_tensor.shape)}")
            return
        p.data.copy_(src_tensor.to(p.dtype))

    # Top-level weights: embedding, final norm, lm_head
    for name, tensor in state.items():
        if name == "model.embed_tokens.weight":
            copy_param("model.embed_tokens.weight", tensor)
        elif name == "model.norm.weight":
            copy_param("model.norm.weight", tensor)
        elif name == "lm_head.weight":
            # Copy only if not tied to embeddings
            if hasattr(model, "lm_head") and (not hasattr(model.model, "embed_tokens") or model.lm_head is not model.model.embed_tokens):
                copy_param("lm_head.weight", tensor)

    # Per-layer weights: use regex to parse layer index and suffix
    layer_pat = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

    for name, tensor in state.items():
        m = layer_pat.match(name)
        if not m:
            # Skip unknown keys (e.g., rotary caches or vision/projector if any)
            continue

        lid = int(m.group(1))
        rest = m.group(2)

        # Attention QKV merged into: model.layers.{lid}.self_attn.qkv_proj.qkv_proj.{weight,bias}
        if rest == "self_attn.q_proj.weight":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
            if p is not None:
                p.data[:q_size].copy_(tensor.to(p.dtype))
            continue
        if rest == "self_attn.q_proj.bias":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
            if p is not None:
                p.data[:q_size].copy_(tensor.to(p.dtype))
            continue

        if rest == "self_attn.k_proj.weight":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
            if p is not None:
                p.data[q_size:q_size + kv_size].copy_(tensor.to(p.dtype))
            continue
        if rest == "self_attn.k_proj.bias":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
            if p is not None:
                p.data[q_size:q_size + kv_size].copy_(tensor.to(p.dtype))
            continue

        if rest == "self_attn.v_proj.weight":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.weight", None)
            if p is not None:
                p.data[q_size + kv_size:].copy_(tensor.to(p.dtype))
            continue
        if rest == "self_attn.v_proj.bias":
            p = params.get(f"model.layers.{lid}.self_attn.qkv_proj.qkv_proj.bias", None)
            if p is not None:
                p.data[q_size + kv_size:].copy_(tensor.to(p.dtype))
            continue

        if rest == "self_attn.o_proj.weight":
            copy_param(f"model.layers.{lid}.self_attn.o_proj.weight", tensor)
            continue
        if rest == "self_attn.o_proj.bias":
            copy_param(f"model.layers.{lid}.self_attn.o_proj.bias", tensor)
            continue

        # MLP merged Gate/Up into: model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.{weight,bias}
        if rest == "mlp.gate_proj.weight":
            p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.weight", None)
            if p is not None:
                p.data[:inter].copy_(tensor.to(p.dtype))
            continue
        if rest == "mlp.gate_proj.bias":
            p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.bias", None)
            if p is not None:
                p.data[:inter].copy_(tensor.to(p.dtype))
            continue

        if rest == "mlp.up_proj.weight":
            p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.weight", None)
            if p is not None:
                p.data[inter:].copy_(tensor.to(p.dtype))
            continue
        if rest == "mlp.up_proj.bias":
            p = params.get(f"model.layers.{lid}.mlp.gate_up_proj.gate_up_proj.bias", None)
            if p is not None:
                p.data[inter:].copy_(tensor.to(p.dtype))
            continue

        if rest == "mlp.down_proj.weight":
            copy_param(f"model.layers.{lid}.mlp.down_proj.weight", tensor)
            continue
        if rest == "mlp.down_proj.bias":
            copy_param(f"model.layers.{lid}.mlp.down_proj.bias", tensor)
            continue

        # Norms
        if rest == "input_layernorm.weight":
            copy_param(f"model.layers.{lid}.input_layernorm.weight", tensor)
            continue
        if rest == "post_attention_layernorm.weight":
            copy_param(f"model.layers.{lid}.post_attention_layernorm.weight", tensor)
            continue


# ========= Optional CLI for quick sanity check =========

def _cli():
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Build simplified BabyQwen2 model from HF and run a quick prefill.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    args = parser.parse_args()

    model = build_model_from_hf(args.model_id, device="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

    enc = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].squeeze(0)

    # Resolve device/dtype from model
    p = next(model.parameters())
    device = p.device
    input_ids = input_ids.to(device)

    # Minimal forward (prefill only): the caller should prepare forward_batch externally.
    print("=" * 40)
    print("Model loaded successfully via model_loader.")
    print(f"Prompt tokens: {input_ids.tolist()[:16]}{'...' if input_ids.numel() > 16 else ''}")
    print("=" * 40)


if __name__ == "__main__":
    _cli()