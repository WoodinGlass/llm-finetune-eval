"""Model + tokenizer + LoRA setup (QLoRA 4-bit)."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unsupported dtype: {name!r}")
    return _DTYPES[name]


def load_tokenizer(cfg: Any) -> Any:
    """Load and configure tokenizer (padding side, pad token)."""
    tok = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tok.padding_side = cfg.model.padding_side
    tok.truncation_side = cfg.model.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: Any) -> Any:
    """Load base model in 4-bit QLoRA mode."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.model.load_in_4bit),
        bnb_4bit_quant_type=str(cfg.model.bnb_4bit_quant_type),
        bnb_4bit_compute_dtype=_resolve_dtype(str(cfg.model.bnb_4bit_compute_dtype)),
        bnb_4bit_use_double_quant=bool(cfg.model.bnb_4bit_use_double_quant),
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model.config.pretraining_tp = 1
    return model


def prepare_model(model: Any, cfg: Any) -> Any:
    """Wrap in k-bit training prep + attach LoRA adapter."""
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        gradient_checkpointing_kwargs=dict(cfg.training.gradient_checkpointing_kwargs),
    )

    lora = LoraConfig(
        r=int(cfg.model.lora.r),
        lora_alpha=int(cfg.model.lora.lora_alpha),
        lora_dropout=float(cfg.model.lora.lora_dropout),
        bias=str(cfg.model.lora.bias),
        task_type=str(cfg.model.lora.task_type),
        target_modules=list(cfg.model.lora.target_modules),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model
