"""Unit tests for Hydra config loading + required keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(REPO / "configs")


def _compose(overrides=None):
    """Compose with absolute config_dir (avoids Hydra's caller-relative quirk)."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


def test_root_config_composes():
    cfg = _compose()
    assert cfg.run.seed == 42
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert cfg.data.train_dir == "data/processed/train"


def test_model_config_has_quant_and_lora():
    cfg = _compose()
    assert cfg.model.load_in_4bit is True
    assert cfg.model.bnb_4bit_quant_type == "nf4"
    assert cfg.model.bnb_4bit_compute_dtype == "float16"
    assert cfg.model.lora.r == 16
    assert "q_proj" in cfg.model.lora.target_modules
    assert "gate_proj" in cfg.model.lora.target_modules


def test_data_config_lists_all_domains():
    cfg = _compose()
    for k in ("math", "physics", "ood", "adversarial_ood"):
        assert k in cfg.data.train_files
        assert cfg.data.train_files[k].startswith("data/processed/train/")


def test_system_prompts_present_for_math_and_physics():
    cfg = _compose()
    assert "step by step" in cfg.data.system_prompts.math.lower()
    assert "physics expert" in cfg.data.system_prompts.physics.lower()
    assert cfg.data.system_prompts.ood is None
    assert cfg.data.system_prompts.adversarial_ood is None


def test_qlora_base_composes_via_override():
    cfg = _compose(["+train=qlora_base"])
    assert cfg.training.num_train_epochs == 3
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.optim == "paged_adamw_32bit"
    assert cfg.training.fp16 is True
    assert cfg.training.bf16 is False
    assert cfg.training.gradient_checkpointing is True
    assert "wandb" in cfg.training.report_to


def test_smoke_inherits_and_overrides():
    cfg = _compose(["+train=smoke"])
    assert cfg.model.max_seq_length == 512
    assert cfg.model.lora.r == 8
    assert cfg.data.max_samples == 100
    assert cfg.training.num_train_epochs == 1
    assert cfg.training.optim == "paged_adamw_32bit"
    assert cfg.training.fp16 is True
    assert cfg.training.report_to == ["wandb"]


def test_output_dir_resolver_uses_env(monkeypatch):
    monkeypatch.setenv("LLM_FT_CHECKPOINT_DIR", "/tmp/ckpt-test")
    cfg = _compose(["+train=qlora_base"])
    assert "/tmp/ckpt-test" in cfg.run.output_dir
    assert cfg.run.output_dir.endswith("/m2")


def test_no_bf16_anywhere():
    for train_name in ("qlora_base", "smoke"):
        cfg = _compose([f"+train={train_name}"])
        assert cfg.training.bf16 is False, f"{train_name}: bf16 must be False"
