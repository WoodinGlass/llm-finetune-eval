"""Main training entry point (Hydra).

Usage (Colab / Kaggle):
    python training/train.py +train=smoke
    python training/train.py +train=qlora_base

Resume from checkpoint:
    python training/train.py +train=qlora_base \
        training.resume_from_checkpoint=/path/to/checkpoint-XXX

The script is deliberately linear:
    1. Hydra resolves config -> env-driven paths
    2. seed fixing (python, numpy, torch, cuda)
    3. load tokenizer + model + LoRA
    4. build HF dataset (with chat template applied)
    5. configure W&B + MLflow
    6. Trainer + callbacks
    7. train, save final adapter
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from training.data import build_hf_dataset, load_train_records  # noqa: E402
from training.model import load_model, load_tokenizer, prepare_model  # noqa: E402

log = logging.getLogger("training")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: deterministic algorithms intentionally NOT enabled; it breaks
    # bnb / flash-attn kernels. Seed is enough for our use.


def resolve_output_dir(cfg: DictConfig) -> Path:
    p = Path(OmegaConf.to_container(cfg.run, resolve=True)["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    (p / "final").mkdir(exist_ok=True)
    return p


def resolve_mlflow_dir(cfg: DictConfig) -> Path:
    p = Path(OmegaConf.to_container(cfg.run, resolve=True)["mlflow_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_callbacks(cfg: DictConfig):
    from transformers import EarlyStoppingCallback, TrainerCallback

    class PrintCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_local_process_zero:
                keys = ["loss", "learning_rate", "grad_norm", "epoch"]
                parts = [f"{k}={logs[k]:.4g}" for k in keys if k in logs]
                if parts:
                    log.info("step %5d | %s", state.global_step, " | ".join(parts))

    cbs = [PrintCallback()]
    if bool(cfg.training.get("early_stopping", False)):
        cbs.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.training.get("early_stopping_patience", 2))
            )
        )
    return cbs


def build_sft_kwargs(cfg: DictConfig, out_dir: Path, seed: int, report_to: list) -> dict:
    """Build SFTConfig kwargs as a literal dict (ruff C408 friendly)."""
    kwargs = {
        "loss_type": "nll",  # bypass TRL chunked_nll bug with PEFT
        "output_dir": str(out_dir),
        "num_train_epochs": float(cfg.training.num_train_epochs),
        "per_device_train_batch_size": int(cfg.training.per_device_train_batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "learning_rate": float(cfg.training.learning_rate),
        "lr_scheduler_type": str(cfg.training.lr_scheduler_type),
        "warmup_steps": int(cfg.training.warmup_steps),
        "weight_decay": float(cfg.training.weight_decay),
        "max_grad_norm": float(cfg.training.max_grad_norm),
        "optim": str(cfg.training.optim),
        "fp16": bool(cfg.training.fp16),
        "bf16": bool(cfg.training.bf16),
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "gradient_checkpointing_kwargs": dict(cfg.training.gradient_checkpointing_kwargs),
        "logging_steps": int(cfg.training.logging_steps),
        "save_steps": int(cfg.training.save_steps),
        "save_total_limit": int(cfg.training.save_total_limit),
        "save_safetensors": bool(cfg.training.save_safetensors),
        "dataloader_num_workers": int(cfg.training.dataloader_num_workers),
        "report_to": report_to,
        "seed": seed,
        "data_seed": int(cfg.training.data_seed),
        "dataset_text_field": "text",
        "max_length": int(cfg.model.max_seq_length),
        "packing": False,
        "run_name": str(cfg.run.name),
    }
    if cfg.training.get("neftune_noise_alpha") is not None:
        kwargs["neftune_noise_alpha"] = float(cfg.training.neftune_noise_alpha)
    return kwargs


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    seed = int(cfg.run.seed)
    set_seeds(seed)

    out_dir = resolve_output_dir(cfg)
    mlruns_dir = resolve_mlflow_dir(cfg)
    log.info("output_dir: %s", out_dir)
    log.info("mlflow_dir: %s", mlruns_dir)

    # 1. tokenizer + model
    log.info("loading tokenizer + model ...")
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    model = prepare_model(model, cfg)

    # ── QLoRA precision cast (T4 / fp16 GradScaler) ──
    # Qwen2.5-Math declares torch_dtype=bfloat16 in config.json, so PEFT
    # creates LoRA adapters in bf16. torch's fp16 GradScaler has no bf16
    # unscale kernel -> NotImplementedError at first clip_grad_norm_.
    # Standard recipe: trainable -> fp32, non-trainable buffers -> fp16.
    import torch as _torch
    _n_cast = 0
    for _p in model.parameters():
        if _p.dtype == _torch.bfloat16:
            _p.data = _p.data.to(_torch.float32 if _p.requires_grad else _torch.float16)
            _n_cast += 1
    for _b in model.buffers():
        if _b.dtype == _torch.bfloat16:
            _b.data = _b.data.to(_torch.float16)
            _n_cast += 1
    log.info("cast %d bf16 tensors (trainable->fp32, rest->fp16)", _n_cast)

    # Extra safety: paksa seluruh model ke fp16/fp32, tidak ada bf16
    _bf16_left = sum(1 for _p in model.parameters() if _p.dtype == _torch.bfloat16)
    _bf16_left += sum(1 for _b in model.buffers() if _b.dtype == _torch.bfloat16)
    if _bf16_left > 0:
        log.warning("still %d bf16 tensors — forcing fp16", _bf16_left)
        for _p in model.parameters():
            if _p.dtype == _torch.bfloat16:
                _p.data = _p.data.to(_torch.float32 if _p.requires_grad else _torch.float16)
        for _b in model.buffers():
            if _b.dtype == _torch.bfloat16:
                _b.data = _b.data.to(_torch.float16)

    # Extra safety: paksa seluruh model ke fp16/fp32, tidak ada bf16
    _bf16_left = sum(1 for _p in model.parameters() if _p.dtype == _torch.bfloat16)
    _bf16_left += sum(1 for _b in model.buffers() if _b.dtype == _torch.bfloat16)
    if _bf16_left > 0:
        log.warning("still %d bf16 tensors — forcing fp16", _bf16_left)
        for _p in model.parameters():
            if _p.dtype == _torch.bfloat16:
                _p.data = _p.data.to(_torch.float32 if _p.requires_grad else _torch.float16)
        for _b in model.buffers():
            if _b.dtype == _torch.bfloat16:
                _b.data = _b.data.to(_torch.float16)

    # ── QLoRA precision cast (T4 / fp16 GradScaler) ──
    # Qwen2.5-Math declares torch_dtype=bfloat16 in config.json, so PEFT
    # creates LoRA adapters in bf16. torch's fp16 GradScaler has no bf16
    # unscale kernel -> NotImplementedError at first clip_grad_norm_.
    # Standard recipe: trainable -> fp32, non-trainable buffers -> fp16.
    import torch as _torch
    _n_cast = 0
    for _p in model.parameters():
        if _p.dtype == _torch.bfloat16:
            _p.data = _p.data.to(_torch.float32 if _p.requires_grad else _torch.float16)
            _n_cast += 1
    for _b in model.buffers():
        if _b.dtype == _torch.bfloat16:
            _b.data = _b.data.to(_torch.float16)
            _n_cast += 1
    log.info("cast %d bf16 tensors (trainable->fp32, rest->fp16)", _n_cast)

    # 2. dataset
    log.info("loading train records ...")
    records = load_train_records(
        dict(cfg.data.train_files),
        max_samples=cfg.data.max_samples if cfg.data.max_samples else None,
        require_all_domains=bool(cfg.data.require_all_domains),
        seed=int(cfg.data.shuffle_seed),
    )
    log.info("  got %d records", len(records))
    from collections import Counter

    log.info("  domain split: %s", dict(Counter(r["domain"] for r in records)))

    hf_ds = build_hf_dataset(records, tokenizer, dict(cfg.data.system_prompts))
    log.info("  dataset size: %d", len(hf_ds))

    # 3. tracking
    report_to = list(cfg.training.report_to)
    if "mlflow" in report_to:
        import mlflow

        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        mlflow.set_experiment(cfg.run.project)
    if "wandb" in report_to:
        os.environ.setdefault("WANDB_PROJECT", cfg.run.project)
        os.environ.setdefault("WANDB_NAME", str(cfg.run.name))
        os.environ.setdefault("WANDB_TAGS", ",".join(cfg.run.tags))

    # 4. trainer
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(**build_sft_kwargs(cfg, out_dir, seed, report_to))
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_ds,
        processing_class=tokenizer,
        callbacks=build_callbacks(cfg),
    )

    # 5. train
    resume_from = cfg.training.get("resume_from_checkpoint", None)
    log.info("starting training (resume_from=%s)", resume_from)
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    log.info("training done: %s", train_result.metrics)

    # 6. save final adapter
    final_dir = out_dir / "final"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("final adapter saved to %s", final_dir)

    (final_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    raise SystemExit(main())
