"""Unified secret loader.

Resolution order:
  1. platform-native store (Colab userdata / Kaggle secrets)
  2. os.environ (fallback for local dev & CI)
  3. explicit default

Usage:
    from llm_ft.secrets import get_secret, load_all, DEFAULT_KEYS
    load_all(DEFAULT_KEYS)               # populate os.environ, best-effort
    hf = get_secret("HF_TOKEN", required=True)

Note:
    This module imports platform-specific modules (google.colab, kaggle_secrets)
    behind try/except and is intentionally excluded from strict mypy checking
    because neither module ships type stubs. See pyproject.toml.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from llm_ft.env import IS_COLAB, IS_KAGGLE


def _from_colab(key: str) -> str | None:
    if not IS_COLAB:
        return None
    try:
        from google.colab import userdata
    except Exception:
        return None
    try:
        value = userdata.get(key)
    except Exception:
        return None
    return str(value) if value else None


def _from_kaggle(key: str) -> str | None:
    if not IS_KAGGLE:
        return None
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return None
    try:
        value = UserSecretsClient().get_secret(key)
    except Exception:
        return None
    return str(value) if value else None


def _from_env(key: str) -> str | None:
    value = os.environ.get(key)
    return value or None


def get_secret(
    key: str,
    *,
    required: bool = False,
    default: str | None = None,
) -> str | None:
    """Load a secret with platform-aware fallback chain."""
    value = _from_colab(key) if IS_COLAB else (_from_kaggle(key) if IS_KAGGLE else None)
    if value is None:
        value = _from_env(key)
    if value is None:
        value = default
    if required and not value:
        raise RuntimeError(f"Missing required secret: {key}")
    return value


def load_all(
    keys: Iterable[str],
    *,
    required: bool = False,
) -> dict[str, str | None]:
    """Load several secrets and inject them into os.environ (without overwrite)."""
    out: dict[str, str | None] = {}
    for k in keys:
        v = get_secret(k, required=required)
        out[k] = v
        if v is not None:
            os.environ.setdefault(k, v)
    return out


DEFAULT_KEYS: tuple[str, ...] = ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
