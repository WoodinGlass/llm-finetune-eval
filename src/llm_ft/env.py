"""Environment detection and standard paths across Colab / Kaggle / local.

Import-safe: pure stdlib, no side effects beyond reading the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── platform detection ─────────────────────────────────────────
# Colab mounts /content and sets COLAB_GPU; Kaggle mounts /kaggle and
# sets KAGGLE_URL_BASE. Local is the fallback.
IS_COLAB: bool = bool(os.environ.get("COLAB_GPU")) or (
    Path("/content").is_dir() and not Path("/kaggle").is_dir()
)
IS_KAGGLE: bool = bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").is_dir()
IS_LOCAL: bool = not (IS_COLAB or IS_KAGGLE)

PLATFORM: str = "colab" if IS_COLAB else "kaggle" if IS_KAGGLE else "local"

# ── repo root ──────────────────────────────────────────────────
REPO_NAME = "llm-finetune-eval"

if IS_COLAB:
    ROOT: Path = Path("/content")
elif IS_KAGGLE:
    ROOT = Path("/kaggle/working")
else:
    # local: assume this file lives at <repo>/src/llm_ft/env.py
    ROOT = Path(__file__).resolve().parents[2]

REPO_DIR: Path = ROOT / REPO_NAME
if IS_LOCAL:
    REPO_DIR = ROOT  # already inside the repo

# ── persistent storage (Google Drive, mounted identically in Colab & Kaggle) ──
DRIVE_ROOT: Path = Path(os.environ.get("LLM_FT_DRIVE_ROOT", "/content/drive/MyDrive/llm-ft"))
CHECKPOINT_DIR: Path = DRIVE_ROOT / "checkpoints"
DATA_DIR: Path = DRIVE_ROOT / "data"
LOG_DIR: Path = DRIVE_ROOT / "logs"


def drive_available() -> bool:
    """True if the Drive mount looks live."""
    return DRIVE_ROOT.parent.exists()


def ensure_drive_dirs() -> bool:
    """Create persistent dirs if Drive is mounted. Returns True if created."""
    if not drive_available():
        return False
    for p in (CHECKPOINT_DIR, DATA_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)
    return True


def info() -> dict[str, str | bool]:
    """Return a JSON-serializable snapshot for logging."""
    return {
        "platform": PLATFORM,
        "is_colab": IS_COLAB,
        "is_kaggle": IS_KAGGLE,
        "is_local": IS_LOCAL,
        "root": str(ROOT),
        "repo_dir": str(REPO_DIR),
        "drive_root": str(DRIVE_ROOT),
        "drive_available": drive_available(),
    }
