"""Freeze test set with SHA256 verification.

Usage:
    python -m llm_ft.freeze freeze    # freeze data/raw/* → data/processed/
    python -m llm_ft.freeze verify    # verify data/processed/ against test.lock
    python -m llm_ft.freeze show      # print current lock (human-readable)

Design:
- `data/raw/*_test.jsonl` are the canonical inputs (downloaded once).
- Freezing copies them into `data/processed/test/` and writes a
  `data/processed/test.lock` JSON with:
    - per-file sha256 + row count + byte size
    - combined_sha256 = hash over sorted (filename, sha256) pairs
    - git SHA at freeze time (for provenance)
- Any change to the raw files invalidates the lock; verify fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEST_DIR = PROCESSED_DIR / "test"
LOCK_PATH = PROCESSED_DIR / "test.lock"

# canonical filenames in data/raw that participate in the test set
TEST_FILES = (
    "math_test.jsonl",
    "physics_test.jsonl",
    "ood_test.jsonl",
    "false_refusal_test.jsonl",
)

LOCK_VERSION = 1


# ── helpers ────────────────────────────────────────────────────


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def combined_hash(file_hashes: dict[str, str]) -> str:
    """Deterministic hash across all files: sort by name, concat
    "name:sha256\n", hash the result."""
    h = hashlib.sha256()
    for name in sorted(file_hashes):
        h.update(f"{name}:{file_hashes[name]}\n".encode())
    return h.hexdigest()


def git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


# ── commands ───────────────────────────────────────────────────


def cmd_freeze() -> int:
    """Copy data/raw/*_test.jsonl → data/processed/test/, write test.lock."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}
    file_meta: dict[str, dict] = {}

    print("freezing test set:")
    for name in TEST_FILES:
        src = RAW_DIR / name
        if not src.exists():
            print(f"  ✗ missing: {src}", file=sys.stderr)
            return 1

        dst = TEST_DIR / name
        data = src.read_bytes()
        dst.write_bytes(data)

        digest = sha256_file(dst)
        rows = count_lines(dst)
        file_hashes[name] = digest
        file_meta[name] = {"sha256": digest, "rows": rows, "bytes": len(data)}
        print(f"  ✓ {name:<28} {rows:>4} rows  sha256={digest[:12]}…")

    lock = {
        "version": LOCK_VERSION,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "files": file_meta,
        "combined_sha256": combined_hash(file_hashes),
        "total_rows": sum(m["rows"] for m in file_meta.values()),
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    print(f"\n✓ lock written: {LOCK_PATH.relative_to(REPO_ROOT)}")
    print(f"  combined_sha256 = {lock['combined_sha256']}")
    print(f"  total_rows      = {lock['total_rows']}")
    return 0


def cmd_verify() -> int:
    """Verify data/processed/test/ against test.lock. Exit 1 on mismatch."""
    if not LOCK_PATH.exists():
        print(f"✗ no lock file at {LOCK_PATH}", file=sys.stderr)
        return 1

    lock = json.loads(LOCK_PATH.read_text())
    expected = lock["files"]
    file_hashes: dict[str, str] = {}
    ok = True

    for name, meta in expected.items():
        p = TEST_DIR / name
        if not p.exists():
            print(f"  ✗ missing: {name}", file=sys.stderr)
            ok = False
            continue
        digest = sha256_file(p)
        if digest != meta["sha256"]:
            print(f"  ✗ mismatch: {name}", file=sys.stderr)
            print(f"      expected {meta['sha256']}", file=sys.stderr)
            print(f"      got      {digest}", file=sys.stderr)
            ok = False
        else:
            print(f"  ✓ {name}")
        file_hashes[name] = digest

    if ok:
        combined = combined_hash(file_hashes)
        if combined != lock["combined_sha256"]:
            print("✗ combined hash mismatch", file=sys.stderr)
            print(f"    expected {lock['combined_sha256']}", file=sys.stderr)
            print(f"    got      {combined}", file=sys.stderr)
            return 1
        print(f"\n✓ all files match lock (combined_sha256={combined[:16]}…)")
        return 0

    print("\n✗ verify FAILED", file=sys.stderr)
    return 1


def cmd_show() -> int:
    if not LOCK_PATH.exists():
        print(f"✗ no lock file at {LOCK_PATH}", file=sys.stderr)
        return 1
    lock = json.loads(LOCK_PATH.read_text())
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    cmd = args[0] if args else "show"

    if cmd == "freeze":
        return cmd_freeze()
    if cmd == "verify":
        return cmd_verify()
    if cmd == "show":
        return cmd_show()

    print(f"unknown command: {cmd}", file=sys.stderr)
    print("usage: python -m llm_ft.freeze <freeze|verify|show>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
