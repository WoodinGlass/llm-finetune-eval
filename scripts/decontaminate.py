"""Decontaminate training data against frozen test set (13-gram overlap).

Run:  python scripts/decontaminate.py

Effect:
  - data/processed/train/*.jsonl are overwritten with the decontaminated set.
  - data/processed/decontamination_report.json records what was removed.

The training files were produced by M1.5 (dedup + PII scrub). This step is
the final filter before DVC split (M1.7).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llm_ft.decontaminate import decontaminate  # noqa: E402

TRAIN_DIR = REPO / "data" / "processed" / "train"
TEST_DIR = REPO / "data" / "processed" / "test"
REPORT_PATH = REPO / "data" / "processed" / "decontamination_report.json"

TRAIN_DOMAINS = ("math", "physics", "ood", "adversarial_ood")
TEST_FILES = ("math_test.jsonl", "physics_test.jsonl", "ood_test.jsonl", "false_refusal_test.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    print("=" * 60)
    print("DECONTAMINATE (13-gram overlap)")
    print("=" * 60)

    # 1. load all test records into one combined index
    test_records: list[dict[str, Any]] = []
    for fname in TEST_FILES:
        p = TEST_DIR / fname
        if not p.exists():
            print(f"  skip test {fname}: not found")
            continue
        rows = load_jsonl(p)
        test_records.extend(rows)
        print(f"  test {fname:<28} {len(rows)} rows")
    print(f"  combined test: {len(test_records)} rows")
    print()

    t0 = time.time()
    per_domain: dict[str, Any] = {}
    total_in = total_out = 0

    # 2. iterate over training domains
    for domain in TRAIN_DOMAINS:
        src = TRAIN_DIR / f"{domain}_train.jsonl"
        if not src.exists():
            print(f"  skip {domain}: not found")
            continue

        train_rows = load_jsonl(src)
        t1 = time.time()
        clean, stats = decontaminate(train_rows, test_records, n=13)
        dt = time.time() - t1

        write_jsonl(src, clean)

        total_in += len(train_rows)
        total_out += len(clean)
        per_domain[domain] = {
            "input": len(train_rows),
            "output": len(clean),
            "removed": stats["removed"],
            "removed_pct": stats["removed_pct"],
            "n": stats["n"],
            "wall_seconds": round(dt, 2),
            "sample": stats["contaminated_sample"],
        }
        print(
            f"  {domain:<18} {len(train_rows):>5} -> {len(clean):>5}  "
            f"(removed {stats['removed']} = {stats['removed_pct']:.2f}%)"
        )

    print()
    print(f"  {'TOTAL':<18} {total_in:>5} -> {total_out:>5}  " f"(removed {total_in - total_out})")

    report = {
        "n_gram": 13,
        "n_test_records": len(test_records),
        "per_domain": per_domain,
        "totals": {
            "input": total_in,
            "output": total_out,
            "removed": total_in - total_out,
            "removed_pct": round(100 * (total_in - total_out) / max(total_in, 1), 3),
            "wall_seconds": round(time.time() - t0, 2),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print()
    print(f"report -> {REPORT_PATH.relative_to(REPO)}")
    print(f"wall   = {report['totals']['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
