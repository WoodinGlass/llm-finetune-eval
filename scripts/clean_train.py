"""Clean the raw training data: dedup + domain-aware PII scrub.

Run:  python scripts/clean_train.py
Output: data/processed/train/{domain}_train.jsonl
        data/processed/train/clean_report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llm_ft.clean import clean_records  # noqa: E402

RAW_DIR = REPO / "data" / "raw" / "train"
PROCESSED_DIR = REPO / "data" / "processed" / "train"
DOMAINS = ["math", "physics", "ood", "adversarial_ood"]


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = {"domains": {}, "totals": {}}

    print("=" * 60)
    print("CLEAN TRAINING DATA")
    print("=" * 60)

    t0 = time.time()
    total_in = total_out = 0

    for domain in DOMAINS:
        src = RAW_DIR / f"{domain}_train.jsonl"
        if not src.exists():
            print(f"  skip {domain}: {src} not found")
            continue
        rows = load(src)
        t1 = time.time()
        cleaned, stats = clean_records(rows, dedup_threshold=0.85)
        dt = time.time() - t1
        dst = PROCESSED_DIR / f"{domain}_train.jsonl"
        write(dst, cleaned)

        total_in += len(rows)
        total_out += len(cleaned)
        report["domains"][domain] = {
            "input": len(rows),
            "output": len(cleaned),
            "dedup_removed": stats["dedup"]["removed"],
            "dedup_pct": stats["dedup"]["removed_pct"],
            "pii_records_changed": stats["pii"]["records_changed"],
            "pii_counts": stats["pii"]["counts"],
            "ner_available": stats["pii"]["ner_available"],
            "wall_seconds": round(dt, 1),
        }
        print(
            f"  {domain:<18} {len(rows):>5} -> {len(cleaned):>5}  "
            f"(dedup -{stats['dedup']['removed']} "
            f"{stats['dedup']['removed_pct']:.2f}%, "
            f"pii {stats['pii']['records_changed']} recs)"
        )

    report["totals"] = {
        "input": total_in,
        "output": total_out,
        "removed": total_in - total_out,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print()
    print(f"  {'TOTAL':<18} {total_in:>5} -> {total_out:>5}  " f"(removed {total_in - total_out})")

    rp = PROCESSED_DIR / "clean_report.json"
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print()
    print(f"report -> {rp.relative_to(REPO)}")
    print(f"wall   = {report['totals']['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
