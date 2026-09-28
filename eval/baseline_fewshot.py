"""Baseline few-shot evaluation using lm-evaluation-harness.

Runs base Qwen2.5-7B-Instruct (no fine-tune) on bilingual math+physics tasks.
Results are written to eval/results/baseline.json and printed as a table.

Usage:
    python eval/baseline_fewshot.py
    python eval/baseline_fewshot.py --tasks gsm8k,mmlu_high_school_physics
    python eval/baseline_fewshot.py --limit 100   # fast smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("eval/results")
DEFAULT_TASKS = [
    # English
    "gsm8k",
    "minerva_math_algebra",
    "minerva_math_number_theory",
    "mmlu_high_school_physics",
    "mmlu_college_physics",
    # Indonesian (custom, defined in eval/tasks/)
    "gsm8k_id",
    "minerva_math_id",
    "indommlu_fisika",
]


def build_command(tasks: list[str], limit: int | None, output: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", "pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,trust_remote_code=False",
        "--tasks", ",".join(tasks),
        "--num_fewshot", "5",
        "--batch_size", "1",
        "--device", "cuda:0",
        "--output_path", str(output),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    # Custom ID tasks live in eval/tasks/
    include = Path("eval/tasks")
    if include.exists():
        cmd += ["--include_path", str(include)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap per-task samples (for smoke runs)")
    parser.add_argument("--out", default=str(RESULTS_DIR / "baseline.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = Path(args.out)

    cmd = build_command(tasks, args.limit, out)
    print("▶ running:", " ".join(cmd))

    env = os.environ.copy()
    env.setdefault("HF_HOME", "./.cache/huggingface")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print("❌ lm-eval failed", file=sys.stderr)
        return proc.returncode

    # Summarize
    latest = sorted(RESULTS_DIR.glob("**/results_*.json"))[-1]
    data = json.loads(latest.read_text())
    print("\n📊 Baseline results")
    print(f"{'task':<40} {'metric':<20} {'value':>8}")
    print("-" * 70)
    for task, res in data.get("results", {}).items():
        for k, v in res.items():
            if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                print(f"{task:<40} {k:<20} {v:>8.4f}")
    print(f"\n✅ saved: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
