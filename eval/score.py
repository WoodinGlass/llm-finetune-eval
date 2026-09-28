"""Score baseline generations + bootstrap CI.

Reads:
  experiments/baseline/generations.jsonl
  experiments/baseline/run_meta.json

Writes:
  experiments/baseline/report.json
  experiments/baseline/report.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_ft.constants import CI_ALPHA, DEFAULT_SEED, N_BOOTSTRAP, REFUSAL_STRING  # noqa: E402
from llm_ft.parse import is_correct  # noqa: E402

GEN_PATH = REPO_ROOT / "experiments" / "baseline" / "generations.jsonl"
META_PATH = REPO_ROOT / "experiments" / "baseline" / "run_meta.json"
REPORT_JSON = REPO_ROOT / "experiments" / "baseline" / "report.json"
REPORT_MD = REPO_ROOT / "experiments" / "baseline" / "report.md"


def bootstrap_ci(successes, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA, seed=DEFAULT_SEED):
    arr = np.asarray(successes, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = arr[idx].mean(axis=1)
    return mean, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def load_generations():
    rows = []
    with GEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(row):
    out = dict(row)
    src = row["source_file"]
    gen = row["generation"]
    dom = row["prompt_domain"]

    if src == "ood_test.jsonl":
        out["kind"] = "ood"
        out["correct"] = int(gen.strip() == REFUSAL_STRING)
    elif src == "false_refusal_test.jsonl":
        out["kind"] = "false_refusal"
        out["refused"] = int(gen.strip() == REFUSAL_STRING)
        out["correct"] = int(is_correct(gen, row["answer"], dom))
    else:
        out["kind"] = dom  # 'math' or 'physics'
        out["correct"] = int(is_correct(gen, row["answer"], dom))
    return out


def summarize(scored):
    by_kind = defaultdict(list)
    for r in scored:
        by_kind[r["kind"]].append(r)

    report = {"n_total": len(scored), "kinds": {}}
    for kind, rows in sorted(by_kind.items()):
        succ = [r["correct"] for r in rows]
        m, lo, hi = bootstrap_ci(succ)
        entry = {
            "n": len(rows),
            "correct": int(sum(succ)),
            "accuracy": m,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
        if kind == "false_refusal":
            ref = [r.get("refused", 0) for r in rows]
            rm, rl, rh = bootstrap_ci(ref)
            entry["refusal_rate"] = rm
            entry["refusal_rate_ci95_lo"] = rl
            entry["refusal_rate_ci95_hi"] = rh
        report["kinds"][kind] = entry

    domain_rows = [r for r in scored if r["kind"] in ("math", "physics", "false_refusal")]
    if domain_rows:
        m, lo, hi = bootstrap_ci([r["correct"] for r in domain_rows])
        report["domain_accuracy_combined"] = {
            "n": len(domain_rows),
            "accuracy": m,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
    return report


def render_md(report, meta):
    lines = []
    lines.append("# Baseline Report — Qwen2.5-Math-7B-Instruct (M0.5)\n")
    lines.append(f"- Model: `{meta.get('model_id','?')}`")
    lines.append(f"- Quantization: {meta.get('quantization','?')}")
    lines.append(f"- Decoding: {meta.get('decoding','?')}")
    lines.append(f"- Batch size: {meta.get('batch_size','?')}")
    lines.append(f"- Items: {meta.get('n_items','?')}")
    lines.append(f"- Wall: {meta.get('wall_seconds','?')}s")
    lines.append(f"- Test lock SHA256: `{meta.get('test_lock_sha256','?')}`")
    lines.append(f"- Git SHA: `{meta.get('git_sha','?')}`")
    lines.append("")
    lines.append("## Results (95% bootstrap CI, 1000 resamples)\n")
    lines.append("| Kind | N | Correct | Accuracy | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for kind, e in sorted(report["kinds"].items()):
        ci = f"[{e['ci95_lo']:.3f}, {e['ci95_hi']:.3f}]"
        lines.append(f"| {kind} | {e['n']} | {e['correct']} | {e['accuracy']:.3f} | {ci} |")
    if "domain_accuracy_combined" in report:
        e = report["domain_accuracy_combined"]
        ci = f"[{e['ci95_lo']:.3f}, {e['ci95_hi']:.3f}]"
        lines.append(f"| **combined** | {e['n']} | — | {e['accuracy']:.3f} | {ci} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    rows = load_generations()
    print(f"→ {len(rows)} generations loaded")

    scored = [score_row(r) for r in rows]
    report = summarize(scored)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REPORT_MD.write_text(render_md(report, meta))

    print()
    print("═" * 64)
    print("BASELINE SCORES — Qwen2.5-Math-7B-Instruct")
    print("═" * 64)
    for kind, e in sorted(report["kinds"].items()):
        print(
            f"  {kind:<20} {e['correct']:>3}/{e['n']:<3}  "
            f"acc={e['accuracy']:.3f}  CI=[{e['ci95_lo']:.3f}, {e['ci95_hi']:.3f}]"
        )
        if "refusal_rate" in e:
            print(
                f"    ↳ refusal_rate = {e['refusal_rate']:.3f} "
                f"CI=[{e['refusal_rate_ci95_lo']:.3f}, {e['refusal_rate_ci95_hi']:.3f}]"
            )
    if "domain_accuracy_combined" in report:
        e = report["domain_accuracy_combined"]
        print(
            f"  {'combined':<20} {'':>5}  "
            f"acc={e['accuracy']:.3f}  CI=[{e['ci95_lo']:.3f}, {e['ci95_hi']:.3f}]"
        )
    print()
    print(f"✓ {REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"✓ {REPORT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
