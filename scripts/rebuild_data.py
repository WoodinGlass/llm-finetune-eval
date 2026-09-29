"""Rebuild training data from HuggingFace sources (Kaggle fallback)."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

REPO = Path("/kaggle/working/llm-finetune-eval")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from datasets import load_dataset
from llm_ft.constants import REFUSAL_STRING

random.seed(42)
RAW = Path("data/raw/train")
PROC = Path("data/processed/train")
L = ["A", "B", "C", "D"]

# ══════════════════════════════════════════════════════════════
# 1. MATH
# ══════════════════════════════════════════════════════════════
print("── 1. math ──")
gsm = load_dataset("openai/gsm8k", "main", split="train")
idx = list(range(len(gsm))); random.shuffle(idx); idx = idx[:1200]
math_rows = []
for i in idx:
    ex = gsm[i]
    full = ex["answer"]
    if "####" in full:
        reasoning, final = full.rsplit("####", 1)
        reasoning = reasoning.strip(); final = final.strip()
    else:
        reasoning, final = "", full.strip()
    math_rows.append({
        "id": f"gsm8k-tr-{i:05d}", "domain": "math",
        "prompt": ex["question"].strip(), "answer": final,
        "reasoning": reasoning if reasoning else None,
        "meta": {"source": "openai/gsm8k", "split": "train", "idx": i},
    })

configs = ["algebra", "counting_and_probability", "geometry",
           "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
per_config = 600 // len(configs) + 1
for cfg in configs:
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        ids = list(range(len(ds))); random.shuffle(ids)
        picked = 0
        for i in ids:
            if picked >= per_config or len(math_rows) >= 1800:
                break
            ex = ds[i]
            sol = (ex.get("solution") or "").strip()
            m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", sol)
            if not m:
                continue
            math_rows.append({
                "id": f"math-{cfg[:12]}-{i:05d}", "domain": "math",
                "prompt": ex["problem"].strip(), "answer": m.group(1).strip(),
                "reasoning": sol if sol else None,
                "meta": {"source": "EleutherAI/hendrycks_math",
                         "config": cfg, "split": "train", "idx": i},
            })
            picked += 1
    except Exception as e:
        print(f"  skip {cfg}: {e}")

with (RAW / "math_train.jsonl").open("w") as f:
    for r in math_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  math: {len(math_rows)} rows")

# ══════════════════════════════════════════════════════════════
# 2. PHYSICS
# ══════════════════════════════════════════════════════════════
print("\n── 2. physics ──")
phys = []
for cfg, target in [("high_school_physics", 400),
                    ("conceptual_physics", 400),
                    ("astronomy", 200)]:
    items = []
    for split in ("test", "validation", "dev"):
        try:
            ds = load_dataset("cais/mmlu", cfg, split=split)
            items.extend([dict(x) for x in ds])
        except Exception:
            continue
    random.shuffle(items)
    for i, ex in enumerate(items[:target]):
        if len(ex["choices"]) < 4:
            continue
        p = ex["question"].strip() + "\n\n"
        for l, c in zip(L, ex["choices"][:4]):
            p += f"{l}) {c}\n"
        p += "\nAnswer with the letter only."
        phys.append({
            "id": f"mmlu-{cfg[:20]}-{i:05d}", "domain": "physics",
            "prompt": p.strip(), "answer": L[ex["answer"]],
            "reasoning": None,
            "meta": {"source": "cais/mmlu", "config": cfg, "split": "all"},
        })

mmlu_prompts = {r["prompt"][:200] for r in phys}
for split in ("test", "validation"):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    items = [dict(x) for x in ds if x.get("category") == "physics"]
    items = [x for x in items if (x.get("question") or "")[:200] not in mmlu_prompts]
    random.shuffle(items)
    for i, ex in enumerate(items):
        if len(phys) >= 1800:
            break
        opts = ex.get("options") or []
        al = ex.get("answer") or ""
        if al not in "ABCDEFGHIJ":
            continue
        ai = ord(al) - ord("A")
        if ai >= len(opts):
            continue
        correct = opts[ai]
        distr = [o for j, o in enumerate(opts) if j != ai]
        if len(distr) < 3:
            continue
        rng = random.Random(f"{split}-{i}")
        rng.shuffle(distr)
        d3 = distr[:3]
        pos = rng.randint(0, 3)
        new_opts = d3[:pos] + [correct] + d3[pos:]
        p = ex["question"].strip() + "\n\n"
        for l, c in zip(L, new_opts):
            p += f"{l}) {c}\n"
        p += "\nAnswer with the letter only."
        phys.append({
            "id": f"mmlupro-phys-{split[:3]}-{i:05d}", "domain": "physics",
            "prompt": p.strip(), "answer": L[pos], "reasoning": None,
            "meta": {"source": "TIGER-Lab/MMLU-Pro", "config": "physics",
                     "split": split, "orig_answer": al},
        })

random.shuffle(phys)
phys = phys[:1800]
with (RAW / "physics_train.jsonl").open("w") as f:
    for r in phys:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  physics: {len(phys)} rows")

# ══════════════════════════════════════════════════════════════
# 3. OOD
# ══════════════════════════════════════════════════════════════
print("\n── 3. ood ──")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ood_data import OOD_PROMPTS, ADVERSARIAL, TEMPLATES, SLOTS

ood = []
for cat, prompts in OOD_PROMPTS.items():
    for p in prompts:
        ood.append({
            "id": f"ood-{len(ood)+1:04d}", "domain": "ood",
            "prompt": p.strip(), "answer": REFUSAL_STRING, "reasoning": None,
            "meta": {"source": "manual", "category": cat},
        })

existing = {r["prompt"] for r in ood}
while len(ood) < 320:
    tname = random.choice(list(TEMPLATES.keys()))
    tmpl = random.choice(TEMPLATES[tname])
    placeholders = [w.strip("{}.,!?") for w in tmpl.split() if "{" in w]
    vals = {}
    ok = True
    for ph in placeholders:
        ph_k = ph.rstrip(".,!?")
        if ph_k not in SLOTS:
            ok = False
            break
        vals[ph] = random.choice(SLOTS[ph_k])
    if not ok:
        continue
    try:
        prompt = tmpl.format(**vals)
    except Exception:
        continue
    if prompt in existing:
        continue
    existing.add(prompt)
    ood.append({
        "id": f"ood-{len(ood)+1:04d}", "domain": "ood",
        "prompt": prompt, "answer": REFUSAL_STRING, "reasoning": None,
        "meta": {"source": "template", "template": tname},
    })

with (RAW / "ood_train.jsonl").open("w") as f:
    for r in ood:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  ood: {len(ood)} rows")

# ══════════════════════════════════════════════════════════════
# 4. ADVERSARIAL OOD
# ══════════════════════════════════════════════════════════════
print("\n── 4. adversarial_ood ──")
adv = [{
    "id": f"adv-ood-{i+1:04d}", "domain": "adversarial_ood",
    "prompt": p, "answer": REFUSAL_STRING, "reasoning": None,
    "meta": {"source": "manual", "category": "adversarial"},
} for i, p in enumerate(ADVERSARIAL)]

existing_adv = {r["prompt"] for r in adv}
while len(adv) < 80:
    p1 = random.choice(ADVERSARIAL)
    p2 = random.choice([x for x in ADVERSARIAL if x != p1])
    combined = f"{p1} Also, {p2.lower()}"
    if combined in existing_adv:
        continue
    existing_adv.add(combined)
    adv.append({
        "id": f"adv-ood-{len(adv)+1:04d}", "domain": "adversarial_ood",
        "prompt": combined, "answer": REFUSAL_STRING, "reasoning": None,
        "meta": {"source": "composition", "category": "mixed"},
    })

with (RAW / "adversarial_ood_train.jsonl").open("w") as f:
    for r in adv:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  adv-ood: {len(adv)} rows")

# ══════════════════════════════════════════════════════════════
# 5. copy raw → processed (skip clean for smoke)
# ══════════════════════════════════════════════════════════════
print("\n── 5. copy raw → processed ──")
for f in ["math_train.jsonl", "physics_train.jsonl",
          "ood_train.jsonl", "adversarial_ood_train.jsonl"]:
    shutil.copy2(RAW / f, PROC / f)
    print(f"  ✓ {f}")

# ══════════════════════════════════════════════════════════════
# 6. verify
# ══════════════════════════════════════════════════════════════
print("\n── verify ──")
total = 0
for f in ["math_train.jsonl", "physics_train.jsonl",
          "ood_train.jsonl", "adversarial_ood_train.jsonl"]:
    p = PROC / f
    n = sum(1 for _ in p.open()) if p.exists() else 0
    total += n
    print(f"  {f}  {n}")
print(f"  total: {total}")
