"""Baseline inference: Qwen2.5-Math-7B-Instruct on frozen test set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_ft.constants import DEFAULT_SEED, DOMAIN_MATH, DOMAIN_OOD, DOMAIN_PHYSICS  # noqa: E402
from llm_ft.prompts import build_messages  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_REVISION = "main"

MAX_NEW_TOKENS = {
    DOMAIN_PHYSICS: 1024,
    "adversarial_ood": 384,
    "false_refusal_math": 384,
    "false_refusal_physics": 384,
}

TEST_DIR = REPO_ROOT / "data" / "processed" / "test"
LOCK_PATH = REPO_ROOT / "data" / "processed" / "test.lock"
OUT_DIR = REPO_ROOT / "experiments" / "baseline"

FILE_TO_DOMAIN = {
    "math_test.jsonl": DOMAIN_MATH,
    "physics_test.jsonl": DOMAIN_PHYSICS,
    "ood_test.jsonl": DOMAIN_OOD,
    "false_refusal_test.jsonl": "false_refusal_math",
}


def git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=True
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def load_test_items(domains_filter):
    items = []
    for fname, domain in FILE_TO_DOMAIN.items():
        if (
            domains_filter
            and domain not in domains_filter
            and not any(d in fname for d in domains_filter)
        ):
            continue
        path = TEST_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"missing: {path}")
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_prompt_domain"] = domain.replace("false_refusal_", "")
            row["_source_file"] = fname
            items.append(row)
    return items


def load_model():
    print(f"→ loading {MODEL_ID} (4-bit NF4) ...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    mdl.eval()
    print(f"✓ loaded in {time.time()-t0:.1f}s (VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB)")
    return mdl, tok


@torch.inference_mode()
def generate_batch(model, tokenizer, items):
    prompts = []
    for it in items:
        msgs = build_messages(it["_prompt_domain"], it["prompt"])
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)
    mt = max(MAX_NEW_TOKENS.get(it["_prompt_domain"], 384) for it in items)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(
            **inputs,
            max_new_tokens=mt,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    gen_ids = out[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--domains", type=str, default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(DEFAULT_SEED)
    domains_filter = args.domains.split(",") if args.domains else None
    items = load_test_items(domains_filter)
    if args.limit:
        by_file = {}
        for it in items:
            by_file.setdefault(it["_source_file"], []).append(it)
        items = []
        for fname in sorted(by_file):
            items.extend(by_file[fname][: args.limit])

    print(f"→ {len(items)} items, batch={args.batch_size}")
    model, tokenizer = load_model()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "generations.jsonl"
    tmp_path = OUT_DIR / "generations.jsonl.partial"
    t_start = time.time()
    n_done = 0

    with tmp_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            t0 = time.time()
            try:
                outs = generate_batch(model, tokenizer, batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  ⚠ OOM batch {i//args.batch_size} — fallback size 1")
                outs = []
                for it in batch:
                    outs.extend(generate_batch(model, tokenizer, [it]))
            dt = time.time() - t0
            for it, out in zip(batch, outs, strict=False):
                row = {
                    "id": it["id"],
                    "domain": it["domain"],
                    "source_file": it["_source_file"],
                    "prompt_domain": it["_prompt_domain"],
                    "prompt": it["prompt"],
                    "answer": it["answer"],
                    "generation": out.strip(),
                    "meta": it.get("meta", {}),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
            el = time.time() - t_start
            rate = n_done / max(el, 1e-6)
            eta = (len(items) - n_done) / max(rate, 1e-6)
            print(
                f"  [{n_done:>3}/{len(items)}] batch {i//args.batch_size:>3}  "
                f"{dt:>5.1f}s  elapsed {el:>5.0f}s  eta {eta:>5.0f}s"
            )

    tmp_path.rename(out_path)
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": "4-bit nf4 double-quant, float16 compute",
        "decoding": "greedy",
        "batch_size": args.batch_size,
        "seed": DEFAULT_SEED,
        "git_sha": git_sha(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_lock_sha256": json.loads(LOCK_PATH.read_text())["combined_sha256"],
        "n_items": n_done,
        "wall_seconds": round(time.time() - t_start, 1),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"\n✓ generations → {out_path.relative_to(REPO_ROOT)}")
    print(f"  wall = {meta['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
