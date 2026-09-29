"""Dataset loading + chat-format construction.

Key design:
- build_messages() is pure (no tokenizer) so it is testable on CPU.
- render_chat() needs a tokenizer but does nothing else.
- load_train_records() reads JSONL files listed in config.
- build_hf_dataset() wires the two together.

Domain-specific target construction:
  math / physics       : assistant content = reasoning (if any) + "\n\n" +
                         "\\boxed{answer}"  (matches eval/baseline.py)
  ood / adversarial_ood: assistant content = REFUSAL_STRING (exact)
                         and NO system prompt
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_ft.constants import REFUSAL_STRING

# ── low-level ─────────────────────────────────────────────────


def _target_for(record: dict[str, Any]) -> str:
    """Return the assistant content for a training record."""
    domain = record["domain"]
    if domain in ("ood", "adversarial_ood"):
        return REFUSAL_STRING

    answer = str(record["answer"]).strip()
    reasoning = (record.get("reasoning") or "").strip()

    if domain == "physics":
        # match eval/baseline.py prompt expectation: boxed letter
        return f"\\boxed{{{answer}}}"

    # math
    if reasoning:
        return f"{reasoning}\n\n\\boxed{{{answer}}}"
    return f"\\boxed{{{answer}}}"


def build_messages(
    record: dict[str, Any],
    system_prompts: dict[str, str | None],
) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for one training example.

    Pure function — no tokenizer. Unit-testable on CPU.
    """
    domain = record["domain"]
    sys_prompt = system_prompts.get(domain)

    msgs: list[dict[str, str]] = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": record["prompt"].strip()})
    msgs.append({"role": "assistant", "content": _target_for(record)})
    return msgs


def render_chat(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Apply the model's chat template. Requires a real tokenizer."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


# ── mid-level ─────────────────────────────────────────────────


def load_train_records(
    train_files: dict[str, str],
    *,
    max_samples: int | None = None,
    require_all_domains: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Read JSONL files, optionally cap, optionally balance by domain."""
    import random

    rng = random.Random(seed)
    per_domain: dict[str, list[dict[str, Any]]] = {}

    for domain, path_str in train_files.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"missing train file for {domain}: {path}")
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        for r in rows:
            if r["domain"] != domain:
                raise ValueError(f"domain mismatch in {path}: {r['id']} says {r['domain']}")
        per_domain[domain] = rows

    if require_all_domains and max_samples is not None:
        # equal share per domain; take min available
        n_domains = len(per_domain)
        quota = max(1, max_samples // n_domains)
        selected = []
        for _domain, rows in per_domain.items():
            rng.shuffle(rows)
            selected.extend(rows[:quota])
        rng.shuffle(selected)
        return selected[:max_samples]

    # concatenate and (optionally) subsample
    all_rows = [r for rows in per_domain.values() for r in rows]
    rng.shuffle(all_rows)
    if max_samples is not None:
        all_rows = all_rows[:max_samples]
    return all_rows


def build_hf_dataset(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    system_prompts: dict[str, str | None],
):
    """Convert records into a HuggingFace Dataset with 'text' field.

    The 'text' is the full chat (system + user + assistant) rendered with the
    tokenizer's chat template. SFTTrainer / Trainer will tokenize this.
    """
    from datasets import Dataset

    texts = []
    for rec in records:
        msgs = build_messages(rec, system_prompts)
        texts.append(render_chat(msgs, tokenizer, add_generation_prompt=False))
    return Dataset.from_dict({"text": texts})
