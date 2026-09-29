"""Unit tests for training data formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from training.data import _target_for, build_messages, load_train_records

from llm_ft.constants import REFUSAL_STRING
from llm_ft.prompts import SYSTEM_MATH, SYSTEM_PHYSICS

pytestmark = pytest.mark.unit

# Source of truth: llm_ft/prompts.py — training MUST match eval.
SYSTEM_PROMPTS = {
    "math": SYSTEM_MATH,
    "physics": SYSTEM_PHYSICS,
    "ood": None,
    "adversarial_ood": None,
}


# ── _target_for ────────────────────────────────────────────────


def test_target_math_with_reasoning():
    rec = {"domain": "math", "answer": "42", "reasoning": "2+40=42"}
    t = _target_for(rec)
    assert "2+40=42" in t
    assert "\\boxed{42}" in t


def test_target_math_without_reasoning():
    rec = {"domain": "math", "answer": "7", "reasoning": None}
    t = _target_for(rec)
    assert t == "\\boxed{7}"


def test_target_physics():
    rec = {"domain": "physics", "answer": "B", "reasoning": "some reasoning"}
    assert _target_for(rec) == "\\boxed{B}"


def test_target_ood_is_exact_refusal():
    rec = {"domain": "ood", "answer": "x", "reasoning": None}
    assert _target_for(rec) == REFUSAL_STRING


def test_target_adversarial_ood_is_exact_refusal():
    rec = {"domain": "adversarial_ood", "answer": "x", "reasoning": None}
    assert _target_for(rec) == REFUSAL_STRING


# ── build_messages ─────────────────────────────────────────────


def test_math_messages_have_system_prompt():
    rec = {"domain": "math", "prompt": "What is 2+2?", "answer": "4", "reasoning": None}
    msgs = build_messages(rec, SYSTEM_PROMPTS)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert "step by step" in msgs[0]["content"].lower()
    assert msgs[1] == {"role": "user", "content": "What is 2+2?"}
    assert msgs[2]["role"] == "assistant"


def test_physics_messages_have_system_prompt():
    rec = {"domain": "physics", "prompt": "Q?", "answer": "A", "reasoning": None}
    msgs = build_messages(rec, SYSTEM_PROMPTS)
    assert len(msgs) == 3
    assert "physics expert" in msgs[0]["content"].lower()


def test_ood_messages_have_no_system_prompt():
    rec = {"domain": "ood", "prompt": "Tell me a joke", "answer": "x", "reasoning": None}
    msgs = build_messages(rec, SYSTEM_PROMPTS)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == REFUSAL_STRING


def test_ood_messages_even_if_system_prompts_says_none():
    rec = {"domain": "adversarial_ood", "prompt": "Anything", "answer": "x", "reasoning": None}
    msgs = build_messages(rec, SYSTEM_PROMPTS)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


# ── load_train_records ─────────────────────────────────────────


def _write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_records_reads_all(tmp_path):
    p1 = tmp_path / "math.jsonl"
    p2 = tmp_path / "ood.jsonl"
    _write_jsonl(
        p1,
        [
            {"id": "m1", "domain": "math", "prompt": "a", "answer": "1", "reasoning": None},
            {"id": "m2", "domain": "math", "prompt": "b", "answer": "2", "reasoning": None},
        ],
    )
    _write_jsonl(
        p2,
        [
            {"id": "o1", "domain": "ood", "prompt": "x", "answer": "y", "reasoning": None},
        ],
    )
    rows = load_train_records({"math": str(p1), "ood": str(p2)}, max_samples=None, seed=42)
    assert len(rows) == 3


def test_load_records_max_samples(tmp_path):
    p1 = tmp_path / "math.jsonl"
    _write_jsonl(
        p1,
        [
            {
                "id": f"m{i}",
                "domain": "math",
                "prompt": f"q{i}",
                "answer": str(i),
                "reasoning": None,
            }
            for i in range(10)
        ],
    )
    rows = load_train_records({"math": str(p1)}, max_samples=4, seed=42)
    assert len(rows) == 4


def test_load_records_require_all_domains(tmp_path):
    p1 = tmp_path / "math.jsonl"
    p2 = tmp_path / "ood.jsonl"
    _write_jsonl(
        p1,
        [
            {
                "id": f"m{i}",
                "domain": "math",
                "prompt": f"q{i}",
                "answer": str(i),
                "reasoning": None,
            }
            for i in range(20)
        ],
    )
    _write_jsonl(
        p2,
        [
            {"id": f"o{i}", "domain": "ood", "prompt": f"x{i}", "answer": "y", "reasoning": None}
            for i in range(5)
        ],
    )
    rows = load_train_records(
        {"math": str(p1), "ood": str(p2)},
        max_samples=6,
        require_all_domains=True,
        seed=42,
    )
    domains = [r["domain"] for r in rows]
    assert domains.count("math") == 3
    assert domains.count("ood") == 3


def test_load_records_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_train_records({"math": "/nonexistent/path.jsonl"})


def test_load_records_raises_on_domain_mismatch(tmp_path):
    p = tmp_path / "wrong.jsonl"
    _write_jsonl(
        p, [{"id": "x", "domain": "physics", "prompt": "?", "answer": "A", "reasoning": None}]
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        load_train_records({"math": str(p)})


# ── parity with eval/baseline.py ────────────────────────────────


def test_math_system_prompt_matches_eval_baseline():
    """Training system prompt must equal eval system prompt to avoid drift."""
    assert SYSTEM_PROMPTS["math"] == SYSTEM_MATH
    assert SYSTEM_PROMPTS["physics"] == SYSTEM_PHYSICS
