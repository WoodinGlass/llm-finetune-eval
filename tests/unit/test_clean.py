"""Unit tests for dedup + PII scrub."""

from __future__ import annotations

import pytest

from llm_ft.clean import (
    PLACEHOLDER,
    clean_records,
    dedupe_minhash,
    scrub_pii,
    scrub_text_aggressive,
    scrub_text_conservative,
)

pytestmark = pytest.mark.unit


# ── helpers ────────────────────────────────────────────────────
def _rec(i: int, prompt: str) -> dict:
    return {
        "id": f"r-{i:03d}",
        "domain": "math",
        "prompt": prompt,
        "answer": "x",
        "reasoning": None,
        "meta": {"source": "test"},
    }


# ═══════════════════════════════════════════════════════════════
# Dedup
# ═══════════════════════════════════════════════════════════════


def test_dedup_identical_keeps_one():
    recs = [
        _rec(1, "the quick brown fox jumps over the lazy dog"),
        _rec(2, "the quick brown fox jumps over the lazy dog"),
    ]
    kept, st = dedupe_minhash(recs, threshold=0.85)
    assert len(kept) == 1
    assert st["removed"] == 1


def test_dedup_near_duplicate_removed():
    """Two prompts that share >0.85 Jaccard are deduped.

    Note: 5-word shingles means very short prompts (<=6 words) cannot reach
    0.85 with a single word different. Real training prompts are longer;
    this test uses a realistic length.
    """
    base = "Find the derivative of the function f of x equals x cubed"
    recs = [_rec(1, base), _rec(2, base + " please")]
    kept, st = dedupe_minhash(recs, threshold=0.85)
    assert len(kept) == 1, f"expected 1, got {len(kept)} (removed={st['removed']})"


def test_dedup_short_prompts_need_lower_threshold():
    """Very short prompts need a lower threshold to be considered dups."""
    recs = [
        _rec(1, "Compute the area of a circle with radius 5"),
        _rec(2, "Compute the area of a circle with radius 5."),
    ]
    # at 0.85 not removed (too short)
    kept_hi, _ = dedupe_minhash(recs, threshold=0.85)
    assert len(kept_hi) == 2
    # at 0.6 removed
    kept_lo, _ = dedupe_minhash(recs, threshold=0.6)
    assert len(kept_lo) == 1


def test_dedup_distinct_kept():
    recs = [
        _rec(1, "Compute the area of a circle with radius 5"),
        _rec(2, "Solve the quadratic equation x squared minus 4 equals zero"),
    ]
    kept, _ = dedupe_minhash(recs, threshold=0.85)
    assert len(kept) == 2


def test_dedup_stats_shape():
    recs = [_rec(1, "unique prompt number one"), _rec(2, "unique prompt number one")]
    _, st = dedupe_minhash(recs)
    for k in ("input", "output", "removed", "removed_pct", "threshold"):
        assert k in st


# ═══════════════════════════════════════════════════════════════
# Conservative scrub
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text",
    [
        "Contact me at john@example.com",
        "Call +62812345678 anytime",
        "Call 0812345678 anytime",
        "API token: https://x.com/a?token=secret123",
    ],
)
def test_conservative_scrubs_pii(text):
    out = scrub_text_conservative(text)
    assert PLACEHOLDER in out


@pytest.mark.parametrize(
    "text",
    [
        "x = 1234567890123456",
        "Answer: 3.14159265358979",
        "The value is 1000000000000",
        "John has 5 apples and 3 oranges",
        "Radius = 4, height = 7, volume = ?",
    ],
)
def test_conservative_preserves_legit_text(text):
    assert scrub_text_conservative(text) == text


# ═══════════════════════════════════════════════════════════════
# Aggressive scrub
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text,needle",
    [
        ("Email me at john@example.com", "john@example.com"),
        ("Phone +62812345678", "+62812345678"),
        ("NIK 1234567890123456", "1234567890123456"),
        ("IP 192.168.1.1", "192.168.1.1"),
        ("Visit https://example.com/page", "https://example.com/page"),
    ],
)
def test_aggressive_scrubs_pii(text, needle):
    out = scrub_text_aggressive(text)
    assert needle not in out
    assert PLACEHOLDER in out


def test_aggressive_scrubs_nik_in_ood():
    recs = [
        {
            "id": "o1",
            "domain": "ood",
            "prompt": "my NIK is 1234567890123456",
            "answer": "x",
            "reasoning": None,
        }
    ]
    cleaned, _ = scrub_pii(recs)
    assert "1234567890123456" not in cleaned[0]["prompt"]


def test_aggressive_preserves_placeholder():
    assert scrub_text_aggressive(PLACEHOLDER) == PLACEHOLDER


# ═══════════════════════════════════════════════════════════════
# Domain-aware
# ═══════════════════════════════════════════════════════════════


def test_domain_aware_math_preserves_16digit():
    recs = [
        {
            "id": "m1",
            "domain": "math",
            "prompt": "x = 1234567890123456",
            "answer": "1234567890123456",
            "reasoning": None,
        }
    ]
    cleaned, _ = scrub_pii(recs)
    assert "1234567890123456" in cleaned[0]["answer"]


def test_domain_aware_ood_scrubs_16digit():
    recs = [
        {
            "id": "o1",
            "domain": "ood",
            "prompt": "x = 1234567890123456",
            "answer": "y",
            "reasoning": None,
        }
    ]
    cleaned, _ = scrub_pii(recs)
    assert "1234567890123456" not in cleaned[0]["prompt"]


def test_domain_aware_per_domain_stats():
    recs = [
        {"id": "m1", "domain": "math", "prompt": "a@b.com", "answer": "1", "reasoning": None},
        {"id": "o1", "domain": "ood", "prompt": "a@b.com", "answer": "x", "reasoning": None},
    ]
    _, stats = scrub_pii(recs)
    assert stats["per_domain_changed"].get("math", 0) == 1
    assert stats["per_domain_changed"].get("ood", 0) == 1


# ═══════════════════════════════════════════════════════════════
# Idempotency
# ═══════════════════════════════════════════════════════════════


def test_scrub_idempotent_conservative():
    text = "Email a@b.com or call +62812345678"
    once = scrub_text_conservative(text)
    twice = scrub_text_conservative(once)
    assert once == twice


def test_scrub_idempotent_aggressive():
    text = "NIK 1234567890123456, email a@b.com, IP 1.2.3.4"
    once = scrub_text_aggressive(text)
    twice = scrub_text_aggressive(once)
    assert once == twice


# ═══════════════════════════════════════════════════════════════
# Combined
# ═══════════════════════════════════════════════════════════════


def test_clean_records_runs_both_stages():
    recs = [
        {
            "id": "m1",
            "domain": "math",
            "prompt": "Solve x + 1 = 2, email a@b.com",
            "answer": "1",
            "reasoning": None,
            "meta": {"source": "test"},
        },
        {
            "id": "m2",
            "domain": "math",
            "prompt": "Solve x + 1 = 2, email a@b.com",
            "answer": "1",
            "reasoning": None,
            "meta": {"source": "test"},
        },
    ]
    cleaned, stats = clean_records(recs)
    assert len(cleaned) == 1
    assert "a@b.com" not in cleaned[0]["prompt"]
    assert stats["dedup"]["removed"] == 1
    assert stats["pii"]["records_changed"] == 1
