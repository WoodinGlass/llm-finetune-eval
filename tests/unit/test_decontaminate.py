"""Unit tests for 13-gram decontamination."""

from __future__ import annotations

import pytest

from llm_ft.decontaminate import (
    _ngrams,
    build_test_index,
    decontaminate,
    find_contamination,
)

pytestmark = pytest.mark.unit

LEAK_BLOCK = (
    "the integral of x squared dx from zero to one equals one third "
    "and this is a classic example in calculus textbooks worldwide"
)


# ── _ngrams ────────────────────────────────────────────────────


def test_ngrams_short_returns_empty():
    assert _ngrams("a b c", n=5) == set()


def test_ngrams_exact_length_one():
    out = _ngrams("one two three four five", n=5)
    assert out == {"one two three four five"}


def test_ngrams_multiple():
    out = _ngrams("a b c d e f", n=3)
    assert out == {"a b c", "b c d", "c d e", "d e f"}


def test_ngrams_case_insensitive():
    assert _ngrams("Hello World Foo", n=2) == _ngrams("hello world foo", n=2)


# ── build_test_index ───────────────────────────────────────────


def test_build_index_empty_for_short_prompts():
    idx = build_test_index([{"id": "t1", "prompt": "hi there"}], n=13)
    assert idx == {}


def test_build_index_contains_long_ngrams():
    idx = build_test_index([{"id": "t1", "prompt": LEAK_BLOCK}], n=13)
    assert len(idx) > 0
    assert all(v == "t1" for v in idx.values())


# ── find_contamination ─────────────────────────────────────────


def test_contamination_detected():
    train = [{"id": "tr-1", "prompt": LEAK_BLOCK + " more words here"}]
    test = [{"id": "te-1", "prompt": LEAK_BLOCK}]
    idx = build_test_index(test, n=13)
    clean, contam = find_contamination(train, idx, n=13)
    assert len(clean) == 0
    assert len(contam) == 1
    assert contam[0]["id"] == "tr-1"
    assert contam[0]["matched_test_id"] == "te-1"


def test_contamination_not_detected_for_distinct_text():
    train = [
        {
            "id": "tr-1",
            "prompt": (
                "calculate the surface area of a sphere of radius seven meters "
                "then divide by two and report the result in square centimeters"
            ),
        }
    ]
    test = [
        {
            "id": "te-1",
            "prompt": (
                "compute the volume of a cylinder with radius three and height ten "
                "and express the answer in cubic millimeters rounded to integers"
            ),
        }
    ]
    idx = build_test_index(test, n=13)
    clean, contam = find_contamination(train, idx, n=13)
    assert len(clean) == 1
    assert len(contam) == 0


def test_contamination_short_train_safe():
    train = [{"id": "tr-1", "prompt": "What is 2 + 2"}]
    test = [{"id": "te-1", "prompt": LEAK_BLOCK}]
    idx = build_test_index(test, n=13)
    clean, contam = find_contamination(train, idx, n=13)
    assert len(clean) == 1
    assert len(contam) == 0


# ── decontaminate (end-to-end) ─────────────────────────────────


def test_decontaminate_stats():
    train = [
        {"id": "tr-1", "domain": "math", "prompt": LEAK_BLOCK + " tail words"},
        {
            "id": "tr-2",
            "domain": "math",
            "prompt": (
                "solve for x in the linear equation three x plus seven equals "
                "twenty two and provide the exact rational answer in lowest terms"
            ),
        },
    ]
    test = [{"id": "te-1", "domain": "math", "prompt": LEAK_BLOCK}]
    clean, stats = decontaminate(train, test, n=13)
    assert stats["input"] == 2
    assert stats["output"] == 1
    assert stats["removed"] == 1
    assert stats["removed_pct"] > 0
    assert clean[0]["id"] == "tr-2"


def test_decontaminate_no_test_data():
    train = [{"id": "tr-1", "domain": "math", "prompt": LEAK_BLOCK}]
    clean, stats = decontaminate(train, [], n=13)
    assert stats["output"] == 1
    assert stats["removed"] == 0


def test_decontaminate_empty_train():
    clean, stats = decontaminate([], [{"id": "t", "prompt": LEAK_BLOCK}], n=13)
    assert stats["input"] == 0
    assert stats["output"] == 0
    assert stats["removed"] == 0
