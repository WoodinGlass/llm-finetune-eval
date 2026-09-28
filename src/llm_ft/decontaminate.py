"""Train-vs-test decontamination via 13-gram overlap.

Standard method from GPT-3 / Llama papers:
  A training sample is contaminated if its prompt shares ANY 13-token
  contiguous subsequence with ANY test prompt.

Tokenization: whitespace, lowercased. We do NOT use tiktoken here because
  (a) subword tokenization has different overlap semantics than word n-grams,
  (b) it would multiply the number of n-grams we need to compare,
  (c) dataset contamination is a word-level phenomenon (quoted questions).

Only `prompt` is compared. Comparing `answer` would false-positive on
  constant answers like REFUSAL_STRING.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_N = 13


def _ngrams(text: str, n: int = DEFAULT_N) -> set[str]:
    """Whitespace-tokenized, lowercased word n-grams."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_test_index(
    test_records: Iterable[dict[str, Any]],
    n: int = DEFAULT_N,
) -> dict[str, str]:
    """Map every 13-gram of every test prompt → test record id.

    If the same n-gram appears in multiple test records, the last one wins —
    that's fine because we only need to know *that* a test sample exists.
    """
    index: dict[str, str] = {}
    for rec in test_records:
        for ng in _ngrams(rec["prompt"], n):
            index[ng] = rec["id"]
    return index


def find_contamination(
    train_records: Iterable[dict[str, Any]],
    test_index: dict[str, str],
    n: int = DEFAULT_N,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split train into (clean, contaminated).

    A train sample is contaminated if ANY of its prompt 13-grams is in the
    test index.
    """
    clean: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []

    for rec in train_records:
        ngs = _ngrams(rec["prompt"], n)
        hits = ngs.intersection(test_index.keys())
        if hits:
            matched = next(iter(hits))
            contaminated.append(
                {
                    "id": rec["id"],
                    "domain": rec.get("domain", "?"),
                    "matched_test_id": test_index[matched],
                    "matched_ngram": matched,
                    "prompt_preview": rec["prompt"][:160],
                }
            )
        else:
            clean.append(rec)

    return clean, contaminated


def decontaminate(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    n: int = DEFAULT_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full pipeline: build index, filter train, produce stats."""
    test_index = build_test_index(test_records, n=n)
    clean, contaminated = find_contamination(train_records, test_index, n=n)

    stats = {
        "n": n,
        "test_ngrams": len(test_index),
        "input": len(train_records),
        "output": len(clean),
        "removed": len(contaminated),
        "removed_pct": round(100 * len(contaminated) / max(len(train_records), 1), 3),
        "contaminated_sample": contaminated[:20],
    }
    return clean, stats


# ── self-test ──────────────────────────────────────────────────


def _self_test() -> None:
    # 13-word repeated block — this IS a leak
    leak_block = (
        "the integral of x squared dx from zero to one equals one third "
        "and this is a classic example in calculus textbooks worldwide"
    )
    train = [
        {"id": "t-leak", "domain": "math", "prompt": leak_block + " extra words"},
        {"id": "t-ok", "domain": "math", "prompt": "What is 2 plus 2"},
    ]
    test = [
        {"id": "te-1", "domain": "math", "prompt": leak_block},
    ]
    clean, stats = decontaminate(train, test, n=13)
    assert len(clean) == 1
    assert clean[0]["id"] == "t-ok"
    assert stats["removed"] == 1
    assert stats["contaminated_sample"][0]["id"] == "t-leak"
    print(f"✓ decontam leak: removed {stats['removed']}")

    # short prompts → no n-grams, never contaminated
    train2 = [{"id": "t-short", "domain": "math", "prompt": "What is 2 plus 2"}]
    test2 = [{"id": "te-short", "domain": "math", "prompt": "What is 2 plus 2"}]
    clean2, stats2 = decontaminate(train2, test2, n=13)
    assert len(clean2) == 1
    assert stats2["removed"] == 0
    print(f"✓ short prompts safe: {stats2['removed']} removed (too short for 13-gram)")

    # longer prompts, different content
    train3 = [
        {
            "id": "t-uniq",
            "domain": "math",
            "prompt": (
                "calculate the surface area of a sphere of radius seven meters "
                "then divide by two and report the result in square centimeters"
            ),
        }
    ]
    test3 = [
        {
            "id": "te-uniq",
            "domain": "math",
            "prompt": (
                "compute the volume of a cylinder with radius three and height ten "
                "and express the answer in cubic millimeters rounded to integers"
            ),
        }
    ]
    clean3, stats3 = decontaminate(train3, test3, n=13)
    assert len(clean3) == 1
    assert stats3["removed"] == 0
    print(f"✓ distinct prompts safe: {stats3['removed']} removed")


if __name__ == "__main__":
    _self_test()
