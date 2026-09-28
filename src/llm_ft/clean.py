"""Deduplication and PII scrubbing for training data.

Pipeline:
  1. dedupe_minhash: near-duplicate removal via MinHash LSH
  2. scrub_pii: regex + NER, domain-aware

Domain-aware PII policy:
  math / physics (fictional characters, numeric answers):
    - Apply CONSERVATIVE regex only: email, phone_id, phone_intl, url_with_token
    - Do NOT scrub numbers, names, locations, IPs, NIK, NPWP, CC-like —
      those would corrupt legitimate math/physics content.
    - NER is NOT used (would scrub "John" from "John has 5 apples").
  ood / adversarial_ood (real user-facing text):
    - Apply AGGRESSIVE regex: NIK, NPWP, passport, SIM, CC-like, IP v4, all URLs.
    - Apply Presidio NER: PERSON, LOCATION, GPE, NRP, DATE_TIME (score >= 0.6).

Idempotency:
  Both stages are pure; running twice on cleaned input yields the same output.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from datasketch import MinHash, MinHashLSH

# ── regex: conservative set (all domains) ───────────────────────
_R_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_R_PHONE_ID = re.compile(r"(?:\+62|62|0)8[1-9][0-9]{6,11}\b")
_R_PHONE_INTL = re.compile(r"\+[1-9]\d{1,3}[\s\-]?\d{4,14}\b")
_R_URL_TOKEN = re.compile(
    r"https?://\S*(?:token|key|secret|password|api[_-]?key)=[^\s&]+\S*",
    re.IGNORECASE,
)

# ── regex: aggressive set (ood / adversarial_ood only) ──────────
_R_NIK = re.compile(r"\b\d{16}\b")
_R_NPWP = re.compile(r"\b\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}\b|\b\d{15}\b")
_R_PASSPORT_ID = re.compile(r"\b[A-Z]\d{7}\b")
_R_CC_LIKE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
_R_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_R_URL_ANY = re.compile(r"https?://\S+")

PII_CONSERVATIVE: dict[str, re.Pattern[str]] = {
    "email": _R_EMAIL,
    "phone_id": _R_PHONE_ID,
    "phone_intl": _R_PHONE_INTL,
    "url_with_token": _R_URL_TOKEN,
}

PII_AGGRESSIVE: dict[str, re.Pattern[str]] = {
    **PII_CONSERVATIVE,
    "nik": _R_NIK,
    "npwp": _R_NPWP,
    "passport": _R_PASSPORT_ID,
    "cc_like": _R_CC_LIKE,
    "ipv4": _R_IPV4,
    "url": _R_URL_ANY,
}

# Which domains get the aggressive treatment.
_AGGRESSIVE_DOMAINS = {"ood", "adversarial_ood"}

# NER entities (Presidio)
NER_ENTITIES = [
    "PERSON",
    "LOCATION",
    "GPE",
    "NRP",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
]
NER_MIN_SCORE = 0.6
NER_MAX_TEXT_LEN = 2000  # skip NER on very long text

PLACEHOLDER = "[REDACTED]"

# ── lazy Presidio singleton ─────────────────────────────────────
_ANALYZER: Any = None
_ANALYZER_READY: bool | None = None


def _get_analyzer() -> Any:
    """Return a cached AnalyzerEngine or None if unavailable."""
    global _ANALYZER, _ANALYZER_READY
    if _ANALYZER_READY is None:
        try:
            from presidio_analyzer import AnalyzerEngine

            _ANALYZER = AnalyzerEngine()
            _ANALYZER_READY = True
        except Exception as e:
            print(f"[clean] Presidio unavailable: {type(e).__name__}: {e}")
            _ANALYZER = None
            _ANALYZER_READY = False
    return _ANALYZER


# ═══════════════════════════════════════════════════════════════
# DEDUP
# ═══════════════════════════════════════════════════════════════


def _shingles(text: str, k: int = 5) -> set[str]:
    words = text.lower().split()
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash_for(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in _shingles(text, k=5):
        m.update(s.encode("utf-8"))
    return m


def dedupe_minhash(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    key: Callable[[dict[str, Any]], str] = lambda r: r["prompt"],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove near-duplicates. Keep first-seen per cluster."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict[str, Any]] = []
    removed_ids: list[str] = []

    for i, rec in enumerate(records):
        mh = _minhash_for(key(rec), num_perm=num_perm)
        try:
            if lsh.query(mh):
                removed_ids.append(rec["id"])
                continue
        except Exception:
            pass
        lsh.insert(str(i), mh)
        kept.append(rec)

    stats = {
        "input": len(records),
        "output": len(kept),
        "removed": len(removed_ids),
        "removed_pct": round(100 * len(removed_ids) / max(len(records), 1), 3),
        "removed_ids_sample": removed_ids[:10],
        "threshold": threshold,
        "num_perm": num_perm,
    }
    return kept, stats


# ═══════════════════════════════════════════════════════════════
# PII SCRUB — regex helpers
# ═══════════════════════════════════════════════════════════════


def _regex_matches(text: str, patterns: dict[str, re.Pattern[str]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for name, pat in patterns.items():
        for m in pat.finditer(text):
            out.append((m.start(), m.end(), name))
    return out


def _ner_matches(text: str) -> list[tuple[int, int, str]]:
    analyzer = _get_analyzer()
    if analyzer is None or len(text) > NER_MAX_TEXT_LEN:
        return []
    try:
        results = analyzer.analyze(text=text, language="en", entities=NER_ENTITIES)
    except Exception:
        return []
    return [
        (r.start, r.end, f"ner_{r.entity_type.lower()}")
        for r in results
        if r.score >= NER_MIN_SCORE
    ]


def _apply_matches(
    text: str,
    matches: list[tuple[int, int, str]],
    counts: dict[str, int],
) -> str:
    """Replace matches with placeholder, avoiding overlaps. Longest wins."""
    if not matches:
        return text
    # sort by (start asc, length desc) — keep first non-overlapping
    matches = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    accepted: list[tuple[int, int, str]] = []
    for s, e, name in matches:
        if any(s < fe and e > fs for fs, fe, _ in accepted):
            continue
        accepted.append((s, e, name))
    # apply in reverse
    out = text
    for s, e, name in sorted(accepted, key=lambda x: x[0], reverse=True):
        counts[name] = counts.get(name, 0) + 1
        out = out[:s] + PLACEHOLDER + out[e:]
    return out


def scrub_text_conservative(text: str, counts: dict[str, int] | None = None) -> str:
    if not text:
        return text
    c = counts if counts is not None else {}
    return _apply_matches(text, _regex_matches(text, PII_CONSERVATIVE), c)


def scrub_text_aggressive(text: str, counts: dict[str, int] | None = None) -> str:
    if not text:
        return text
    c = counts if counts is not None else {}
    matches = _regex_matches(text, PII_AGGRESSIVE) + _ner_matches(text)
    return _apply_matches(text, matches, c)


# ═══════════════════════════════════════════════════════════════
# PII SCRUB — record-level
# ═══════════════════════════════════════════════════════════════


def _scrub_record(
    rec: dict[str, Any],
    fields: tuple[str, ...],
    counts: dict[str, int],
) -> tuple[dict[str, Any], bool]:
    domain = rec.get("domain", "")
    aggressive = domain in _AGGRESSIVE_DOMAINS
    fn = scrub_text_aggressive if aggressive else scrub_text_conservative
    new_rec = dict(rec)
    changed = False
    for f in fields:
        v = new_rec.get(f)
        if not v or not isinstance(v, str):
            continue
        cleaned = fn(v, counts)
        if cleaned != v:
            new_rec[f] = cleaned
            changed = True
    return new_rec, changed


def scrub_pii(
    records: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = ("prompt", "answer", "reasoning"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Domain-aware PII scrub. Returns (cleaned, stats)."""
    counts: dict[str, int] = {}
    per_domain_changed: dict[str, int] = {}
    out: list[dict[str, Any]] = []

    for rec in records:
        cleaned, changed = _scrub_record(rec, fields, counts)
        if changed:
            d = rec.get("domain", "?")
            per_domain_changed[d] = per_domain_changed.get(d, 0) + 1
        out.append(cleaned)

    n_changed = sum(per_domain_changed.values())
    stats = {
        "records_changed": n_changed,
        "records_changed_pct": round(100 * n_changed / max(len(records), 1), 3),
        "per_domain_changed": per_domain_changed,
        "counts": dict(sorted(counts.items())),
        "placeholder": PLACEHOLDER,
        "ner_available": _ANALYZER_READY is True,
    }
    return out, stats


# ═══════════════════════════════════════════════════════════════
# COMBINED
# ═══════════════════════════════════════════════════════════════


def clean_records(
    records: list[dict[str, Any]],
    *,
    dedup_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run dedup then domain-aware PII scrub."""
    after_dedup, dedup_stats = dedupe_minhash(records, threshold=dedup_threshold)
    after_scrub, pii_stats = scrub_pii(after_dedup)
    return after_scrub, {"dedup": dedup_stats, "pii": pii_stats}


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════


def _self_test() -> None:
    # ── dedup ──
    recs = [
        {"id": "a1", "prompt": "What is the force on a mass m under gravity?"},
        {"id": "a2", "prompt": "What is the force on a mass m under gravity?"},
        {"id": "a3", "prompt": "What is the force on a mass m under gravity"},
        {"id": "b1", "prompt": "Compute the integral of x squared dx"},
    ]
    kept, st = dedupe_minhash(recs, threshold=0.7)
    assert len(kept) == 2, f"expected 2 kept, got {len(kept)}"
    print(f"✓ dedup: kept {len(kept)}/4, removed {st['removed']}")

    # ── PII conservative (math) ──
    math_rec = {
        "id": "m1",
        "domain": "math",
        "prompt": "Find x if x + 1234567890123456 = 0",
        "answer": "1234567890123456",
        "reasoning": "x = -1234567890123456. Contact me at a@b.com",
    }
    c = scrub_text_conservative(math_rec["reasoning"])
    assert "[REDACTED]" in c
    assert "a@b.com" not in c
    # answer must NOT be touched by conservative
    assert scrub_text_conservative("1234567890123456") == "1234567890123456"
    print("✓ conservative: scrubs email, preserves 16-digit number")

    # ── PII aggressive (ood) ──
    ood_text = "Email me at john.doe@example.com, phone +62812345678, NIK 1234567890123456"
    c = scrub_text_aggressive(ood_text)
    assert "[REDACTED]" in c
    assert "john.doe@example.com" not in c
    assert "+62812345678" not in c
    assert "1234567890123456" not in c
    print("✓ aggressive: scrubs email + phone + NIK")

    # ── domain-aware ──
    recs = [
        {
            "id": "m1",
            "domain": "math",
            "prompt": "NIK 1234567890123456",
            "answer": "x",
            "reasoning": "",
        },
        {
            "id": "o1",
            "domain": "ood",
            "prompt": "NIK 1234567890123456",
            "answer": "y",
            "reasoning": "",
        },
    ]
    cleaned, stats = scrub_pii(recs)
    # math: NIK not scrubbed (conservative)
    assert "1234567890123456" in cleaned[0]["prompt"]
    # ood: NIK scrubbed
    assert "1234567890123456" not in cleaned[1]["prompt"]
    print("✓ domain-aware: math preserves, ood scrubs")

    # ── idempotency ──
    once = scrub_text_aggressive(ood_text)
    twice = scrub_text_aggressive(once)
    assert once == twice, "not idempotent"
    print("✓ idempotent")

    # ── placeholder idempotency ──
    # [REDACTED] shouldn't match anything
    assert scrub_text_aggressive(PLACEHOLDER) == PLACEHOLDER
    print("✓ placeholder stable")


if __name__ == "__main__":
    _self_test()
