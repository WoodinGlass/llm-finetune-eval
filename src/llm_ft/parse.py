"""Answer extraction from model generations.

Fallback chain (first match wins):
  1. \\boxed{...}  — canonical for Qwen2.5-Math and our physics prompt
  2. "the answer is X" / "final answer: X" pattern
  3. last standalone number in the text (math only)
  4. last single A/B/C/D letter (physics only)
  5. fallback: stripped last 120 chars

Normalization for comparison:
  - strip whitespace, $, %, commas used as thousands separators
  - normalize fractions like "1 / 2" → "1/2"
  - lowercase for text answers; keep uppercase letter for physics

Physics answers are always single uppercase A/B/C/D.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\s*\}")
_ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*(.+?)(?:[.\n]|$)", re.I),
    re.compile(r"final\s+answer\s*[:\-]\s*(.+?)(?:[.\n]|$)", re.I),
]
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_MATH_FALLBACK_CHARS = 120


def extract_boxed(text: str) -> str | None:
    """Extract content of the LAST \\boxed{...} (handles nested braces)."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    # strip LaTeX wrappers like \text{B} or \mathrm{42}
    raw = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", raw)
    return str(raw.strip())


def extract_answer(text: str, domain: str) -> str:
    """Best-effort extraction with domain-aware fallbacks."""
    text = text.strip()
    if not text:
        return ""

    # 1. boxed
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # 2. "the answer is X" style
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return str(m.group(1)).strip()

    # 3. physics: last standalone A/B/C/D
    if domain == "physics":
        letters = _LETTER_RE.findall(text)
        if letters:
            return str(letters[-1]).upper()

    # 4. math: last number
    if domain == "math":
        nums = _NUMBER_RE.findall(text)
        if nums:
            return str(nums[-1])

    # 5. fallback
    return text[-_MATH_FALLBACK_CHARS:].strip()


def normalize_math(answer: str) -> str:
    """Normalize for numeric/symbolic comparison."""
    if not answer:
        return ""
    s = answer.strip()
    # strip $ and % and \( \) and \[ \]
    s = s.replace("$", "").replace("%", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    # remove thousands separators: 1,234 → 1234  (but keep 1,5 decimal? assume US)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # remove whitespace
    s = re.sub(r"\s+", "", s)
    # normalize fractions "1/2" (already ok)
    # strip trailing period
    s = s.rstrip(".")
    return str(s).lower()


def normalize_physics(answer: str) -> str:
    """Reduce to a single uppercase A/B/C/D or empty string."""
    if not answer:
        return ""
    m = _LETTER_RE.search(answer.upper())
    if m:
        return m.group(1)
    # try first character if it's A-D
    a = answer.strip().upper()
    if a and a[0] in "ABCD":
        return a[0]
    return ""


def normalize(answer: str, domain: str) -> str:
    if domain == "physics":
        return normalize_physics(answer)
    return normalize_math(answer)


def is_correct(generation: str, expected: str, domain: str) -> bool:
    """Exact-match correctness.

    Physics is scored conservatively: correct ONLY if the model emitted
    \\boxed{X} with X in {A,B,C,D}. Free-form letters inside reasoning are
    ignored (they cause false positives: "A is wrong, so it must be B").
    Math and other domains use the general extraction chain.
    """
    if domain == "physics":
        boxed = extract_boxed(generation)
        if not boxed:
            return False
        pred = normalize_physics(boxed)
        gold = normalize_physics(expected)
        return bool(pred) and pred == gold

    pred_raw = extract_answer(generation, domain)
    pred = normalize(pred_raw, domain)
    gold = normalize(expected, domain)
    if not pred or not gold:
        return False
    return pred == gold


# ── self-test ──────────────────────────────────────────────────


def _self_test() -> None:
    cases = [
        # (generation, expected, domain, want_match)
        ("Let me think... \\boxed{42}", "42", "math", True),
        ("The answer is 42.", "42", "math", True),
        ("We compute 3+4=7. \\boxed{7}", "7", "math", True),
        ("\\boxed{\\text{36}}", "36", "math", True),
        ("$\\boxed{1/2}$", "1/2", "math", True),
        ("so the result is 1,234", "1234", "math", True),
        ("blah blah no answer", "42", "math", False),
        # physics
        ("Reasoning... \\boxed{B}", "B", "physics", True),
        ("The correct choice is C.", "C", "physics", True),
        ("\\boxed{\\text{D}}", "D", "physics", True),
        ("I think A is wrong, so it must be B", "B", "physics", True),
        ("random text with no letters", "A", "physics", False),
    ]
    passed = 0
    for gen, exp, dom, want in cases:
        got = is_correct(gen, exp, dom)
        mark = "✓" if got == want else "✗"
        if got == want:
            passed += 1
        else:
            print(
                f"  {mark} FAIL: dom={dom} exp={exp!r} gen={gen!r} "
                f"→ pred={extract_answer(gen, dom)!r} norm={normalize(extract_answer(gen, dom), dom)!r}"
            )
    print(f"self-test: {passed}/{len(cases)} passed")


if __name__ == "__main__":
    _self_test()
