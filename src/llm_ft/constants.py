"""Project-wide constants.

Only stdlib imports here. Every constant that appears in more than one
milestone lives here — changing it is a deliberate, version-bumping act.
"""

from __future__ import annotations

# ── refusal behavior (ADR 0005) ────────────────────────────────
# EXACT string the model must output when refusing an out-of-domain request.
# Do NOT change without bumping the model minor version and re-running the
# OOD refusal baseline.
REFUSAL_STRING: str = "I'm configured to help with mathematics and physics only."

# ── domain labels ──────────────────────────────────────────────
DOMAIN_MATH: str = "math"
DOMAIN_PHYSICS: str = "physics"
DOMAIN_OOD: str = "ood"
DOMAIN_ADVERSARIAL_OOD: str = "adversarial_ood"

VALID_DOMAINS: tuple[str, ...] = (
    DOMAIN_MATH,
    DOMAIN_PHYSICS,
    DOMAIN_OOD,
    DOMAIN_ADVERSARIAL_OOD,
)

# ── dataset field names (stable contract for M1 schema) ────────
FIELD_ID: str = "id"
FIELD_DOMAIN: str = "domain"
FIELD_PROMPT: str = "prompt"
FIELD_ANSWER: str = "answer"
FIELD_META: str = "meta"

REQUIRED_FIELDS: tuple[str, ...] = (
    FIELD_ID,
    FIELD_DOMAIN,
    FIELD_PROMPT,
    FIELD_ANSWER,
)

# ── test-set size targets (used to sanity-check splits) ────────
TEST_SIZE_MATH: int = 100
TEST_SIZE_PHYSICS: int = 100
TEST_SIZE_OOD: int = 50
TEST_SIZE_FALSE_REFUSAL: int = 50

# ── reproducibility ────────────────────────────────────────────
DEFAULT_SEED: int = 42
N_BOOTSTRAP: int = 1000
CI_ALPHA: float = 0.05  # 95% CI
