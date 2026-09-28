"""Pandera schema for all dataset records (train + eval + test).

Single source of truth for the shape of a record. Any code that reads a
dataset file MUST validate it against this schema. Data validation failures
are hard failures (see docs/slo.md §4).

Record shape:
    id        : str  — stable unique identifier, filesystem-safe
    domain    : str  — one of constants.VALID_DOMAINS
    prompt    : str  — user-facing question / instruction
    answer    : str  — reference answer (numeric, symbolic, or A/B/C/D)
    reasoning : str|None — optional chain-of-thought
    meta      : dict — per-source metadata (source, split, idx, ...)

Domain-specific rules (cross-field, checked after Pandera passes):
    math            : answer must not be empty; digits or symbolic only
    physics         : answer must be exactly one of {A, B, C, D}
    ood             : answer must equal constants.REFUSAL_STRING
    adversarial_ood : answer must equal constants.REFUSAL_STRING
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from llm_ft.constants import REFUSAL_STRING, VALID_DOMAINS


class RecordSchema(pa.DataFrameModel):
    """Structural schema — no cross-field logic here."""

    id: Series[str] = pa.Field(
        nullable=False,
        unique=True,
        str_matches=r"^[a-z0-9][a-z0-9_\-]{2,63}$",
        description="Stable, unique, filesystem-safe identifier.",
    )
    domain: Series[str] = pa.Field(
        nullable=False,
        isin=list(VALID_DOMAINS),
        description="One of: math, physics, ood, adversarial_ood.",
    )
    prompt: Series[str] = pa.Field(
        nullable=False,
        str_length={"min_value": 4, "max_value": 8000},
        description="User-facing question or instruction.",
    )
    answer: Series[str] = pa.Field(
        nullable=False,
        str_length={"min_value": 1, "max_value": 4000},
        description="Reference answer (numeric, symbolic, or letter).",
    )
    reasoning: Series[str] = pa.Field(
        nullable=True,
        description="Optional chain-of-thought / explanation.",
    )
    meta: Series[object] = pa.Field(
        nullable=False,
        description="Free-form per-source metadata (dict).",
    )

    class Config:
        strict = True
        coerce = False
        ordered = False


# ── cross-field validation ─────────────────────────────────────


def _domain_specific_errors(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for i, row in df.iterrows():
        dom = row["domain"]
        ans = row["answer"]

        if dom == "physics":
            if ans not in {"A", "B", "C", "D"}:
                errors.append(
                    f"row {i} (id={row['id']}): physics answer must be " f"A/B/C/D, got {ans!r}"
                )
        elif dom in {"ood", "adversarial_ood"}:
            if ans != REFUSAL_STRING:
                errors.append(
                    f"row {i} (id={row['id']}): {dom} answer must equal "
                    f"REFUSAL_STRING, got {ans!r}"
                )
        elif dom == "math":
            # math answer must have at least one digit or be a simple symbolic
            if not any(c.isdigit() for c in ans) and not any(
                c in r"abcdefghijklmnopqrstuvwxyz/\{}()[]+-=." for c in ans.lower()
            ):
                errors.append(f"row {i} (id={row['id']}): math answer looks empty/blank: {ans!r}")
        else:
            errors.append(f"row {i} (id={row['id']}): unknown domain {dom!r}")
    return errors


def validate_records(
    df: pd.DataFrame,
    *,
    lazy: bool = True,
    max_errors_shown: int = 10,
) -> pd.DataFrame:
    """Validate a dataframe against RecordSchema + cross-field rules.

    Raises:
        ValueError if the dataframe fails structural or domain-specific checks.
    """
    # 1. structural
    try:
        df = RecordSchema.validate(df, lazy=lazy)
    except pa.errors.SchemaErrors as e:
        # surface a compact summary, not a 100-line traceback
        n = len(e.failure_cases) if hasattr(e, "failure_cases") else 0
        preview = ""
        if hasattr(e, "failure_cases") and e.failure_cases is not None:
            preview = e.failure_cases.head(max_errors_shown).to_string(index=False)
        raise ValueError(f"RecordSchema validation failed ({n} failures):\n{preview}") from e

    # 2. cross-field
    errors = _domain_specific_errors(df)
    if errors:
        preview = "\n".join(errors[:max_errors_shown])
        more = (
            f"\n  ... and {len(errors) - max_errors_shown} more"
            if len(errors) > max_errors_shown
            else ""
        )
        raise ValueError(f"domain-specific validation failed:\n{preview}{more}")

    return df


# ── self-test ──────────────────────────────────────────────────


def _self_test() -> None:
    """Quick smoke: good df passes, bad df fails."""
    ok = pd.DataFrame(
        [
            {
                "id": "gsm8k-0001",
                "domain": "math",
                "prompt": "What is 2+2?",
                "answer": "4",
                "reasoning": "2+2=4",
                "meta": {"source": "gsm8k"},
            },
            {
                "id": "mmlu-phys-0001",
                "domain": "physics",
                "prompt": "Which is correct? A) ...",
                "answer": "B",
                "reasoning": None,
                "meta": {"source": "mmlu"},
            },
            {
                "id": "ood-0001",
                "domain": "ood",
                "prompt": "How are you?",
                "answer": REFUSAL_STRING,
                "reasoning": None,
                "meta": {"source": "manual"},
            },
        ]
    )
    validate_records(ok)
    print("✓ self-test: valid dataframe passes")

    bad_physics = ok.copy()
    bad_physics.loc[1, "answer"] = "X"
    try:
        validate_records(bad_physics)
        raise AssertionError("should have failed for bad physics answer")
    except ValueError as e:
        assert "physics answer must be" in str(e)
        print("✓ self-test: bad physics answer rejected")

    bad_dup = pd.concat([ok, ok.iloc[[0]]], ignore_index=True)
    try:
        validate_records(bad_dup)
        raise AssertionError("should have failed for duplicate id")
    except ValueError as e:
        assert "validation failed" in str(e).lower()
        print("✓ self-test: duplicate id rejected")

    bad_ood = ok.copy()
    bad_ood.loc[2, "answer"] = "I don't know"
    try:
        validate_records(bad_ood)
        raise AssertionError("should have failed for bad ood answer")
    except ValueError as e:
        assert "REFUSAL_STRING" in str(e)
        print("✓ self-test: bad ood answer rejected")


if __name__ == "__main__":
    _self_test()
