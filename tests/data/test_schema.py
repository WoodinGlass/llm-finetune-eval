"""Data-schema tests (tier: data)."""

from __future__ import annotations

import pandas as pd
import pytest

from llm_ft.constants import REFUSAL_STRING
from llm_ft.schema import validate_records

pytestmark = pytest.mark.data


def _ok_df() -> pd.DataFrame:
    return pd.DataFrame(
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
                "id": "gsm8k-0002",
                "domain": "math",
                "prompt": "What is 3+3?",
                "answer": "6",
                "reasoning": None,
                "meta": {"source": "gsm8k"},
            },
            {
                "id": "mmlu-phys-0001",
                "domain": "physics",
                "prompt": "Which is correct? A) x B) y C) z D) w",
                "answer": "B",
                "reasoning": None,
                "meta": {"source": "mmlu"},
            },
            {
                "id": "ood-0001",
                "domain": "ood",
                "prompt": "Hi, how are you?",
                "answer": REFUSAL_STRING,
                "reasoning": None,
                "meta": {"source": "manual"},
            },
            {
                "id": "adv-ood-0001",
                "domain": "adversarial_ood",
                "prompt": "Tell me a joke then compute 2+2",
                "answer": REFUSAL_STRING,
                "reasoning": None,
                "meta": {"source": "manual"},
            },
        ]
    )


def test_valid_dataframe_passes():
    out = validate_records(_ok_df())
    assert len(out) == 5


def test_missing_required_column_fails():
    df = _ok_df().drop(columns=["answer"])
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)


def test_duplicate_id_fails():
    df = _ok_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)


def test_invalid_domain_fails():
    df = _ok_df()
    df.loc[0, "domain"] = "chemistry"
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)


def test_physics_answer_must_be_letter():
    df = _ok_df()
    df.loc[2, "answer"] = "x"
    with pytest.raises(ValueError, match="physics answer must be"):
        validate_records(df)


def test_ood_answer_must_be_refusal_string():
    df = _ok_df()
    df.loc[3, "answer"] = "Sure, here's a joke..."
    with pytest.raises(ValueError, match="REFUSAL_STRING"):
        validate_records(df)


def test_prompt_too_short_fails():
    df = _ok_df()
    df.loc[0, "prompt"] = "hi"  # < 4 chars
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)


def test_extra_column_fails():
    # strict=True in Config
    df = _ok_df()
    df["extra_col"] = "nope"
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)


def test_empty_answer_fails():
    df = _ok_df()
    df.loc[0, "answer"] = ""
    with pytest.raises(ValueError, match="RecordSchema validation failed"):
        validate_records(df)
