"""Domain-aware prompt builders for Qwen2.5-Math-7B-Instruct."""

from __future__ import annotations

from llm_ft.constants import (
    DOMAIN_ADVERSARIAL_OOD,
    DOMAIN_MATH,
    DOMAIN_OOD,
    DOMAIN_PHYSICS,
)

SYSTEM_MATH = "Please reason step by step, and put your final answer within \\boxed{}."

SYSTEM_PHYSICS = (
    "You are a physics expert. Read the multiple-choice question carefully. "
    "Reason step by step if needed, then output your final answer as a single "
    "capital letter (A, B, C, or D) inside \\boxed{}, e.g. \\boxed{A}. "
    "Do not include any other text after the boxed answer."
)

SYSTEM_OOD = None


def system_prompt_for(domain: str) -> str | None:
    if domain == DOMAIN_MATH:
        return SYSTEM_MATH
    if domain == DOMAIN_PHYSICS:
        return SYSTEM_PHYSICS
    if domain in (DOMAIN_OOD, DOMAIN_ADVERSARIAL_OOD):
        return SYSTEM_OOD
    raise ValueError(f"unknown domain: {domain!r}")


def build_messages(domain: str, prompt: str) -> list[dict[str, str]]:
    sys = system_prompt_for(domain)
    msgs: list[dict[str, str]] = []
    if sys is not None:
        msgs.append({"role": "system", "content": sys})
    msgs.append({"role": "user", "content": prompt})
    return msgs
