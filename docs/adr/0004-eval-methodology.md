# ADR 0004 — Three-layer eval with calibrated LLM-as-judge

- **Status:** accepted
- **Date:** M0.4
- **Deciders:** WoodinGlass
- **Supersedes:** —

## Context

"Did fine-tuning help?" is a deceptively hard question. Common failure modes
in portfolio projects:

- Reporting a single accuracy number with no CI, no seeds, no error analysis.
- Using an LLM-as-judge with no human calibration → judge drift goes unnoticed.
- Not checking that fine-tuning didn't lobotomize the model on general tasks.
- No frozen test set → numbers from yesterday aren't comparable to today.

We need an eval harness whose output is **trustworthy enough to gate a merge**
and to justify a rollback.

## Decision

Adopt a **three-layer eval**:

### Layer 1 — Task metrics (frozen test set)

- Exact-match / F1 on math and physics problems, final answer only.
- **Bootstrap 95% CI** (1000 resamples) reported alongside point estimate.
- **3 training seeds** → variance reported, not just mean.
- Frozen test set with SHA256; changing it invalidates the baseline.

### Layer 2 — Regression checks (`lm-evaluation-harness`)

- MMLU (5-shot) and HellaSwag to detect **catastrophic forgetting**.
- Threshold: MMLU Δ ≥ **−2 pts** vs base. HellaSwag Δ ≥ **−3 pts**.
- Run nightly (M6) — not on every PR — to keep CI fast.

### Layer 3 — LLM-as-judge (calibrated)

- Judge = a stronger model (e.g. `gpt-4o-mini`, `claude-haiku`) scoring
  hallucination / faithfulness on 1–5 scale.
- **Calibration set:** 50–100 human labels; judge is only trusted if
  **Cohen's κ ≥ 0.60** vs the human labels.
- Judge model + version + prompt hash logged with every score; changing any
  of these re-triggers calibration.
- Judge is run on a **stratified sample** (not every test item) to control cost.

### Error analysis (mandatory)

Every report includes **top-20 failures**, categorized:

- `wrong_formula` — correct reasoning, wrong algebra
- `wrong_final` — right approach, arithmetic slip
- `refusal` — model declines to answer
- `format` — answer correct but not parseable
- `hallucination` — invents a constant, theorem, or unit

Aggregate numbers without failure-mode breakdown do not count as a report.

## Consequences

### Positive

- Numbers come with uncertainty, so "improvement" is falsifiable.
- Catastrophic forgetting caught before it ships.
- Judge has a paper trail (κ, prompts, calibration set) so it can be audited.
- Error analysis feeds the **error bank** (M9) that drives retraining.

### Negative / risks

- 3 seeds × bootstrap × LLM-judge = non-trivial GPU + API cost. Mitigation:
  full eval nightly, golden-set (500 items) eval on every PR.
- Judge calibration requires human labeling — one-time cost of 2–3 hours;
  recalibration every time the judge model bumps.
- MMLU / HellaSwag add ~15 min to nightly; accepted.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Single accuracy number | No uncertainty → no legitimate gate |
| ROUGE / BLEU on full derivation | Rewards token overlap, not correctness |
| Hand-written rubric only | Doesn't scale; can't run in CI |
| LLM-judge only (no humans) | Judge can drift; κ unknown; unbounded trust |
| Ragas / DeepEval out of the box | Useful *tools*, but methodology still needed |

## References

- `docs/slo.md` §3.1, §5 (targets + protocol)
- `eval/` (M3 — implementation)
- `docs/cards/data.md` (M1 — decontamination vs MMLU/HellaSwag)
- `docs/runbook/retrain.md` (M9 — where error bank feeds in)
