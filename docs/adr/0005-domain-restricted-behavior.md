# ADR 0005 — Domain-restricted behavior with fixed refusal

- **Status:** Accepted
- **Date:** M0.4 (recreated M2.6 after git-history loss)
- **Deciders:** project owner
- **Related:** ADR 0002 (base model), ADR 0004 (eval methodology)

## Context

The fine-tuned model is intended for **mathematics and physics** problem
solving. It is not a general-purpose assistant and must not behave like one.

Two independent failure modes motivate this:

1. **Safety.** A 7B model fine-tuned on math data has no reliable guardrails
   against confidently wrong advice on health, law, or finance. Silence is
   better than hallucinated advice.
2. **Portfolio scope.** A model that answers everything looks unfocused.
   A model that demonstrably refuses outside its domain is honest about
   its boundaries — a production-quality signal.

Refusal must be **testable**. Varied phrasing (Sorry, I only do math) is
friendlier but not CI-assertable. Exact-match is.

## Decision

### 1. Refusal is a fixed, exact string (English)

The model must respond with **exactly**:

```
I'm configured to help with mathematics and physics only.
```

- No prefix, no suffix, no variation.
- Language: English regardless of prompt language.
- Stored as `REFUSAL_STRING` in `src/llm_ft/constants.py`; imported by
  training, eval, and serving. Changing it is a breaking change and bumps
  the model minor version.

### 2. Two layers of enforcement

| Layer | Mechanism | Milestone |
|---|---|---|
| Model behaviour | SFT on ~4% refusal examples | M2 |
| Gateway guardrail | regex / classifier check on input; replace output if OOD | M5 |

- **Model layer** ensures the adapter itself refuses even if the gateway is
  bypassed or down.
- **Gateway layer** ensures that a jailbroken model cannot leak an OOD
  answer to the user.

### 3. Training data mix

Target: refusal signal at 3-5% of the total training set (see ADR 0007 for
why this was lowered from 8% to 5% to 4% across v1/v2/v3).

| Domain | Share (v3) | Purpose |
|---|---|---|
| math | ~47% | Positive — reasoning |
| physics | ~45% | Positive — reasoning |
| ood | ~3% | Negative — refusal |
| adversarial_ood | ~1% | Negative — hard / mixed |
| false_refusal | ~4% | Anti-over-refusal — legit math phrased conversationally |

### 4. Eval metrics

Added to `docs/slo.md` §3.1:

| Metric | Minimum | Stretch |
|---|---|---|
| OOD refusal rate | >= 95% | >= 99% |
| False refusal rate | <= 2% | <= 1% |
| Refusal string exact-match | 100% | — |

### 5. Eval sets

- `ood_test.jsonl` — 50 OOD prompts, stratified across categories.
- `adversarial_ood` cases — mixed-domain prompts, human-reviewed.
- `false_refusal_test.jsonl` — 50 conversational math/physics prompts that
  MUST be answered, not refused.

## Consequences

### Positive

- Behaviour is **binary testable** — unit tests assert exact match.
- Two-layer enforcement → single point of failure cannot leak OOD.
- Refusal mix is small enough not to hurt reasoning (when tuned; see ADR 0007).
- Model card declares a crisp scope: math + physics only.

### Negative / risks

| Risk | Mitigation |
|---|---|
| **Over-refusal** — legit math refused | `false_refusal_test` in every eval; hard cap 2% |
| **Adversarial leak** — mixed prompt slips | `adversarial_ood` training + gateway classifier |
| **String drift** | Exact-match unit test in CI |
| **Language mismatch** — Indonesian prompt → English refusal | Documented; future multilingual work tracked in `docs/limitations.md` |
| **Trade-off with MMLU** | MMLU has non-math items; forgetting check uses math/physics subset |

### Non-goal

Multi-turn graceful degradation. Refusal is a single-turn, final response.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| No refusal — let base model answer | Unsafe; unbounded scope; bad portfolio signal |
| Varied refusal phrasing | Not exactly testable; drift goes unnoticed |
| Refusal only in gateway | Model still hallucinates if gateway bypassed or during training |
| Refusal only in model | Jailbreaks leak; no hard block at HTTP layer |
| Refuse in user's language | Multiplies eval surface; defer to future work |
| Separate classifier model | Adds second model to train/serve; overkill for MVP |

## References

- `docs/slo.md` §3.1 — metrics
- `docs/adr/0002` — base model
- `docs/adr/0004` — eval methodology
- `src/llm_ft/constants.py` — `REFUSAL_STRING`
- `docs/cards/model.md` — declares scope
