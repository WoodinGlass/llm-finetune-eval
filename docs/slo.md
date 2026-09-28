# Service Level Objectives (SLO)

**Status:** accepted · **Owner:** WoodinGlass · **Last updated:** M0.5
**Baseline numbers:** measured and frozen at M0.5 (see § [Baseline](#baseline), § [Revision history](#6-revision-history)).

This document is the contract between the model we ship and the users of
that model. Every threshold here is *measurable*, *reproducible*, and
*enforced in CI at M6*. If a number cannot be reproduced by
`make eval-baseline` on a clean machine, it does not belong here.

---

## 1. Scope

| In scope | Out of scope |
|---|---|
| Task accuracy on the frozen math + physics test set | General chit-chat quality |
| Hallucination / faithfulness on closed-form answers | Multimodal / vision |
| General-capability regression (MMLU, HellaSwag) | Long-context beyond 8k tokens |
| p50 / p95 / p99 latency, TTFT, throughput | Multi-tenant quota fairness |
| Cost per 1k requests, cost per 1M tokens | Energy / carbon accounting |

Target task: **solve competition-style math and physics problems**,
producing a final numeric or symbolic answer plus a short derivation.

---

## 2. Baseline

Baseline = **`Qwen2.5-Math-7B-Instruct` + 4-shot prompt**, no fine-tuning.
Measured on the frozen test set (see `docs/cards/data.md`, filled in M1).

| Metric | Baseline | Measured at | Command |
|---|---|---|---|
| Task accuracy (exact-match, math) | **96.0%** (CI [92.0%, 99.0%]) | M0.5 | `make eval-baseline` |
| Task accuracy (exact-match, physics) | **72.5%** (CI [62.7%, 81.4%]) | M0.5 | `make eval-baseline` |
| Hallucination rate (LLM-judge) | deferred to M3 | M3 | `make eval-baseline` |
| MMLU (5-shot) | deferred to M3 | M3 | `make eval-baseline` |
| HellaSwag | deferred to M3 | M3 | `make eval-baseline` |
| OOD refusal rate (pre-FT, expected ~0%) | **0.0%** | M0.5 | `make eval-baseline` |
| False refusal rate (pre-FT) | **0.0%** | M0.5 | `make eval-baseline` |
| False-refusal accuracy (pre-FT) | **44.0%** | M0.5 | `make eval-baseline` |
| Test-set size | 302 items (100 math + 102 physics + 50 OOD + 50 false-refusal) | M0.5 | `make data-report` |
| Test-set hash (SHA256) | `f10f4893220d2d283cf6b5c4f35807f51158eca06f5a2ac39dac6f33a7d37f5a` | M0.5 | `dvc.lock` |

> ⚠ **Baseline is frozen.** Any change to the test set invalidates the
> baseline; the hash above is the ground truth for "same test set".

---

## 3. Target SLO (post fine-tune)

Fine-tuned model = QLoRA adapter on top of the same base, evaluated on the
**same frozen test set**, with **3 seeds**, reporting **bootstrap 95% CI**.

### 3.1 Quality

| Metric | Minimum (CI gate) | Stretch (dashboard) | Tolerance | Enforcement |
|---|---|---|---|---|
| Task accuracy — math | ≥ **96%** | ≥ **99%** | −1 pt vs min | CI eval gate (M6) |
| Task accuracy — physics | ≥ **80%** | ≥ **90%** | −1 pt vs min | CI eval gate (M6) |
| Hallucination rate (LLM-judge) | ≤ **3%** | ≤ **1.5%** | +0.5 pt | CI eval gate (M6) |
| Judge agreement with humans (Cohen's κ) | ≥ **0.60** | ≥ **0.75** | — | manual review (M3) |
| Catastrophic forgetting (MMLU Δ) | ≥ **−2 pts** | ≥ **−1 pt** | −0.5 pt | CI eval gate (M6) |
| Refusal on red-team set | ≥ **95%** | ≥ **99%** | — | nightly eval (M6) |
| OOD refusal rate | ≥ **95%** | ≥ **99%** | −1 pt vs min | CI eval gate (M6) |
| False refusal rate | ≤ **2%** | ≤ **1%** | +0.5 pt | CI eval gate (M6) |

**Dual target rationale:** the *minimum* column is what the CI eval
gate enforces on every PR (M6). The *stretch* column is the dashboard
target we aim for after full fine-tune. Both are reported on every run;
only the minimum blocks merges.

### 3.2 Latency & throughput (serving, M5)

Reference hardware: **1× NVIDIA A10G (24 GB)**, vLLM ≥ 0.6.3, batch concurrency = 8.

| Metric | Target | p-value | Notes |
|---|---|---|---|
| Time to first token (TTFT) | ≤ **300 ms** | p50 | excludes network RTT |
| End-to-end latency | ≤ **400 ms** | p50 | non-streaming, ≤ 256 output tokens |
| End-to-end latency | ≤ **1.2 s** | p95 | ditto |
| End-to-end latency | ≤ **2.5 s** | p99 | ditto |
| Throughput | ≥ **25 tok/s/user** | mean | at concurrency 8 |
| Error rate (5xx) | ≤ **0.5%** | — | rolling 5 min |
| Availability | ≥ **99.5%** | monthly | excluding announced maintenance |

### 3.3 Cost

Reference price: A10G on-demand ≈ **$1.00 / GPU-hour** (RunPod / Modal, Q4 2024).

| Metric | Target | Notes |
|---|---|---|
| Cost per 1k requests | ≤ **$0.15** | includes GPU amortization + obs |
| Cost per 1M tokens (input+output) | ≤ **$1.20** | ditto |
| Cold-start time (serverless) | ≤ **90 s** | model load + warmup |
| Monthly fixed cost (idle) | ≤ **$5** | registry + storage + obs |

> Costs above are the **targets**. Real numbers from `tests/load/` go into
> `docs/cost.md` at M5.

---

## 4. Regression policy

Any PR that changes **training/, eval/, serving/, configs/, or data/** must
pass the CI eval gate:

| Change type | Required check |
|---|---|
| Config-only (lr, batch, rank) | eval gate on golden set |
| Data (add/remove samples) | data tests + full eval + decontamination |
| Code (training, eval) | unit tests + eval gate |
| Serving (gateway, guardrails) | smoke + load test on staging |
| Docs only | no gate |

**Definition of "regression":** any of these fail on the golden set
(500 samples, seeded, frozen):

- math accuracy < `baseline_math − EVAL_GATE_TOLERANCE_PTS` (default 1.0)
- physics accuracy < `baseline_physics − EVAL_GATE_TOLERANCE_PTS`
- hallucination rate > `baseline_halluc + 0.5 pt`
- MMLU drop > 2 pts

Gate is enforced at M6 via GitHub Actions. Thresholds are env-driven:
`EVAL_GATE_TOLERANCE_PTS` in `.env.example`.

---

## 5. Measurement protocol

To keep numbers comparable, all measurements follow:

1. **Frozen test set** — hash recorded in `dvc.lock`; changing it invalidates baseline.
2. **Deterministic inference** — `temperature=0`, `seed=42`, `top_p=1.0`.
3. **3 seeds** for training; report mean and 95% bootstrap CI (1000 resamples).
4. **Judge calibration** — LLM-as-judge is only trusted if Cohen's κ ≥ 0.6 vs 50–100 human labels (M3).
5. **Hardware recorded** — GPU model, driver, CUDA, vLLM version logged with each number.
6. **Cost model formula** published in `docs/cost.md` (M5) — no hand-waved $/1k.

---

## 6. Revision history

| Date | Author | Change |
|---|---|---|
| (M0.4) | WoodinGlass | Initial draft with TBD baselines |
| (M0.5) | WoodinGlass | Baseline measured: math 96.0%, physics 72.5%, OOD refusal 0.0%, false-refusal rate 0.0% |

---

## 7. References

- Baseline command: `make eval-baseline` (M0.5)
- Frozen test set: `data/processed/test.lock` (M1)
- Cost report: `docs/cost.md` (M5)
- Eval methodology: `docs/adr/0004-eval-methodology.md` (M0.4)
- Roadmap: `README.md § Roadmap`
