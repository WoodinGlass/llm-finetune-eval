# llm-finetune-eval

> End-to-end LLM fine-tuning, evaluation, serving, and retraining loop — built to production standards.

[![CI](https://github.com/WoodinGlass/llm-finetune-eval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WoodinGlass/llm-finetune-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A reference implementation of a **production-grade LLM pipeline**: QLoRA fine-tuning of an open-weight model, statistical evaluation with LLM-as-judge, containerized serving via vLLM, guardrails, CI/CD quality gates, observability, and a closed retraining loop.

**Goal:** demonstrate the full lifecycle from a raw dataset to a monitored, rollback-able, evaluated model in production — not just a notebook.

---

## Problem (one sentence)

Teams need a **reproducible, testable, and observable LLM lifecycle** — from raw data → QLoRA fine-tune → statistical eval → guarded serving → drift-aware retraining.

## What this is / is not

| This is | This is not |
|---|---|
| End-to-end LLM lifecycle (data → train → eval → serve → retrain) | A notebook dump or a single fine-tune script |
| Production serving (vLLM + FastAPI + guardrails) | A chatbot UI |
| Statistical eval with bootstrap CI + calibrated LLM-judge | Vibes-based "it looks better" |
| Portfolio-grade with CI, tests, Docker, observability | Production SaaS with billing / SLA |
| Reproducible via `git SHA + data hash + config hash` | Non-reproducible seed-of-the-day training |

---

## Table of Contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Service Level Objectives (SLO)](#service-level-objectives-slo)
- [Roadmap](#roadmap)
- [Development Workflow](#development-workflow)
- [Reproducibility & Versioning](#reproducibility--versioning)
- [Evaluation Methodology](#evaluation-methodology)
- [Serving & Guardrails](#serving--guardrails)
- [Observability](#observability)
- [Retraining Loop](#retraining-loop)
- [Security & Threat Model](#security--threat-model)
- [Cost Model](#cost-model)
- [Documentation](#documentation)
- [License](#license)

---

## Why this project

Most portfolio "LLM fine-tune" projects stop at a Colab notebook that produces a model file. That's not what production looks like.

This repo answers the questions that actually matter when shipping LLMs:

- **Is the new model genuinely better?** → eval harness with bootstrap CI, multi-seed, human-calibrated LLM-as-judge, and catastrophic-forgetting checks.
- **Can I ship it safely?** → CI eval gate, canary deploy, one-command rollback, guardrails.
- **Can I afford it?** → p50/p95/p99 latency, TTFT, tokens/s, cost per 1k requests, cost per 1M tokens tracked and alerted.
- **Can I trust it tomorrow?** → drift detection, quality sampling in prod, feedback loop that feeds the next retrain.

The **payload** is the fine-tuned adapter + eval harness. Everything else (DVC, CI/CD, observability, registry) is **pipeline** that makes the payload safe to ship.

---

## Architecture

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Data (DVC)  │──▶│  QLoRA Train │──▶│  Eval Harness│──▶│   Registry   │
│  + validation│   │  (HF + PEFT) │   │ (CI gate)    │   │  (staging)   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                │
                                                                ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Retraining  │◀──│ Observability│◀──│   Gateway    │◀──│  vLLM Server │
│    Loop      │   │ (Prom/Graf)  │   │ (FastAPI +   │   │ (OpenAI-API) │
│              │   │              │   │  guardrails) │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                          ▲                   ▲
                          │                   │
                    ┌─────┴──────┐      ┌─────┴──────┐
                    │  Langfuse  │      │  Feedback  │
                    │  (tracing) │      │  (👍 / 👎) │
                    └────────────┘      └────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language / deps | Python 3.11+, `uv` |
| Fine-tuning | HuggingFace Transformers, PEFT (LoRA / QLoRA), bitsandbytes |
| Config | Hydra |
| Tracking | W&B / MLflow (git SHA + data hash + config) |
| Data versioning | DVC, Pandera (schema), MinHash dedup, Presidio (PII) |
| Evaluation | `lm-evaluation-harness`, DeepEval / Ragas, promptfoo, bootstrap CI |
| Serving | vLLM (OpenAI-compatible), FastAPI gateway, Llama Guard |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions, Trivy, pip-audit |
| Observability | Prometheus, Grafana, Langfuse / OpenTelemetry, Evidently |
| Load testing | Locust / k6 |
| IaC (supporting) | Terraform |

---

## Project Structure

```text
llm-finetune-eval/
├── data/
│   ├── raw/                # immutable raw dumps (DVC-tracked)
│   └── processed/          # cleaned, deduped, PII-scrubbed
├── training/               # QLoRA fine-tune scripts
├── eval/                   # eval harness, LLM-judge, bootstrap CI
├── configs/                # Hydra configs (model, data, train, eval)
├── experiments/            # small artifacts, sweep results
├── serving/                # vLLM config, FastAPI gateway, guardrails
├── deploy/                 # docker-compose, IaC, env templates
├── monitoring/             # dashboards, alerts, drift jobs
├── registry/               # model promote / rollback scripts
├── tests/
│   ├── unit/               # fast, no external deps
│   ├── data/               # schema + DQ tests
│   ├── smoke/              # small end-to-end train+serve
│   └── load/               # Locust / k6 scripts
├── docs/
│   ├── adr/                # Architecture Decision Records
│   ├── runbook/            # incident playbooks (OOM, latency, quality)
│   └── cards/              # model card, data card
├── .github/workflows/      # ci.yml, nightly.yml, release.yml
├── dvc.yaml
├── Makefile
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## Quickstart

```bash
# 1. Clone & install (uv handles everything)
git clone https://github.com/<username>/llm-finetune-eval.git
cd llm-finetune-eval
make setup

# 2. Configure
cp .env.example .env
# edit .env with your W&B key, HF token, etc.

# 3. Pull data (DVC)
make data-pull

# 4. Smoke: fine-tune a tiny model on 100 samples
make train-smoke

# 5. Evaluate
make eval

# 6. Serve locally
make serve

# 7. Full pipeline
make pipeline
```

**One command to reproduce everything:** `make pipeline`

---

## Service Level Objectives (SLO)

Targets for the fine-tuned model on the target task, measured against the frozen test set. See [`docs/slo.md`](docs/slo.md) for the full spec and revision history.

| Metric | Baseline (few-shot) | Target | Notes |
|---|---|---|---|
| Task accuracy | TBD (M0) | ≥ baseline + 10 pts | Frozen test set, 3 seeds |
| Hallucination rate | TBD | ≤ 3% | LLM-judge, calibrated to human labels |
| Catastrophic forgetting | — | MMLU delta ≥ -2 pts | via `lm-evaluation-harness` |
| p50 latency | — | ≤ 400 ms | 8B model, 1× A10G |
| p95 latency | — | ≤ 1.2 s | Streaming excluded from SLO |
| TTFT (time to first token) | — | ≤ 300 ms | |
| Throughput | — | ≥ 25 tok/s/user | Target concurrency |
| Cost per 1k requests | — | ≤ $0.15 | Includes GPU-hour amortization |
| Cost per 1M tokens | — | ≤ $1.20 | |

Baseline numbers are populated in **M0**. Any regression beyond these thresholds blocks merge in **M6**.

---

## Development Workflow

### Commits
Conventional Commits — enforced by `pre-commit`:
```text
feat(eval): add bootstrap CI for accuracy metric
fix(serving): guard against empty prompt
chore(ci): bump trivy-action to v0.28
```

### Tests
Three tiers, never mixed:
- `pytest -m unit` — fast, no external deps (default)
- `pytest -m integration` — needs Docker/DB
- `pytest -m slow` — long-running, nightly only

### CI Gates
Every PR runs:
1. `ruff`, `mypy`
2. unit + data tests
3. smoke-train (tiny model, 100 samples)
4. **eval gate** — fails if task accuracy drops > 1 pt vs `main`
5. Trivy + `pip-audit`

Nightly: full eval, load test, drift check.

---

## Reproducibility & Versioning

Every training run is pinned by the triple **(git SHA, DVC data hash, Hydra config hash)**. This triple is logged to W&B/MLflow and stored in the model card.

- **Data:** DVC (`dvc.lock` is committed)
- **Code:** git SHA
- **Config:** Hydra hash
- **Weights:** registered with semver, checksum verified on load

If any of the three changes, it's a new artifact — never overwrite.

---

## Evaluation Methodology

Three independent layers:

1. **Task metrics** — accuracy / F1 on the frozen test set, reported with **bootstrap CI (1000 resamples, 95%)** and **multi-seed** variance.
2. **Regression checks** — `lm-evaluation-harness` on MMLU / HellaSwag to catch catastrophic forgetting (delta ≥ −2 pts required).
3. **LLM-as-judge** — hallucination / faithfulness, **calibrated against 50–100 human labels**. Judge agreement with humans is reported (Cohen's κ). Judge is not trusted if κ < 0.6.

Every report includes **error analysis**: top-20 failures with categorization, not just aggregate numbers.

---

## Serving & Guardrails

- **vLLM** backend (OpenAI-compatible API) in Docker.
- **FastAPI gateway** in front of it:
  - API key auth
  - per-key rate limiting
  - input validation (length, encoding, PII)
  - max tokens, hard timeout
  - streaming SSE
  - `/health`, `/ready`
- **Guardrails:**
  - Input: Llama Guard / regex blocklist
  - Output: PII redaction, toxicity check
- **Load tested** with Locust / k6 — SLO numbers in `docs/slo.md` are the pass criteria.

---

## Observability

Structured JSON logs with correlation IDs from the first commit. No retrofit.

- **Metrics:** Prometheus → Grafana (latency, error rate, tokens, GPU util, queue depth)
- **Traces:** Langfuse / OpenTelemetry (per-request span: gateway → guardrail → vLLM → response)
- **Quality:** sampled prod traffic → LLM-judge → dashboard
- **Drift:** Evidently on input distribution + output length + refusal rate
- **Feedback:** 👍/👎 UI, piped into the error bank for M9

Alerts (page-worthy):
- p95 latency > 2× SLO for 5 min
- error rate > 2% for 5 min
- judge-scored quality drop > 5 pts over 1h
- GPU memory > 95% for 10 min

---

## Retraining Loop

```text
prod traffic ──▶ feedback (👍/👎) ──▶ error bank
                                          │
                                          ▼
                                     curation
                                          │
                                          ▼
                                new dataset version (DVC)
                                          │
                                          ▼
                                     retrain (M2)
                                          │
                                          ▼
                                    auto eval (M3)
                                          │
                             ┌────────────┴────────────┐
                             ▼                         ▼
                        challenger wins           challenger loses
                             │                         │
                             ▼                         ▼
                      manual approval            discard + log
                             │
                             ▼
                     promote in registry
```

One full cycle is a **M9 acceptance test** and must be reproducible via `make retrain-cycle`.

---

## Security & Threat Model

Mapped to **OWASP LLM Top 10** (2025):
- **LLM01 Prompt Injection** → input sanitization, Llama Guard, output constraints
- **LLM02 Insecure Output** → PII redaction, structured output validation
- **LLM03 Training Data Poisoning** → data provenance, decontamination, MinHash dedup
- **LLM04 Model DoS** → rate limiting, max tokens, timeout, request queue caps
- **LLM05 Supply Chain** → `pip-audit`, Trivy, pinned deps, verified model checksums
- **LLM06 Sensitive Info Disclosure** → Presidio scrub in data + at serving time
- **LLM07 Insecure Plugin Design** → n/a (no tool use in v1)
- **LLM08 Excessive Agency** → n/a (no agent actions in v1)
- **LLM09 Overreliance** → disclaimer, calibrated confidence, eval gate
- **LLM10 Model Theft** → API key rotation, no raw weight download without auth

Full threat model in `docs/threat-model.md` (M10).

---

## Cost Model

Tracked per 1k requests and per 1M tokens. Populated in M5 with real numbers from load tests.

| Component | Cost driver | Notes |
|---|---|---|
| GPU (serving) | $/hr × uptime | A10G / L4 / similar |
| GPU (training) | $/hr × hours | QLoRA on 8B ≈ 4–8 hrs |
| Storage | $/GB-month | DVC remote + registry |
| Egress | $/GB | HF Hub downloads |
| Observability | $/event | Langfuse / Grafana Cloud |

See `docs/cost.md` for the current numbers.

---

## Documentation

- [`SETUP.md`](SETUP.md) — one-command reproduction
- [`docs/slo.md`](docs/slo.md) — service level objectives
- [`docs/adr/`](docs/adr/) — Architecture Decision Records (0001–0004 accepted)
- [`docs/runbook/`](docs/runbook/) — incident playbooks
- [`docs/cards/model.md`](docs/cards/model.md) — model card
- [`docs/cards/data.md`](docs/cards/data.md) — data card
- [`docs/threat-model.md`](docs/threat-model.md) — OWASP LLM Top 10 mapping
- [`docs/cost.md`](docs/cost.md) — cost report

---

## Roadmap

Each milestone ships **runnable, tested, and documented** code — not stubs.
Status legend: `[todo]` not started · `[wip]` in progress · `[done]` accepted.

### M0 — Scoping & baseline [done]

- [x] `docs/slo.md` with measurable targets (accuracy, hallucination, p95 latency, cost/1k, cost/1M)
- [x] Baseline = base model + few-shot prompt, scores committed to `docs/slo.md`
- [x] `uv` project setup, `pyproject.toml` single source of truth with self-contained extras
- [x] `pre-commit` hooks (ruff, ruff-format, mypy, end-of-file-fixer, detect-secrets)
- [x] Multi-stage `Dockerfile` (builder + runtime, non-root)
- [x] `Makefile` with `setup`, `lint`, `test`, `eval-baseline`
- [x] `.gitignore`, `LICENSE` (MIT), `.env.example`, `CHANGELOG.md`, `README.md`
- [x] **Exit criteria:** `make setup` runs on a clean machine; `make eval-baseline` reproduces committed baseline scores ± tolerance

### M1 — Data pipeline [todo]

- [ ] Pandera schema for train/eval records (required fields, dtypes, ranges)
- [ ] MinHash + LSH dedup (near-duplicate removal, threshold documented)
- [ ] PII scrub: Presidio + regex emails
- [ ] License audit per source dataset; recorded in `docs/cards/data.md`
- [ ] Train-vs-test decontamination (13-gram overlap check)
- [ ] Frozen test set: `data/processed/test.lock` with content hash
- [ ] DVC pipeline (`dvc.yaml`): raw → clean → dedup → split → frozen
- [ ] `docs/cards/data.md` (provenance, size, dedup ratio, PII stats, known biases)
- [ ] **Exit criteria:** `make test-data` green; `dvc.lock` records data hashes; test set locked and reproducible

### M2 — Training [todo]

- [ ] QLoRA fine-tune via Hydra config (`configs/train/*.yaml`)
- [ ] Fixed seed (Python, NumPy, PyTorch, CUDA) + deterministic flags
- [ ] Checkpoint + resume from `checkpoints/`
- [ ] Tracking to W&B / MLflow: git SHA + DVC data hash + Hydra config hash + all hyperparams
- [ ] Small sweep: rank ∈ {8, 16, 32}, LR ∈ {1e-4, 2e-4}, data size ∈ {1k, 5k, full}
- [ ] Ablation table → `experiments/ablation.md`
- [ ] **Exit criteria:** re-run from scratch reproduces metrics within tolerance; ablation table published

### M3 — Eval harness [todo]

- [ ] Before/after task metrics (accuracy / F1) on the frozen test set
- [ ] Regression check via `lm-evaluation-harness` (MMLU, HellaSwag) for catastrophic forgetting
- [ ] LLM-as-judge for hallucination / faithfulness, calibrated against 50–100 human labels; report Cohen's κ
- [ ] Red-team / safety set (jailbreak, PII leak, off-topic refusal)
- [ ] Bootstrap CI (1000 resamples, 95%) + multi-seed variance
- [ ] Error analysis report (top-20 failures, categorized)
- [ ] **Exit criteria:** `make eval-report` produces a report that answers: *is the new model genuinely better without breaking anything else?*

### M4 — Packaging & registry [todo]

- [ ] Merge adapter (or serve via vLLM multi-LoRA)
- [ ] Optional quantization (AWQ / GGUF) + re-eval after quantization
- [ ] Push to HF Hub / MLflow Registry with semver tag
- [ ] Model card (`docs/cards/model.md`): training data, eval results, limitations, intended use
- [ ] SHA256 checksums stored alongside artifact
- [ ] **Exit criteria:** one immutable artifact promoted to `staging` with verifiable hash

### M5 — Serving & guardrails [todo]

- [ ] vLLM (OpenAI-compatible) in Docker, pinned version
- [ ] FastAPI gateway: API key, rate limit, input validation, max tokens, timeout, streaming SSE, `/health`, `/ready`
- [ ] Input guardrail: Llama Guard + regex blocklist
- [ ] Output guardrail: PII redaction, toxicity check
- [ ] Load test (Locust / k6): p50/p95/p99, TTFT, tokens/s, cost per 1M tokens
- [ ] **Exit criteria:** SLO from M0 met under target load; `tests/load/` reproducible

### M6 — CI/CD & quality gate [todo]

- [ ] GitHub Actions PR pipeline: `ruff` → `mypy` → `pytest` → data tests → smoke-train (tiny model, 100 samples)
- [ ] **Eval gate**: PR fails if task accuracy drops > `EVAL_GATE_TOLERANCE_PTS` on golden set
- [ ] Nightly: full eval + load test + drift check
- [ ] Build image + Trivy scan + `pip-audit`
- [ ] Release tag workflow (on merge to `main` with passing gate)
- [ ] **Exit criteria:** no merge to `main` without passing the gate

### M7 — Deploy & rollback [todo]

- [ ] Staging → prod target (Cloud Run GPU / Modal / RunPod / k8s), chosen and documented
- [ ] Config via env, secrets via secret manager (no secrets in image)
- [ ] Canary or shadow traffic rollout
- [ ] Model version pinned in deployment manifest
- [ ] One-command rollback script (`registry/rollback.py`)
- [ ] **Exit criteria:** rollback tested for real; target < 5 minutes

### M8 — Observability & drift [todo]

- [ ] Structured JSON logs + correlation IDs (already in M5 code)
- [ ] Tracing via Langfuse / OpenTelemetry (opt-in, no-op default)
- [ ] Prometheus metrics: latency, error rate, tokens, GPU util, queue depth
- [ ] Grafana dashboard committed as JSON in `monitoring/dashboards/`
- [ ] Sampled prod traffic → LLM-judge → quality dashboard
- [ ] Drift detection (Evidently) on input distribution + output length + refusal rate
- [ ] Alerts: p95 > 2× SLO, error > 2%, quality drop > 5 pts/h, GPU mem > 95%
- [ ] 👍 / 👎 feedback capture endpoint
- [ ] **Exit criteria:** alert fires on simulated latency spike and quality drop

### M9 — Retraining loop [todo]

- [ ] Feedback + failures → error bank (SQLite adapter; protocol allows Postgres/Redis)
- [ ] Curation step → new dataset version (DVC-tagged)
- [ ] Retrain (M2) → auto eval (M3)
- [ ] Champion vs challenger comparison report
- [ ] Manual approval gate → promote in registry
- [ ] Full cycle reproducible via `make retrain-cycle`
- [ ] `docs/runbook/retrain.md`
- [ ] **Exit criteria:** one full cycle runs end-to-end

### M10 — Docs & production readiness review [todo]

- [ ] `SETUP.md` — one-command reproduction
- [ ] Architecture diagram (Mermaid + SVG in `docs/`)
- [ ] ADRs in `docs/adr/` (at least: model choice, serving choice, storage tier, DVC vs alternatives)
- [ ] Model card + data card finalized
- [ ] Incident runbooks: OOM, latency spike, quality drop, rollback
- [ ] Threat model mapped to OWASP LLM Top 10 (`docs/threat-model.md`)
- [ ] Cost report (`docs/cost.md`)
- [ ] `LICENSE` (MIT) + `CHANGELOG.md` finalized
- [ ] **Exit criteria:** an outsider can clone → reproduce → deploy without asking questions

### Future work (no schedule)

Ideas that came up during design but are deliberately **not** in scope for the MVP are recorded in [`docs/limitations.md`](docs/limitations.md) and [`docs/portfolio-notes.md`](docs/portfolio-notes.md) § "What I would do next". Open to discussion via issues.

Candidates (deliberately deferred):

- Multi-LoRA serving with per-tenant adapters
- gRPC serving path alongside HTTP
- Distillation from a larger teacher model
- On-device deployment (GGUF + llama.cpp) with mobile SDK
- Retrieval-augmented fine-tune (RAG + FT joint training)
- Speculative decoding for latency reduction

---

## License

MIT — see [`LICENSE`](LICENSE).
