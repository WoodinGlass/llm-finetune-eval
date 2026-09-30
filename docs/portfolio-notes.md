# Portfolio Notes

> **Status:** active. Narrative + design trade-offs + lessons learned.

## What this project demonstrates

- End-to-end LLM lifecycle (data → train → eval → serve → retrain), not a
  notebook dump.
- Production discipline: DVC, CI, tests, ADRs, SLO, limitations, cost model.
- **Honest iteration story:** v1 → v2 → v3, with failed runs published as
  evidence, not hidden. See README.md M2 section and `docs/limitations.md`.

## The one-sentence pitch

Teams need a **reproducible, testable, and observable LLM lifecycle** — from
raw data → QLoRA fine-tune → statistical eval → guarded serving → drift-aware
retraining. Most portfolio projects stop at step two; this one goes all the
way to the monitoring and feedback loop.

## What makes this different from a typical portfolio

Most "LLM fine-tune" portfolios are:

1. A Colab notebook that downloads a dataset, runs `Trainer.train()`, saves
   a model, and calls it done.
2. Or a curated success story where only the final happy number is shown.

This project ships the **pipeline and the process**, not just the payload:

- **Data pipeline** with schema validation, PII scrubbing, dedup, and
  train-vs-test decontamination (`docs/adr/0003`, `docs/cards/data.md`).
- **Statistical evaluation** with bootstrap CI, calibrated LLM-as-judge,
  and catastrophic-forgetting checks (`docs/adr/0004`).
- **Failure documentation** — v1 and v2 shipped to HF Hub as *failed
  experiments*, not deleted (`docs/adr/0007`, `docs/limitations.md`).
- **Design rationale** — every significant decision has an ADR explaining
  what was chosen, why, and what was rejected.

## The iteration story

Three training runs, each with a specific hypothesis and a specific outcome:

| Run | Change from previous | Outcome | Lesson |
|---|---|---|---|
| v1 | Baseline QLoRA — rank 16, all linear layers, LR 1e-4, 3 epochs | math −23 pts, physics −28 pts, refusal 0.96 | LoRA on attention layers overwrites reasoning |
| v2 | LR reduced to 2e-5, rank 8, 2 epochs | math −7 pts more, refusal collapsed to 0.00 | Too little capacity + too little LR = no learning at all |
| v3 | Rank-1 MLP-only + completion-only loss + RFT + ORPO + refusal ratio 3-5% | pending GPU | — (see ADR 0007) |

Each failed run is **published**, not hidden:
- v1 adapter: https://huggingface.co/WoodinGlass/qwen25-math-7b-finetuned-math-physics
- v2 adapter: https://huggingface.co/WoodinGlass/qwen25-math-7b-finetuned-math-physics-v2
- v3 adapter: pending

The lesson from v1 and v2 is not "we tried and failed". It is: **which
specific design choices cause which specific failure modes**. This is what
production ML debugging looks like.

## Design trade-offs worth discussing in an interview

- **Why FP32 on T4?** See `docs/adr/0006-fp32-training-on-t4.md`.
- **Why Qwen2.5-Math-7B?** See `docs/adr/0002-model-and-serving-choice.md`.
- **Why DVC + HF Hub + Git split?** See `docs/adr/0003-storage-and-versioning.md`.
- **Why three-layer eval?** See `docs/adr/0004-eval-methodology.md`.
- **Why rank-1 MLP-only LoRA for v3?** See `docs/adr/0007-lora-design-evolution.md`.
- **Why exact refusal string instead of varied phrasing?** See `docs/adr/0005-domain-restricted-behavior.md`.

## Lessons learned during the build

1. **Free-tier GPU quota is the #1 bottleneck**, not model size. T4 on
   Kaggle/Colab is fine for 7B QLoRA; the wall is 30 hours/month and session
   resets every 12h.
2. **Library APIs break between minor versions.** TRL renamed
   `max_seq_length` → `max_length` and `tokenizer` → `processing_class`
   between versions we hit. Pin minor versions and test upgrade paths.
3. **Reference checkpoints look like progress but can propagate stale
   assumptions.** Our v1 checkpoint was trained on pre-clean data; resuming
   v2 from it silently mixed clean/unclean data.
4. **Small rank ≠ safe rank.** Rank-1 on MLP-only preserved reasoning; rank
   8/16 on attention layers destroyed it. Location matters more than size.
5. **Over-refusal is a metric, not a bug.** v1 refusal rate at 0.96 looked
   good until we measured false-refusal at 0.26. Both must be tracked.
6. **Failed experiments are portfolio assets.** Publishing v1 and v2 as
   "failed, here's why" is more impressive than a polished success story.

## What I would do next

Deliberately out of scope for v1 (tracked here, not in the README roadmap):

- Multi-LoRA serving with per-tenant adapters.
- gRPC serving path alongside HTTP.
- Distillation from a larger teacher model.
- On-device deployment (GGUF + llama.cpp) with mobile SDK.
- Retrieval-augmented fine-tune (RAG + FT joint training).
- Speculative decoding for latency reduction.
- **PiSSA initialization** (deferred; needs 24 GB GPU — see ADR 0007).
- **DPO stage 3** if ORPO on v3 is insufficient.
- **Multilingual refusal** (currently English-only; tracked in
  `docs/limitations.md`).

## Metrics over time

Full table of every evaluated adapter (updated when v3 lands):

| Adapter | Math | Physics | OOD refusal | False refusal | Gate |
|---|---|---|---|---|---|
| baseline | 0.960 | 0.725 | 0.000 | 0.000 | — |
| v1 | 0.730 | 0.441 | 0.960 | 0.260 | FAIL |
| v2 | 0.660 | 0.471 | 0.000 | 0.000 | FAIL |
| v3 | pending | pending | pending | pending | pending |

## Full narrative

The full write-up (project brief, lessons learned, plots, and interview
answers) will live at `docs/portfolio-notes-full.md` once v3 lands. This
file is the working summary.
