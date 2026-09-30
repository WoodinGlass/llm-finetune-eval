# Portfolio Notes

> **Status:** stub - narrative + design trade-offs.

## What this project demonstrates

- End-to-end LLM lifecycle (data -> train -> eval -> serve -> retrain), not a notebook dump.
- Production discipline: DVC, CI, tests, ADRs, SLO, limitations, cost model.
- **Honest iteration story:** v1 -> v2 -> v3, with failed runs published as
  evidence, not hidden. See README.md M2 section and docs/limitations.md.

## What I would do next

Deliberately out of scope for v1 (tracked here, not in the README roadmap):

- Multi-LoRA serving with per-tenant adapters.
- gRPC serving path alongside HTTP.
- Distillation from a larger teacher model.
- On-device deployment (GGUF + llama.cpp) with mobile SDK.
- Retrieval-augmented fine-tune (RAG + FT joint training).
- Speculative decoding for latency reduction.

## Design trade-offs worth discussing in an interview

- Why FP32 on T4? See docs/adr/0006-fp32-training-on-t4.md.
- Why Qwen2.5-Math-7B? See docs/adr/0002-model-and-serving-choice.md.
- Why DVC + HF Hub + Git split? See docs/adr/0003-storage-and-versioning.md.
- Why three-layer eval? See docs/adr/0004-eval-methodology.md.

Full narrative (project brief, lessons learned, metrics-over-time plots) will
be added in M10.
