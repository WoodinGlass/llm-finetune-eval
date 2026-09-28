# ADR 0002 — Qwen2.5-Math-7B as base model, vLLM as serving engine

- **Status:** accepted
- **Date:** M0.4
- **Deciders:** WoodinGlass
- **Supersedes:** —

## Context

The target task is **competition-style math and physics problem solving**.
We need:

- A base model strong enough that few-shot baseline is non-trivial.
- A model small enough to QLoRA on a free GPU (T4 16 GB / P100 16 GB).
- An open-weight license that permits redistribution of the adapter.
- A serving path with an OpenAI-compatible API, streaming, and continuous
  batching, so SLO numbers (§3.2 of `docs/slo.md`) are achievable on a single
  A10G.

## Decision

### Base model

**`Qwen2.5-Math-7B-Instruct`** as the base for both baseline and fine-tune.

Rationale:

- Qwen2.5-Math is explicitly trained on chain-of-thought math data; few-shot
  baseline on MATH / GSM8K is meaningfully above general-purpose 7B models.
- 7B parameters fit QLoRA on T4/P100 (4-bit → ≈ 5.4 GB weights, leaving
  ≈ 10 GB for activations, optimizer states, and grad checkpointing).
- Apache-2.0 license; adapter can be redistributed.
- Native 32k context, 128k with YaRN — headroom for future long-derivation use.

### Serving engine

**vLLM** (OpenAI-compatible server) behind a FastAPI gateway.

Rationale:

- Continuous batching + PagedAttention give the throughput needed for §3.2.
- OpenAI-compatible API means the FastAPI gateway is thin (auth, rate limit,
  guardrails) rather than reimplementing generation.
- Multi-LoRA serving lets us host base + one or more adapters on one GPU —
  useful for champion/challenger in M9.
- Battle-tested at time of writing; alternative (TGI) has less LoRA flexibility.

## Consequences

### Positive

- One adapter fits both our free-tier training and our cheap serving target.
- Gateway stays framework-agnostic; swapping vLLM → TGI is a 1-file change.
- Math-heavy base reduces the amount of fine-tuning needed to hit §3.1 targets.

### Negative / risks

- 7B is at the edge of free-tier VRAM; sweep must stay small (rank ∈ {8,16,32}).
- Qwen2.5-Math is weaker on general chat; the eval harness *must* include
  MMLU / HellaSwag to catch catastrophic forgetting.
- Physics is out-of-domain for Qwen2.5-Math; we may need more physics data
  than math data to hit §3.1 physics target.

## Alternatives considered

| Model | Why rejected |
|---|---|
| Llama-3.1-8B-Instruct | Larger (8B) → tighter VRAM budget; weaker math prior |
| Qwen2.5-Coder-7B | Wrong domain (code, not math) |
| Qwen2.5-1.5B-Instruct | Too small; baseline too weak to show lift |
| Mistral-7B-Instruct | Older; weaker math benchmark at same size |
| DeepSeek-Math-7B | Strong math, but license more restrictive than Apache-2.0 |

| Engine | Why rejected |
|---|---|
| HF `text-generation-inference` | Fewer LoRA management options |
| TensorRT-LLM | CUDA-version and hardware lock-in, harder to demo on free-tier |
| `llama.cpp` server | Great for CPU/GGUF, not for GPU-throughput SLO |
| Custom PyTorch loop | Would reimplement batching we get free from vLLM |

## References

- `docs/slo.md` §3.1 (quality targets)
- `training/` (M2 — QLoRA config)
- `serving/` (M5 — vLLM + gateway)
