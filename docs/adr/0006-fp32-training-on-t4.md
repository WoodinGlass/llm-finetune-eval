# ADR 0006 — FP32 training on T4 (fp16/bf16 disabled)

- **Status:** Accepted
- **Date:** 2026-09-30
- **Deciders:** project owner

## Context

QLoRA fine-tune of `Qwen/Qwen2.5-Math-7B-Instruct` on a Colab / Kaggle **T4** GPU.

Observed constraints:

- **bf16 — hardware.** T4 (Turing, SM 7.5) does not support bf16 natively.
- **bf16 adapters — software.** Even when forced, `torch` GradScaler hits a kernel gap
  (`RuntimeError: expected scalar type BFloat16 but found Float`) when 4-bit NF4 base
  meets bf16 LoRA adapter tensors on T4. Unresolved in `bitsandbytes`/`torch` at time of writing.
- **fp16 — instability.** With 4-bit NF4 + `paged_adamw_32bit`, fp16 runs show loss
  spikes and occasional NaN gradients at lr ≥ 1e-4.
- **FP32 — stable.** ~2× slower per step than fp16, but reproducible across sessions.

## Decision

Train in **FP32** for v1 / v2 / v3 on T4.

Accepted consequences:

- Training time roughly doubled (~4–8 h → ~8–16 h per full run).
- No GradScaler instability, no NaN loss.
- Reproducible across T4 sessions.

## Alternatives considered

| Option | Why rejected |
|---|---|
| fp16 + bf16 adapters | GradScaler kernel gap on T4 (torch bug, unresolved) |
| bf16 everything | T4 does not support bf16 |
| fp16 + fp32 adapters | OOM on T4 (15 GB) at seq_len 2048 with r=16 |
| Upgrade to L4 / A10G / A100 | Cost + availability (Kaggle/Colab quota, region blocks) |

## Revisit when

- GPU changes to L4 / A10G / A100 → switch to **bf16**.
- `torch` fixes the T4 GradScaler bug → retry **fp16**.
- 24 GB GPU available → PiSSA init + seq_len 4096 become feasible.
