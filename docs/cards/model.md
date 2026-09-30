# Model Card - Qwen2.5-Math-7B-Instruct (math + physics QLoRA adapter)

> **Status:** pointer card. Canonical card lives on HF Hub.

## Canonical location

- **v1 (failed experiment):** https://huggingface.co/WoodinGlass/qwen25-math-7b-finetuned-math-physics
- **v2 (failed experiment):** https://huggingface.co/WoodinGlass/qwen25-math-7b-finetuned-math-physics-v2
- **v3 (pending GPU):** not yet released

## Honest status

**No adapter has passed the eval gate yet.** v1 and v2 are published as
*failed experiments* - evidence of the iteration story, not production
artifacts. v3 (rank-1 MLP-only + RFT + ORPO) is the next attempt.

## Eval summary

| Metric | Baseline | Target (min) | v1 | v2 | v3 |
|---|---|---|---|---|---|
| Math accuracy | 0.960 | >= 0.960 | 0.730 FAIL | 0.660 FAIL | pending |
| Physics accuracy | 0.725 | >= 0.800 | 0.441 FAIL | 0.471 FAIL | pending |
| OOD refusal rate | 0.000 | >= 0.950 | 0.960 OK | 0.000 FAIL | pending |
| False refusal rate | 0.000 | <= 0.020 | 0.260 FAIL | 0.000 OK | pending |

Detailed numbers, evaluation methodology, and limitations are in the parent
repo: docs/slo.md, docs/limitations.md, and the HF Hub model cards linked above.

## Intended use

Math + physics Q&A, single-turn, English-only. Out-of-domain requests are
refused with a fixed string. See limitations for what this model is **not**
for.
