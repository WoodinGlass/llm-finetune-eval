# Service Level Objectives (SLO)

> Status: **baseline TBD** — diisi setelah `make eval-baseline` selesai (M0 exit criteria).

## Task
Fine-tune Qwen2.5-7B-Instruct untuk **math + physics reasoning**, bilingual **Indonesia + English**.

## Target

| Metric | Baseline (few-shot) | Target | Gate (M6) |
|---|---|---|---|
| GSM8K accuracy (EN) | TBD | ≥ 85% | ≥ baseline − 1pt |
| GSM8K accuracy (ID) | TBD | ≥ baseline + 5pt | ≥ baseline − 1pt |
| MATH accuracy (EN) | TBD | ≥ 50% | ≥ baseline − 1pt |
| MATH accuracy (ID) | TBD | ≥ baseline + 3pt | ≥ baseline − 1pt |
| MMLU HS Physics (EN) | TBD | ≥ 78% | ≥ baseline − 2pt |
| MMLU College Physics (EN) | TBD | ≥ 60% | ≥ baseline − 2pt |
| IndoMMLU Fisika (ID) | TBD | ≥ baseline + 5pt | ≥ baseline − 2pt |
| GPQA Physics (EN) | TBD | ≥ baseline + 3pt | ≥ baseline − 3pt |
| Answer parse rate | — | ≥ 98% | hard gate |
| MMLU general delta (forgetting) | — | ≥ −2pt | hard gate |
| p95 latency | — | ≤ 1.5 s | SLO M5 |
| Cost per 1k requests | — | ≤ $0.10 | SLO M5 |

## Revisi
- 2026-09-28 — draft awal (M0), baseline TBD
