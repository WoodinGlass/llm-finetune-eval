# Cost Report

> **Status:** stub - populated in **M5** (serving) with real numbers from load tests.

## What will be tracked

| Component | Cost driver | Populated in |
|---|---|---|
| GPU (serving) | $/hr x uptime | M5 |
| GPU (training) | $/hr x hours | M2 (partial) |
| Storage | $/GB-month | M1 |
| Egress | $/GB | M4 |
| Observability | $/event | M8 |

## Current training cost estimate

- QLoRA 8B fine-tune (v1): ~2h 08m on T4, ~$0 (Colab/Kaggle free tier)
- Full v3 run (projected): ~4-6h on T4/L4

Real numbers for serving (cost per 1k requests, cost per 1M tokens) will land
in M5. See docs/slo.md for target cost thresholds.
