# Architecture Decision Records

Each ADR is immutable once accepted; superseding ADRs link to the old one.

| # | Title | Status | Date |
|---|---|---|---|
| 0001 | [End-to-end LLM lifecycle as a single repo](0001-architecture-overview.md) | accepted | M0.4 |
| 0002 | [Qwen2.5-Math-7B as base model, vLLM as serving engine](0002-model-and-serving-choice.md) | accepted | M0.4 |
| 0003 | [DVC for data, Git for code, HF Hub for models](0003-storage-and-versioning.md) | accepted | M0.4 |
| 0004 | [Three-layer eval with calibrated LLM-as-judge](0004-eval-methodology.md) | accepted | M0.4 |

## Format

Each ADR follows Michael Nygard's format:

- **Title** — short imperative statement
- **Status** — proposed / accepted / superseded by NNNN
- **Context** — what forces are at play
- **Decision** — what we chose
- **Consequences** — what becomes easier / harder
- **Alternatives considered** — with explicit rejection reasons
