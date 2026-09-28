# ADR 0001 — End-to-end LLM lifecycle as a single repo

- **Status:** accepted
- **Date:** M0.4
- **Deciders:** WoodinGlass
- **Supersedes:** —

## Context

Fine-tuning an LLM and shipping it to production involves at least six
distinct concerns: (1) data preparation and versioning, (2) training,
(3) evaluation, (4) packaging & registry, (5) serving with guardrails,
(6) observability + retraining. A common failure mode for portfolio
projects is to leave (4)–(6) as slides, not code, and to treat (1) as
"whatever the notebook downloaded".

We want a **single repository** that demonstrates the full lifecycle,
runnable on free / cheap infrastructure, with no hand-waved steps.

## Decision

We implement the lifecycle as **one monorepo** with these hard rules:

1. **Payload vs pipeline separation.** The *payload* is the fine-tuned
   adapter + eval harness. Everything else (DVC, CI, registry, observability)
   is *pipeline* that makes the payload safe to ship.
2. **One entrypoint per stage.** `make train`, `make eval`, `make serve`,
   `make deploy`. No "run this notebook then that notebook".
3. **Reproducibility triple.** Every artifact is pinned by
   `(git SHA, DVC data hash, Hydra config hash)`. The triple is logged to
   W&B/MLflow and stored in the model card.
4. **Fail-loud placeholders.** Milestones that aren't implemented yet ship
   as `make` targets that exit non-zero with a message, never silent no-ops.
5. **Portable across Colab / Kaggle / local.** All environment differences
   are isolated in `src/llm_ft/env.py` and `src/llm_ft/secrets.py`; the rest
   of the codebase is platform-agnostic.

## Consequences

### Positive

- Reviewers can trace **one** data → train → eval → serve path.
- CI can run the same eval gate locally with a single `make` target.
- Adding a new stage (e.g. distillation) means adding one `make` target and
  one folder — not a new repo.

### Negative / risks

- Repo grows large; CI must be tiered (unit / data / smoke / slow).
- Heavy deps (torch, vllm) are pulled via extras, not base install, to keep
  `make setup` fast for docs-only contributors.
- Kaggle / Colab have different default Python versions; we standardize on
  Python 3.11 via `uv python install 3.11` and `.venv`.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Separate repos per stage (data / train / serve) | Increases friction, kills end-to-end traceability |
| One notebook + external git for scripts | No test tiers, no CI gate, no reproducibility triple |
| Use a managed platform (Vertex AI, SageMaker) | Hides the mechanics we want to demonstrate; cost > $0 |
| Lightning / HF AutoTrain as the only path | Hides eval + serving, which are the hard parts |

## References

- `README.md § Architecture`
- `docs/slo.md` (contract this architecture must satisfy)
- `.github/workflows/` (M6 — enforcement)
