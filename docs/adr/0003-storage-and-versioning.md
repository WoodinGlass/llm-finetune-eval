# ADR 0003 — DVC for data, Git for code, HF Hub for models

- **Status:** accepted
- **Date:** M0.4
- **Deciders:** WoodinGlass
- **Supersedes:** —

## Context

The project has three kinds of versioned artifacts:

1. **Code** — small, text, diffable, reviewed via PR.
2. **Data** — medium (hundreds of MB), immutable in principle, must be
   reproducible from a manifest, must support decontamination checks.
3. **Model weights** — large (GB), immutable, must be verifiable by hash,
   must support promote/rollback.

A single storage strategy would be wasteful:

- Putting data in Git bloats clones forever.
- Putting code in DVC breaks review workflows.
- Putting model weights in Git is impossible; putting them only on local disk
  breaks reproducibility.

## Decision

Split by artifact type:

| Artifact | Storage | Reason |
|---|---|---|
| Code | **Git** (GitHub) | Review, diff, blame, CI trigger |
| Configs | **Git** (Hydra YAML) | Small, reviewable; hash recorded per run |
| Raw / processed data | **DVC** remote (S3-compatible) | Content-addressed, versioned, deduped |
| Data manifest | **Git** (`dvc.lock`) | Small; pins exact data hashes |
| Experiment metrics | **W&B** (primary), **MLflow** (local backup) | Time-series, comparable runs |
| Fine-tuned adapters | **HF Hub** (private → public) + **local Drive mirror** | Immutable, hashable, easy to pull on any platform |
| Baseline scores | **Git** (`docs/slo.md`) | Human-readable, part of contract |

Rules:

1. **DVC hash is truth for data.** Any code path that reads a dataset must
   read from the path recorded in `dvc.lock`, not a hard-coded URL.
2. **HF Hub is truth for adapters.** The `registry/` scripts (M4) treat
   HF Hub as the source of record; local Drive is a cache.
3. **No large file goes into Git.** `.gitignore` already excludes
   `data/raw/*`, `data/processed/*`, `checkpoints/`, `*.safetensors`, `*.bin`.
4. **Test set is frozen by hash.** `data/processed/test.lock` records a
   SHA256 over the exact file content; CI fails if it changes.

## Consequences

### Positive

- Clones stay fast; `git log` stays readable.
- Data can be re-downloaded from DVC remote on any machine with one command.
- Adapters can be pulled on Colab / Kaggle / prod without Drive symlink games.
- Rollback (§M7) is a one-liner: point the deployment at an older HF revision.

### Negative / risks

- DVC remote must be configured (S3 / GDrive / MinIO). Until then, data
  lives only on Drive — acceptable for M1 but must be closed before M6.
- Two secrets systems to manage (DVC remote creds + HF token). Both go
  through `src/llm_ft/secrets.py`.
- HF Hub rate limits could throttle CI; mitigate with a local cache in CI.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Git LFS only | Poor diff/summary for Parquet; no pipeline DAG |
| Plain S3 + manual manifest | No dedup, no pipeline cache, easy to drift |
| All-in on Kaggle Datasets | Ties us to Kaggle; Drive already needed for checkpoints |
| Weights & Biases Artifacts only | Great for models, awkward for raw text/parquet |
| Zenodo / OSF | Designed for publication, not for CI pulls |

## References

- `dvc.yaml` (M1)
- `docs/cards/data.md` (M1)
- `docs/cards/model.md` (M4)
- `registry/` (M4, M7)
- `src/llm_ft/secrets.py` (unified secret access)
