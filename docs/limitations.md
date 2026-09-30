# Known Limitations

Honest, current-as-of-writing list of what this project does **not** yet do, and why.

## Data

- **Presidio NER not run.** PII scrub uses domain-aware **regex only**.
  `presidio-analyzer` requires a `spacy` model (`en_core_web_lg`) that is not
  installable in the Kaggle/Colab environment used for training. Impact:
  context-dependent PII (names, addresses) may survive scrubbing. Mitigation
  planned in a later data pass.
- **Curated OOD set.** Adversarial-OOD (80 rows) is hand-written + template-composed,
  not scraped from real jailbreak corpora. Coverage of real-world adversarial
  prompts is therefore limited.

## Training

- **Hybrid checkpoint (v1).** Steps 0–400 of v1 used **pre-clean** data;
  steps 400–738 used cleaned data. Physics eval on v1 may be inflated by a
  few points. A fresh-from-zero retrain (v3) resolves this.
- **FP32-only on T4.** fp16 / bf16 training is blocked by a torch GradScaler
  kernel gap on Turing GPUs. ~2× slower than fp16. See ADR 0006.
- **v1 & v2 failed the eval gate.** Both adapters are on HF Hub as
  *failed experiments* (iteration evidence), not as production artifacts.
  See README M2 section.
- **Sweep deferred.** Hyperparameter sweep (rank, LR, data size) is deferred
  due to FP32 time cost. Single-config runs only so far.

## Model behaviour

- **English-only refusal.** The refusal string is English regardless of input
  language. Non-English prompts get the English refusal.
- **Single-turn only.** No multi-turn conversation support.
- **No tool use / function calling.**
- **No multimodal support.**

## Evaluation

- **No judge calibration shipped with the adapter.** Hallucination rate via
  LLM-as-judge (with human calibration, Cohen's κ) lives in the parent repo
  (M3) and is not yet applied to the released adapters.
- **Frozen test set only.** No held-out "unseen benchmark" evaluation yet.

## Reproducibility

- **GPU quota dependency.** Training requires Colab or Kaggle GPU; both have
  monthly quotas that can block reproduction on a fresh account.
