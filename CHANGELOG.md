# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — docs honesty pass

- README M2 status: `[done]` -> `[wip]` (v1/v2 failed eval gate, v3 pending)
- README M1 PII: clarify regex-only (Presidio NER deferred)
- README roadmap: note v1/v2 adapters are failed experiments, not production artifacts
- Add ADR 0006 (FP32 on T4, GradScaler/bf16 blocker)
- Add `docs/limitations.md` (honest current limits)
- Add `docs/cards/model.md` (pointer to HF Hub + honest status)
- Add stub docs: `SETUP.md`, `docs/cost.md`, `docs/threat-model.md`, `docs/portfolio-notes.md`
- Fix dead links from README (5 files)


### Added — M2 (Training)

- Hydra configs: model, data, qlora_base, smoke, full
- training/{data,model,train}.py with SFTTrainer (trl)
- QLoRA 4-bit NF4 + LoRA r=16 alpha=32
- FP32 training (fp16/bf16 disabled - T4 GradScaler bf16 bug)
- Full run: 3935 samples x 3 epochs, loss 0.245, wall 2h 08m
- Adapter v1 on HF Hub (failed eval gate — math -23 pts, physics -28 pts, false-refusal +26 pts)
  WoodinGlass/qwen25-math-7b-finetuned-math-physics
- Adapter v2 on HF Hub (failed eval gate — math -7 pts, OOD refusal regressed to 0)
  WoodinGlass/qwen25-math-7b-finetuned-math-physics-v2
- v3 training (rank-1 MLP-only + RFT + ORPO) pending GPU

### Added — M1 (Data pipeline)

- Pandera schema (`src/llm_ft/schema.py`) with domain-specific cross-checks
- MinHash LSH deduplication (threshold 0.85, 5-word shingles)
- Domain-aware PII scrub: regex (Presidio NER deferred — spacy model gap on Kaggle/Colab)
- 13-gram train-vs-test decontamination (GPT-3 / Llama style)
- DVC tracking for raw + processed data (`.dvc` pointers, Drive remote)
- Data card (`docs/cards/data.md`) with provenance, composition, biases
- 4,000 raw training samples -> 3,935 after cleaning
  - math 1800, physics 1735 (64 decontaminated), ood 320, adversarial_ood 80
- 27 unit tests for `clean.py` + 11 for `decontaminate.py`

### Added

- Initial repository skeleton (M0.1)
- MIT license, `.gitignore`, `.env.example`, `CHANGELOG.md`
- Full folder structure per architecture spec (data / training / eval / serving / deploy / monitoring / registry / tests / docs / .github)
- `README.md` describing problem, architecture, SLO, roadmap M0–M10

### Notes

- CI badge in README will resolve after M6 (first workflow added).
