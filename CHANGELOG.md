# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — M2 (Training)

- Hydra configs: model, data, qlora_base, smoke, full
- training/{data,model,train}.py with SFTTrainer (trl)
- QLoRA 4-bit NF4 + LoRA r=16 alpha=32
- FP32 training (fp16/bf16 disabled - T4 GradScaler bf16 bug)
- Full run: 3935 samples x 3 epochs, loss 0.245, wall 2h 08m
- Adapter on HF Hub: WoodinGlass/qwen25-math-7b-finetuned-math-physics

### Added — M1 (Data pipeline)

- Pandera schema (`src/llm_ft/schema.py`) with domain-specific cross-checks
- MinHash LSH deduplication (threshold 0.85, 5-word shingles)
- Domain-aware PII scrub: Presidio NER + regex; math/physics preserved
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
