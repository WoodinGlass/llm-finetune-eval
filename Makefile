SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── meta ─────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── setup ────────────────────────────────────
.PHONY: setup
setup: ## Install uv, sync env, install pre-commit hooks
	@command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run pre-commit install
	@test -f .env || cp .env.example .env
	@echo "✅ setup done. Edit .env then run: make eval-baseline"

# ── quality ──────────────────────────────────
.PHONY: lint
lint: ## ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy training eval serving

.PHONY: format
format: ## Auto-fix formatting
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: test
test: ## Unit tests only (fast)
	uv run pytest -m unit -v

.PHONY: test-all
test-all: ## All tests
	uv run pytest -v

# ── eval ─────────────────────────────────────
.PHONY: eval-baseline
eval-baseline: ## Baseline few-shot eval (EN + ID). SLOW.
	uv run python eval/baseline_fewshot.py

.PHONY: eval-report
eval-report: ## Full eval report (M3)
	uv run python eval/report.py

# ── training ─────────────────────────────────
.PHONY: train-smoke
train-smoke: ## Tiny QLoRA run: 100 samples, 1 step (CI gate)
	uv run python training/train.py --config-name=smoke

.PHONY: train
train: ## Full QLoRA training
	uv run python training/train.py

# ── pipeline ─────────────────────────────────
.PHONY: pipeline
pipeline: ## Full reproducible pipeline (M9)
	uv run dvc repro
