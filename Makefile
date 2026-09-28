# ═════════════════════════════════════════════════════════════════
# llm-finetune-eval — developer entrypoints
# Every target is safe to run on a clean machine (`make setup` first).
# ═════════════════════════════════════════════════════════════════

SHELL          := /bin/bash
.DEFAULT_GOAL  := help

# ── interpreter selection ───────────────────────────────────────
PY311          := python3.11
VENV           := .venv
PY             := $(VENV)/bin/python
PIP            := $(VENV)/bin/pip
UV             := uv

# ── project metadata (single source of truth: pyproject.toml) ──
NAME           := $(shell grep -m1 '^name' pyproject.toml | cut -d'"' -f2)
VERSION        := $(shell grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
IMAGE          := $(NAME):$(VERSION)
IMAGE_LATEST   := $(NAME):latest

# ── pytest tiers ───────────────────────────────────────────────
PYTEST_UNIT    := -m unit
PYTEST_DATA    := -m data
PYTEST_SMOKE   := -m smoke
PYTEST_ALL     := -m "unit or data or smoke"

# ═════════════════════════════════════════════════════════════════
.PHONY: help
help:  ## show this help
	@printf "\n$(NAME) $(VERSION)\n\n"
	@printf "Targets:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf "\n"

# ── setup ───────────────────────────────────────────────────────
.PHONY: setup
setup: ensure-python ## install dev deps into .venv (uv sync) + git hooks
	$(UV) sync --extra dev
	$(VENV)/bin/pre-commit install
	$(VENV)/bin/pre-commit install --hook-type pre-push
	@printf "\n✓ setup done.  activate:  source $(VENV)/bin/activate\n"

.PHONY: setup-colab
setup-colab: ensure-python ## setup + expose python3.11 to PATH (Colab)
	$(UV) sync --extra dev
	$(VENV)/bin/pre-commit install
	$(VENV)/bin/pre-commit install --hook-type pre-push
	@printf "\n✓ setup-colab done.  Run tools via: $(PY) -m <tool>\n"

.PHONY: ensure-python
ensure-python:  ## ensure python3.11 available (uv-managed)
	@command -v $(PY311) >/dev/null 2>&1 || { \
	  echo "→ installing $(PY311) via uv ..."; \
	  $(UV) python install 3.11; \
	  PY311_PATH=$$($(UV) python find 3.11); \
	  ln -sf $$PY311_PATH /usr/local/bin/$(PY311); \
	}
	@$(PY311) --version

# ── quality ─────────────────────────────────────────────────────
.PHONY: lint
lint:  ## ruff check + mypy (src/)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy --config-file pyproject.toml src/

.PHONY: format
format:  ## ruff format + auto-fix
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

.PHONY: pre-commit
pre-commit:  ## run pre-commit on all files
	$(VENV)/bin/pre-commit run --all-files

# ── tests ───────────────────────────────────────────────────────
.PHONY: test
test: test-unit  ## default: fast unit tests

.PHONY: test-unit
test-unit:  ## fast tests, no external deps
	$(PY) -m pytest $(PYTEST_UNIT)

.PHONY: test-data
test-data:  ## data validation + schema tests
	$(PY) -m pytest $(PYTEST_DATA)

.PHONY: test-smoke
test-smoke:  ## tiny end-to-end train+serve
	$(PY) -m pytest $(PYTEST_SMOKE)

.PHONY: test-all
test-all:  ## unit + data + smoke
	$(PY) -m pytest $(PYTEST_ALL)

# ── diagnostics ─────────────────────────────────────────────────
.PHONY: info
info:  ## print platform / paths (JSON)
	$(PY) -m llm_ft info

.PHONY: version
version:  ## print package version
	@$(PY) -m llm_ft version

# ── data freeze / verify ────────────────────────────────────────
.PHONY: freeze-test-set
freeze-test-set:  ## freeze data/raw/* -> data/processed/test/ + lock
	$(PY) -m llm_ft.freeze freeze

.PHONY: verify-test-set
verify-test-set:  ## verify data/processed/test/ against test.lock
	$(PY) -m llm_ft.freeze verify

# ── docker ──────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:  ## build the multi-stage image
	docker build -t $(IMAGE) -t $(IMAGE_LATEST) .

.PHONY: docker-run
docker-run:  ## run the image (diagnostic entrypoint)
	docker run --rm $(IMAGE) info

# ── housekeeping ────────────────────────────────────────────────
.PHONY: clean
clean:  ## remove caches & build artifacts (keeps .venv)
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	@printf "✓ cleaned (kept .venv)\n"

.PHONY: distclean
distclean: clean  ## clean + remove .venv and uv.lock
	@rm -rf $(VENV) uv.lock
	@printf "✓ distclean done\n"

# ── placeholders (filled in later milestones) ───────────────────
.PHONY: eval-baseline
eval-baseline:  ## M0.5 — evaluate base model + few-shot prompt
	@printf "not yet implemented — arrives in M0.5\n"; exit 1

.PHONY: data-pull
data-pull:  ## M1 — dvc pull
	@printf "not yet implemented — arrives in M1\n"; exit 1

.PHONY: train-smoke
train-smoke:  ## M2 — tiny QLoRA run on 100 samples
	@printf "not yet implemented — arrives in M2\n"; exit 1

.PHONY: serve
serve:  ## M5 — run vLLM + gateway locally
	@printf "not yet implemented — arrives in M5\n"; exit 1

.PHONY: pipeline
pipeline:  ## end-to-end (data → train → eval → serve)
	@printf "not yet implemented — arrives in M6\n"; exit 1
