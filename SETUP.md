# SETUP

> **Status:** stub - full content lands in **M10** (production readiness).

One-command reproduction target:

    git clone https://github.com/WoodinGlass/llm-finetune-eval.git
    cd llm-finetune-eval
    make setup      # uv sync + pre-commit install
    make data-pull  # DVC pull from Drive remote

Full instructions (env vars, secrets, GPU requirements) will be documented here
as part of M10. Until then, the **Quickstart** section in README.md is the
working reference.
