# syntax=docker/dockerfile:1.7
# ═════════════════════════════════════════════════════════════════
# llm-finetune-eval — multi-stage image
#
#   Stage 1 (builder): uv resolves and installs deps into /app/.venv
#   Stage 2 (runtime): slim image, non-root user, only .venv + src copied
#
# At M0 the entrypoint is the diagnostic CLI (`python -m llm_ft info`).
# At M5 this becomes the vLLM + FastAPI gateway.
# ═════════════════════════════════════════════════════════════════

# ── Stage 1: builder ────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# uv binary (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.4.26 /uv /uvx /bin/

WORKDIR /app

# 1) copy manifests first — maximise layer cache
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./

# 2) install all extras we need at runtime
#    at M0: serve + obs are cheap placeholders; the heavy train/eval
#    extras are intentionally NOT installed into the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra serve --extra obs

# 3) copy source and install the project itself (editable off)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra serve --extra obs

# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    LLM_FT_IMAGE=1

# non-root user (uid/gid 1000, standard)
RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --create-home --shell /bin/bash app

WORKDIR /app

# copy venv + source from builder
COPY --from=builder --chown=app:app /app /app

USER app

# fail fast if the package is broken
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import llm_ft; print(llm_ft.__version__)" || exit 1

ENTRYPOINT ["python", "-m", "llm_ft"]
CMD ["info"]
