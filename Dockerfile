# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Root Dockerfile for Hugging Face Spaces (Docker SDK).
# Mirrors server/Dockerfile so Spaces can build from repository root.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure required build tools are available.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo.
ARG BUILD_MODE=in-repo
ARG ENV_NAME=phylo

# Copy full environment code from repository root.
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it).
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install both Python runtimes:
# - 3.12: main OpenEnv/runtime stack
# - 3.11: Blender bridge runtime (bpy-compatible)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 3.11

# Pin the project virtualenv to Python 3.12 for the primary runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 .venv

# Install dependencies using uv sync into .venv (Python 3.12).
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Dedicated Python 3.11 bridge environment for bpy scenes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.11 .venv_311 && \
    uv pip install --python .venv_311/bin/python \
        bpy \
        numpy \
        pyyaml \
        hydra-core \
        omegaconf \
        mujoco \
        ipdb

FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual environments and code.
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env/.venv_311 /app/env/.venv_311
COPY --from=builder /app/env /app/env
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

# Runtime environment.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PHYLO_BRIDGE_PYTHON="/app/env/.venv_311/bin/python"
ENV MUJOCO_GL="egl"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Spaces expects a web server listening on the configured app_port (8000).
# Use the venv Python explicitly to avoid falling back to system uvicorn.
CMD ["sh", "-c", "cd /app/env && /app/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"]
