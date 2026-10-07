# syntax=docker/dockerfile:1
# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY . /app
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12 --frozen --no-dev
RUN FORCE_SYNC=0 PATCH_PYTHON=/opt/venv/bin/python bash scripts/install_transformers_patch.sh
ENV PATH="/opt/venv/bin:${PATH}"
CMD ["/bin/bash", "-c", "python scripts/serve_policy.py $SERVER_ARGS"]
