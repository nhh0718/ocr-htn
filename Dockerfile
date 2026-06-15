# syntax=docker/dockerfile:1.6
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/models \
    HOME=/root

# System libs needed by OpenCV / Paddle / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (smaller, no CUDA), then the rest of the deps.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2 \
 && pip install -r requirements.txt

# Pre-download model weights at build time so the first request is fast.
COPY app ./app
COPY scripts ./scripts
RUN python scripts/prefetch_models.py || echo "Model prefetch failed; will download at runtime."

EXPOSE 8000

# Single worker: each worker loads the full model (~1GB RAM). Increase only if
# the host has plenty of memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
