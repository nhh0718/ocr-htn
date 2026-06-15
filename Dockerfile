# syntax=docker/dockerfile:1.6
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Hugging Face Spaces runs the container as a non-root user (uid 1000). We
# create the same user here so the image works on both HF Spaces and Render.
ARG USERNAME=appuser
ARG UID=1000
RUN useradd -m -u ${UID} ${USERNAME}

WORKDIR /app

# Install CPU-only torch first (smaller, no CUDA), then the rest of the deps.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2 \
 && pip install -r requirements.txt

# Copy source and ensure all paths are owned by the non-root user.
COPY --chown=${USERNAME}:${USERNAME} app ./app
COPY --chown=${USERNAME}:${USERNAME} scripts ./scripts
RUN mkdir -p /app/models /home/${USERNAME}/.cache /home/${USERNAME}/.paddleocr \
 && chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}

USER ${USERNAME}

# Cache dirs must be writable by the runtime user.
ENV HOME=/home/${USERNAME} \
    MODEL_DIR=/home/${USERNAME}/models \
    XDG_CACHE_HOME=/home/${USERNAME}/.cache \
    HF_HOME=/home/${USERNAME}/.cache/huggingface

# Pre-download model weights at build time so the first request is fast.
RUN python scripts/prefetch_models.py || echo "Model prefetch failed; will download at runtime."

EXPOSE 7860

# HF Spaces expects port 7860; Render injects $PORT. Default to 7860 so HF
# works out of the box; Render's $PORT (typically 10000) overrides it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
