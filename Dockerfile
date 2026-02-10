# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.12
ARG TARGETPLATFORM=linux/amd64

FROM --platform=$TARGETPLATFORM python:${PYTHON_VERSION}-slim-bookworm

# Metadata labels
LABEL org.opencontainers.image.title="AudibleLight Dataset Generator"
LABEL org.opencontainers.image.description="Framework for generating synthetic acoustic datasets using AudibleLight"
LABEL org.opencontainers.image.authors="Philipp Schmidt"
LABEL org.opencontainers.image.url="https://github.com/PhilippXXY/audiblelight-dataset-generator"
LABEL org.opencontainers.image.source="https://github.com/PhilippXXY/audiblelight-dataset-generator"
LABEL org.opencontainers.image.documentation="https://github.com/PhilippXXY/audiblelight-dataset-generator/blob/main/README.md"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgeos-dev \
    libgeos-c1v5 \
    proj-bin \
    proj-data \
    libproj-dev \
    gcc \
    g++ \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (avoid GHCR auth issues)
RUN python -m pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and configuration
COPY src/ ./src/
COPY config/config.yaml ./config/config.yaml

# Copy local data into the image (foreground/audio/meshes if present)
COPY data/ ./data/

# Set entrypoint to run generator
ENTRYPOINT ["uv", "run", "--no-dev", "python", "src/generator.py"]
CMD ["--config", "/app/config/config.yaml"]
