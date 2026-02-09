FROM python:3.12.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgeos-dev \
    proj-bin \
    proj-data \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
