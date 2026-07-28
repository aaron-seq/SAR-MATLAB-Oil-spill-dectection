# syntax=docker/dockerfile:1
#
# Multi-stage build for the SAR oil spill detection API.
#
#   docker build --target production -t sar-oil-spill .
#   docker build --target development -t sar-oil-spill:dev .

# ----------------------------------------------------------------- builder

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tools live only in this stage; the runtime image never sees them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies resolve from the manifest alone, so this layer is cached until
# pyproject.toml actually changes -- not on every source edit.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install '.[api]'

# --------------------------------------------------------------------- base

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# opencv-python-headless still needs libGL and glib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY app.py pyproject.toml README.md ./

RUN mkdir -p results logs models/saved_models

# Run unprivileged: the service only ever reads uploads and writes to results/.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --------------------------------------------------------------- production

FROM base AS production

ENV ENVIRONMENT=production

# Two workers: the pipeline is CPU-bound and already off-threads its heavy work,
# so oversubscribing cores mostly adds memory pressure. Raise with -w on hosts
# with more vCPUs.
CMD ["gunicorn", "api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]

# -------------------------------------------------------------- development

FROM base AS development

ENV ENVIRONMENT=development

USER root
COPY tests/ ./tests/
COPY scripts/ ./scripts/
RUN pip install '.[api,dev]' && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
