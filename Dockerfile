# Uses the official uv image (uv + Python 3.11 preinstalled).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# System dependencies required to build some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer) — project code excluded for cache reuse
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source, then install the project itself (editable)
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Put the venv on PATH so installed entry points resolve
ENV PATH="/app/.venv/bin:$PATH"

# Create directories that may be mounted as volumes at runtime
RUN mkdir -p /app/data /app/data_test /app/vector_databases /app/db

CMD ["uvicorn", "adapters.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]
