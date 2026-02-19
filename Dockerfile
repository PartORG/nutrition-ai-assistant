FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install production dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Create directories that may be mounted as volumes at runtime
RUN mkdir -p /app/data /app/data_test /app/vector_databases /app/db

WORKDIR /app/src

CMD ["uvicorn", "adapters.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]
