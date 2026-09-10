# Step 7: Containerization

## Why Containers?

- Consistent environments
- Easy deployment
- Reproducible builds

## Basic Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## GPU-Enabled Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install vLLM
RUN pip install vllm

# Copy model or download at runtime
ENV MODEL_NAME="meta-llama/Llama-2-7b-hf"

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "$MODEL_NAME", "--host", "0.0.0.0"]
```

## Multi-Stage Build

```dockerfile
# Build stage
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

## Docker Compose

```yaml
version: '3.8'
services:
  llm-api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=llama-2-7b
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Files

- `Dockerfile` - Production Dockerfile
- `docker-compose.yml` - Local development setup
