# Step 10: Logging & Metrics

## Why Observability?

You can't improve what you can't measure.

## Structured Logging

```python
import structlog

logger = structlog.get_logger()

def log_request(request_id: str, model: str, tokens: int, latency: float):
    logger.info(
        "llm_request",
        request_id=request_id,
        model=model,
        input_tokens=tokens,
        latency_ms=latency * 1000,
        status="success"
    )
```

## Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'direction']  # input/output
)

# Histograms
latency_histogram = Histogram(
    'llm_request_latency_seconds',
    'Request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Gauges
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization'
)

# Usage
requests_total.labels(model="llama-2", status="success").inc()
tokens_total.labels(model="llama-2", direction="input").inc(150)
latency_histogram.labels(model="llama-2").observe(0.5)
```

## Key Metrics to Track

```
Request metrics:
- requests_total
- request_latency_seconds
- error_rate

Token metrics:
- input_tokens_total
- output_tokens_total
- tokens_per_second

Resource metrics:
- gpu_utilization
- gpu_memory_used
- kv_cache_utilization

Cost metrics:
- estimated_cost_dollars
```

## FastAPI Integration

```python
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
import time

app = FastAPI()

# Add prometheus endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    
    latency_histogram.labels(
        endpoint=request.url.path
    ).observe(latency)
    
    return response
```

## Files

- `logging_metrics.py` - Complete observability setup
