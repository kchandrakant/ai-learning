# MLOps & Model Serving Demos

## Available Demos

### 1. Basic API Server (`api_demo.py`)
OpenAI-compatible API with FastAPI.

```bash
python demo/api_demo.py
# Visit http://localhost:8000/docs
```

### 2. Streaming Demo (`streaming_demo.py`)
Server-Sent Events streaming.

```bash
python demo/streaming_demo.py
```

### 3. vLLM Server (`vllm_demo.py`)
High-throughput serving with vLLM.

```bash
python demo/vllm_demo.py --model meta-llama/Llama-2-7b-hf
```

### 4. Load Testing (`load_test.py`)
Benchmark your deployment.

```bash
locust -f demo/load_test.py --host http://localhost:8000
```

### 5. Metrics Dashboard (`metrics_demo.py`)
Prometheus metrics collection.

```bash
python demo/metrics_demo.py
# Metrics at http://localhost:8000/metrics
```

### 6. Cost Tracker (`cost_demo.py`)
Track and visualize costs.

```bash
python demo/cost_demo.py
```

## Docker Demos

```bash
# Build and run
docker-compose -f demo/docker-compose.yml up

# With GPU
docker-compose -f demo/docker-compose.gpu.yml up
```

---

Start with `api_demo.py` for a basic working server.
