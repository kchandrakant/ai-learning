# Module 17: Local Deployment with Ollama + Docker

This module provides hands-on experience deploying LLMs locally using Ollama and Docker. No cloud accounts required — everything runs on your machine.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Install and configure Ollama for local LLM inference
- Run open-weight models (Llama, Mistral, Phi, etc.) locally
- Containerize LLM services with Docker
- Build a complete local inference stack
- Connect local models to your applications

---

## 🎯 Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- 8GB+ RAM (16GB recommended for 7B models)
- ~10GB disk space for models
- Basic command line familiarity

**GPU (optional but recommended):**
- NVIDIA GPU with CUDA support for faster inference
- Apple Silicon Macs work well with Ollama's Metal support

---

## 📚 Part 1: Ollama Basics

### What is Ollama?

Ollama is a tool for running open-weight LLMs locally. It handles:
- Model downloading and management
- Quantization for memory efficiency
- GPU acceleration (NVIDIA CUDA, Apple Metal)
- OpenAI-compatible API

### Installation

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

### Verify Installation

```bash
ollama --version
```

### Start Ollama Server

```bash
# Start the server (runs in background)
ollama serve

# Or it may auto-start on installation
```

The API runs at `http://localhost:11434` by default.

---

## 📚 Part 2: Running Models

### Pull a Model

```bash
# Pull Llama 3.2 (3B parameters, ~2GB)
ollama pull llama3.2

# Pull Mistral (7B parameters, ~4GB)
ollama pull mistral

# Pull a small model for testing
ollama pull phi3:mini
```

### List Available Models

```bash
ollama list
```

### Run Interactive Chat

```bash
ollama run llama3.2
```

Type your prompts, press Enter. Use `/bye` to exit.

### Model Variants

Models come in different sizes and quantizations:

```bash
# Specific size
ollama pull llama3.2:1b    # 1B parameter version
ollama pull llama3.2:3b    # 3B parameter version

# Specific quantization
ollama pull mistral:7b-q4_0   # 4-bit quantized
ollama pull mistral:7b-q8_0   # 8-bit quantized
```

**Quantization trade-offs:**
| Quantization | Memory | Quality | Speed |
|--------------|--------|---------|-------|
| Q4_0 | Lowest | Good | Fastest |
| Q4_K_M | Low | Better | Fast |
| Q8_0 | Medium | Very good | Medium |
| FP16 | High | Best | Slower |

---

## 📚 Part 3: Using the API

### REST API

Ollama provides an OpenAI-compatible API:

```bash
# Generate completion
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat completion
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}'
```

### Python Client

```python
import requests
import json

def chat(model: str, message: str) -> str:
    """Send a chat message to Ollama."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# Usage
response = chat("llama3.2", "Explain machine learning in one sentence.")
print(response)
```

### Streaming Responses

```python
import requests
import json

def chat_stream(model: str, message: str):
    """Stream chat responses from Ollama."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk:
                print(chunk["message"]["content"], end="", flush=True)
            if chunk.get("done"):
                print()  # Newline at end
                break

# Usage
chat_stream("llama3.2", "Write a haiku about programming.")
```

### Using the Official Python Library

```bash
pip install ollama
```

```python
import ollama

# Simple generation
response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### OpenAI Compatibility

Ollama can act as a drop-in replacement for OpenAI:

```python
from openai import OpenAI

# Point to local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Any string works
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

This means code written for OpenAI works locally with minimal changes!

---

## 📚 Part 4: Docker Deployment

### Why Docker?

- **Reproducibility:** Same environment everywhere
- **Isolation:** Don't pollute your system
- **Deployment:** Easy to move to servers
- **Compose:** Combine with other services

### Running Ollama in Docker

**CPU only:**
```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

**With NVIDIA GPU:**
```bash
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

### Pull Models in Container

```bash
# Execute in running container
docker exec ollama ollama pull llama3.2
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
    restart: unless-stopped

volumes:
  ollama_data:
```

Start with:
```bash
docker compose up -d
```

---

## 📚 Part 5: Building a Complete Local Stack

### Project: Local Chat API

Let's build a complete local inference service with:
- Ollama for model serving
- FastAPI for custom API
- Docker Compose for orchestration

### Directory Structure

```
local-llm-stack/
├── docker-compose.yml
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
└── README.md
```

### FastAPI Service

**api/requirements.txt:**
```
fastapi==0.109.0
uvicorn==0.27.0
requests==2.31.0
pydantic==2.5.0
```

**api/main.py:**
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
from typing import List, Optional

app = FastAPI(title="Local LLM API")

OLLAMA_URL = "http://ollama:11434"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "llama3.2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    model: str

@app.get("/health")
def health_check():
    """Check if Ollama is available."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return {"status": "healthy", "ollama": "connected"}
    except:
        return {"status": "unhealthy", "ollama": "disconnected"}

@app.get("/models")
def list_models():
    """List available models."""
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    return response.json()

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a chat message."""
    if request.stream:
        raise HTTPException(400, "Use /chat/stream for streaming")
    
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": {"temperature": request.temperature}
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(response.status_code, "Ollama error")
    
    result = response.json()
    return ChatResponse(
        response=result["message"]["content"],
        model=request.model
    )

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream chat responses."""
    def generate():
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": request.model,
                "messages": [m.model_dump() for m in request.messages],
                "stream": True,
                "options": {"temperature": request.temperature}
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    yield f"data: {json.dumps({'content': chunk['message']['content']})}\n\n"
                if chunk.get("done"):
                    yield "data: [DONE]\n\n"
                    break
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**api/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build: ./api
    container_name: local-llm-api
    ports:
      - "8000:8000"
    depends_on:
      - ollama
    environment:
      - OLLAMA_URL=http://ollama:11434
    restart: unless-stopped

volumes:
  ollama_data:
```

### Running the Stack

```bash
# Start services
docker compose up -d

# Pull a model (first time only)
docker exec ollama ollama pull llama3.2

# Check health
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Testing the API

```python
import requests

# Health check
print(requests.get("http://localhost:8000/health").json())

# List models
print(requests.get("http://localhost:8000/models").json())

# Chat
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "What is Docker?"}]
    }
)
print(response.json()["response"])
```

---

## 📚 Part 6: Performance Tuning

### Memory Management

```bash
# Check model memory usage
ollama ps

# Remove unused models
ollama rm model_name

# Set context window (affects memory)
ollama run llama3.2 --num-ctx 2048  # Smaller context = less memory
```

### GPU Configuration

```bash
# Check GPU usage
nvidia-smi

# Limit GPU memory in Ollama
OLLAMA_GPU_MEMORY_FRACTION=0.8 ollama serve
```

### Concurrent Requests

Ollama handles concurrency automatically, but you can tune:

```bash
# Environment variables
OLLAMA_NUM_PARALLEL=2      # Max concurrent requests
OLLAMA_MAX_LOADED_MODELS=2 # Max models in memory
```

### Benchmarking

```python
import time
import requests

def benchmark(model: str, prompt: str, n: int = 10):
    """Benchmark inference speed."""
    times = []
    
    for i in range(n):
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Get token count from response
        result = response.json()
        tokens = result.get("eval_count", 0)
        print(f"Run {i+1}: {elapsed:.2f}s, {tokens} tokens, {tokens/elapsed:.1f} tok/s")
    
    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.2f}s")

# Run benchmark
benchmark("llama3.2", "Explain quantum computing in 100 words.")
```

---

## 📚 Part 7: Troubleshooting

### Common Issues

**Model won't load:**
```bash
# Check available memory
free -h  # Linux
# or
docker stats  # For containers

# Try smaller model or quantization
ollama pull llama3.2:1b
ollama pull mistral:7b-q4_0
```

**Slow inference:**
```bash
# Check if GPU is being used
ollama ps  # Shows GPU memory if used

# For NVIDIA
nvidia-smi  # Should show ollama process

# Ensure GPU drivers are installed
# For Docker, ensure nvidia-container-toolkit is installed
```

**Connection refused:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check Docker networking
docker network ls
docker inspect ollama | grep IPAddress
```

**Out of memory:**
```bash
# Use smaller context
ollama run model --num-ctx 1024

# Use more aggressive quantization
ollama pull model:q4_0

# Clear GPU memory
# Restart Ollama service
```

### Logs

```bash
# Ollama logs (native)
journalctl -u ollama -f  # Linux with systemd

# Docker logs
docker logs ollama -f

# Verbose mode
OLLAMA_DEBUG=1 ollama serve
```

---

## 🏋️ Exercises

### Exercise 1: Basic Setup
1. Install Ollama on your machine
2. Pull and run `phi3:mini` (smallest model)
3. Have a conversation about a topic of your choice
4. Check memory usage with `ollama ps`

### Exercise 2: API Integration
1. Write a Python script that:
   - Connects to Ollama
   - Sends a prompt
   - Streams the response to console
2. Add error handling for when Ollama isn't running

### Exercise 3: Docker Deployment
1. Create the Docker Compose stack from Part 5
2. Add a simple web UI using Gradio or Streamlit
3. Deploy the complete stack

### Exercise 4: Performance Testing
1. Benchmark 3 different models (vary size/quantization)
2. Compare tokens/second
3. Document memory usage for each
4. Create a recommendation for a laptop with 16GB RAM

---

## ✅ Checklist

Before moving on, verify you can:

- [ ] Install and run Ollama
- [ ] Pull and run different models
- [ ] Use the REST API programmatically
- [ ] Run Ollama in Docker
- [ ] Build a Docker Compose stack with Ollama
- [ ] Troubleshoot common issues
- [ ] Measure and compare inference performance

---

## 🔗 What's Next?

You now have hands-on local deployment experience. This complements the cloud deployment options covered earlier and gives you a foundation for:
- Development and testing without API costs
- Privacy-sensitive applications
- Offline deployments
- Edge computing scenarios

---

## 📖 References

- [Ollama Documentation](https://ollama.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
