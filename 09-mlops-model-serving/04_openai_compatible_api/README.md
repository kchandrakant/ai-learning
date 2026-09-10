# Step 4: OpenAI-Compatible APIs

## The Standard

OpenAI's API format is the de facto standard. Compatibility = easy integration.

## Core Endpoints

### Chat Completions

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 1.0
    max_tokens: int = None
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Generate response
    response = generate(request.messages, request.temperature)
    
    return {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": count_tokens(request.messages),
            "completion_tokens": count_tokens(response),
            "total_tokens": total
        }
    }
```

### Embeddings

```python
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    vectors = embed(request.input)
    
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": request.model,
        "usage": {"prompt_tokens": count, "total_tokens": count}
    }
```

### Models List

```python
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llama-2-7b", "object": "model", "owned_by": "meta"}
        ]
    }
```

## Streaming Response

```python
from sse_starlette.sse import EventSourceResponse

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.stream:
        return EventSourceResponse(stream_response(request))
    return generate_response(request)

async def stream_response(request):
    for token in generate_stream(request.messages):
        chunk = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield {"data": json.dumps(chunk)}
    
    # Final chunk
    yield {"data": json.dumps({"choices": [{"finish_reason": "stop"}]})}
    yield {"data": "[DONE]"}
```

## Files

- `openai_api.py` - Complete API implementation
