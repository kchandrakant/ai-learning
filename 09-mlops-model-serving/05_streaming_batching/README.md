# Step 5: Streaming & Batching

## Streaming Responses

Users expect real-time feedback. Stream tokens as generated.

### Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if request.stream:
        return EventSourceResponse(generate_stream(request))
    return generate_sync(request)

async def generate_stream(request):
    for token in model.generate_tokens(request.messages):
        chunk = {"choices": [{"delta": {"content": token}}]}
        yield {"data": json.dumps(chunk)}
        await asyncio.sleep(0)  # Yield control
    yield {"data": "[DONE]"}
```

### Client-Side Consumption

```python
import httpx

async def stream_chat(prompt: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        chunk = json.loads(data)
                        print(chunk["choices"][0]["delta"].get("content", ""), end="")
```

## Request Batching

Batch multiple requests for better GPU utilization.

```python
import asyncio
from collections import deque

class BatchProcessor:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 50):
        self.queue = deque()
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000
    
    async def add_request(self, request) -> asyncio.Future:
        future = asyncio.Future()
        self.queue.append((request, future))
        
        # Process if batch full or timeout
        if len(self.queue) >= self.max_batch:
            await self.process_batch()
        
        return await future
    
    async def process_batch(self):
        batch = []
        futures = []
        while self.queue and len(batch) < self.max_batch:
            req, fut = self.queue.popleft()
            batch.append(req)
            futures.append(fut)
        
        # Process batch
        results = model.generate_batch(batch)
        
        # Return results
        for fut, result in zip(futures, results):
            fut.set_result(result)
```

## Files

- `streaming_batching.py` - Streaming and batching implementations
