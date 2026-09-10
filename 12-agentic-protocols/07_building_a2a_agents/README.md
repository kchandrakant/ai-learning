# Step 7: Building A2A Agents

## Server Implementation

### Basic Structure

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uuid

app = FastAPI()

# In-memory task storage (use database in production)
tasks = {}

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return {
        "name": "Research Agent",
        "description": "Search and summarize information",
        "url": "https://research.example.com",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False
        },
        "skills": [
            {
                "id": "search",
                "name": "Web Search",
                "description": "Search the web"
            }
        ]
    }
```

### Task Handling

```python
@app.post("/tasks/send")
async def send_task(request: dict):
    task_id = request.get("id", str(uuid.uuid4()))
    
    # Create or update task
    if task_id not in tasks:
        tasks[task_id] = {
            "id": task_id,
            "status": {"state": "created"},
            "messages": [],
            "artifacts": []
        }
    
    task = tasks[task_id]
    
    # Add user message
    task["messages"].append(request["message"])
    task["status"] = {"state": "working", "message": "Processing..."}
    
    # Process asynchronously
    asyncio.create_task(process_task(task_id))
    
    return task

async def process_task(task_id: str):
    task = tasks[task_id]
    
    try:
        # Extract user query
        user_message = task["messages"][-1]
        query = user_message["parts"][0]["text"]
        
        # Do actual work
        result = await do_search(query)
        
        # Add agent response
        task["messages"].append({
            "role": "agent",
            "parts": [{"type": "text", "text": result}]
        })
        task["status"] = {"state": "completed"}
        
    except Exception as e:
        task["status"] = {
            "state": "failed",
            "message": str(e)
        }
```

### Get and Cancel Tasks

```python
@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    tasks[task_id]["status"] = {"state": "canceled"}
    return tasks[task_id]
```

### Streaming Responses

```python
@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    
    async def event_generator():
        task = tasks[task_id]
        last_state = None
        
        while True:
            if task["status"]["state"] != last_state:
                last_state = task["status"]["state"]
                yield f"event: status\ndata: {json.dumps(task['status'])}\n\n"
            
            if last_state in ["completed", "failed", "canceled"]:
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

## Client Implementation

```python
import httpx

class A2AClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def get_agent_card(self):
        response = await self.client.get(
            f"{self.base_url}/.well-known/agent.json"
        )
        return response.json()
    
    async def send_task(self, task_id: str, message: str):
        response = await self.client.post(
            f"{self.base_url}/tasks/send",
            json={
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}]
                }
            }
        )
        return response.json()
    
    async def get_task(self, task_id: str):
        response = await self.client.get(
            f"{self.base_url}/tasks/{task_id}"
        )
        return response.json()
    
    async def wait_for_completion(self, task_id: str, timeout: int = 60):
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            task = await self.get_task(task_id)
            state = task["status"]["state"]
            
            if state in ["completed", "failed", "canceled"]:
                return task
            
            await asyncio.sleep(1)
        
        raise TimeoutError("Task did not complete in time")
```

## Using the Client

```python
async def main():
    client = A2AClient("https://research.example.com")
    
    # Discover agent
    card = await client.get_agent_card()
    print(f"Connected to: {card['name']}")
    print(f"Skills: {[s['name'] for s in card['skills']]}")
    
    # Send task
    task = await client.send_task(
        task_id="my-task-123",
        message="Find recent papers on transformer efficiency"
    )
    print(f"Task created: {task['id']}")
    
    # Wait for result
    result = await client.wait_for_completion("my-task-123")
    
    if result["status"]["state"] == "completed":
        agent_response = result["messages"][-1]
        print(f"Result: {agent_response['parts'][0]['text']}")
    else:
        print(f"Task failed: {result['status']['message']}")
```

## Multi-Agent Orchestration

```python
async def orchestrate_research(topic: str):
    search_client = A2AClient("https://search-agent.example.com")
    summary_client = A2AClient("https://summary-agent.example.com")
    
    # Step 1: Search
    search_task = await search_client.send_task(
        task_id=f"search-{uuid.uuid4()}",
        message=f"Find papers about {topic}"
    )
    search_result = await search_client.wait_for_completion(search_task["id"])
    papers = search_result["messages"][-1]["parts"][0]["text"]
    
    # Step 2: Summarize
    summary_task = await summary_client.send_task(
        task_id=f"summary-{uuid.uuid4()}",
        message=f"Summarize these papers:\n\n{papers}"
    )
    summary_result = await summary_client.wait_for_completion(summary_task["id"])
    
    return summary_result["messages"][-1]["parts"][0]["text"]
```

## Files

- `a2a_server.py` - Complete A2A server
- `a2a_client.py` - A2A client library
- `orchestration.py` - Multi-agent example

## Key Takeaways

1. Serve Agent Card at `/.well-known/agent.json`
2. Handle tasks asynchronously
3. Use SSE for streaming updates
4. Client discovers, sends, and polls for results
5. Orchestrate multiple A2A agents for complex workflows

## What's Next?

Step 8: **ACP Protocol** — cloud-native agent deployment.
