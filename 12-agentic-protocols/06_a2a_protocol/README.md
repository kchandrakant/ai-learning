# Step 6: Agent2Agent Protocol (A2A)

## What is A2A?

A2A (Agent2Agent) is Google's open protocol for agent interoperability. It enables agents to discover each other and exchange tasks.

## Core Concepts

### Agent Card
JSON document describing an agent's capabilities:

```json
{
  "name": "Research Assistant",
  "description": "Helps with research and summarization",
  "url": "https://research-agent.example.com",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": false
  },
  "authentication": {
    "schemes": ["bearer"]
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "skills": [
    {
      "id": "web_search",
      "name": "Web Search",
      "description": "Search the web for information",
      "tags": ["search", "research"]
    },
    {
      "id": "summarize",
      "name": "Summarize",
      "description": "Summarize documents or text",
      "tags": ["summarization", "text"]
    }
  ]
}
```

### Discovery

Agents publish their Agent Card at:
```
GET /.well-known/agent.json
```

### Tasks

The unit of work in A2A:

```json
{
  "id": "task-123",
  "sessionId": "session-456",
  "status": {
    "state": "working",
    "message": "Processing your request..."
  },
  "messages": [
    {
      "role": "user",
      "parts": [
        {"type": "text", "text": "Search for recent AI papers"}
      ]
    }
  ],
  "artifacts": []
}
```

### Task States

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    ┌─────────┐    ┌─────────┐    ┌──────────┐ │
    │ created │───▶│ working │───▶│ completed│ │
    └─────────┘    └────┬────┘    └──────────┘ │
                        │                       │
                        │         ┌──────────┐ │
                        └────────▶│  failed  │ │
                                  └──────────┘ │
                                       │       │
                   ┌─────────┐         │       │
                   │ canceled│◀────────┴───────┘
                   └─────────┘
```

### Messages and Parts

Messages contain parts (text, files, data):

```json
{
  "role": "agent",
  "parts": [
    {
      "type": "text",
      "text": "Here are the search results:"
    },
    {
      "type": "file",
      "file": {
        "name": "results.json",
        "mimeType": "application/json",
        "bytes": "base64encoded..."
      }
    }
  ]
}
```

## API Endpoints

### Send Task

```http
POST /tasks/send
Content-Type: application/json

{
  "id": "task-123",
  "message": {
    "role": "user",
    "parts": [{"type": "text", "text": "Find papers on transformers"}]
  }
}
```

### Get Task

```http
GET /tasks/task-123
```

### Cancel Task

```http
POST /tasks/task-123/cancel
```

## Streaming

For real-time updates, use SSE:

```http
GET /tasks/task-123/stream
Accept: text/event-stream
```

```
event: status
data: {"state": "working", "message": "Searching..."}

event: artifact
data: {"name": "result.json", "index": 0}

event: status
data: {"state": "completed"}
```

## Multi-Turn Conversations

Tasks can have multiple turns:

```python
# Turn 1
response = await agent.send_task({
    "id": "task-123",
    "message": {"role": "user", "parts": [{"text": "Search for X"}]}
})

# Agent responds with follow-up question
# response.status.state == "input_required"

# Turn 2
response = await agent.send_task({
    "id": "task-123",  # Same task ID
    "message": {"role": "user", "parts": [{"text": "Focus on recent papers"}]}
})
```

## Push Notifications

For long-running tasks:

```json
{
  "id": "task-123",
  "message": {...},
  "pushNotification": {
    "url": "https://my-app.com/webhook",
    "authentication": {
      "schemes": ["bearer"],
      "credentials": "token-xyz"
    }
  }
}
```

## Files

- `a2a_protocol.py` - A2A concepts demonstration
- `agent_card.json` - Example Agent Card

## Key Takeaways

1. Agent Card describes capabilities and location
2. Tasks are the unit of work
3. Messages contain parts (text, files, data)
4. Streaming for real-time updates
5. Push notifications for long-running tasks

## What's Next?

Step 7: **Building A2A Agents** — implement A2A servers.
