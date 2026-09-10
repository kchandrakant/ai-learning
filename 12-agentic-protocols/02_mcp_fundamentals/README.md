# Step 2: Model Context Protocol (MCP) Fundamentals

## What is MCP?

MCP is an open protocol for connecting AI assistants to external data sources and tools. Created by Anthropic, it's becoming the standard for tool integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Host                              │
│  (Claude Desktop, Cursor, IDE, or your application)         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ MCP Server │   │ MCP Server │   │ MCP Server │
   │ (Database) │   │ (GitHub)   │   │ (Slack)    │
   └────────────┘   └────────────┘   └────────────┘
```

**Key terms:**
- **Host:** Application that wants to use MCP (Claude, Cursor)
- **Client:** Protocol handler within the host
- **Server:** Tool/resource provider you build

## Core Primitives

### 1. Resources
Read-only data the server exposes:

```python
@server.resource("file://config.json")
async def get_config():
    return read_file("config.json")

@server.resource("db://users/{user_id}")
async def get_user(user_id: str):
    return fetch_user(user_id)
```

### 2. Tools
Functions the agent can call:

```python
@server.tool()
async def search_database(query: str, limit: int = 10) -> str:
    """Search the database for matching records."""
    results = await db.search(query, limit)
    return json.dumps(results)
```

### 3. Prompts
Pre-built prompt templates:

```python
@server.prompt()
def code_review_prompt(code: str, language: str) -> str:
    return f"Review this {language} code:\n\n{code}"
```

### 4. Sampling (Advanced)
Server requests LLM completion:

```python
# Server can ask the host to generate text
result = await server.request_sampling(
    messages=[{"role": "user", "content": "Summarize: ..."}],
    max_tokens=500
)
```

## Transport Layers

### stdio (Local)
For local processes:
```json
{
  "command": "python",
  "args": ["my_server.py"]
}
```

### HTTP + SSE (Remote)
For network servers:
```
POST /message     - Send messages
GET  /sse         - Receive server events
```

## Message Flow

```
Host                    Server
  │                       │
  │──── initialize ──────▶│
  │◀─── capabilities ─────│
  │                       │
  │──── list_tools ──────▶│
  │◀─── tool list ────────│
  │                       │
  │──── call_tool ───────▶│
  │◀─── result ───────────│
```

## Minimal Server Example

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("demo-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "add":
        return str(arguments["a"] + arguments["b"])

# Run with stdio transport
if __name__ == "__main__":
    server.run()
```

## Configuration (Claude Desktop)

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

## Files

- `mcp_architecture.py` - MCP concepts demonstration
- `minimal_server.py` - Minimal working server

## Key Takeaways

1. MCP connects hosts (Claude) to servers (your tools)
2. Four primitives: Resources, Tools, Prompts, Sampling
3. Two transports: stdio (local), HTTP+SSE (remote)
4. Servers expose capabilities, hosts discover and use them

## What's Next?

Step 3: **Building MCP Servers** — implement production servers.
