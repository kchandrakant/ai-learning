# Step 13: Multi-Protocol Architectures

## Why Multiple Protocols?

Real systems often need:
- Tools (MCP) AND agent collaboration (A2A)
- Discovery (A2A) AND deployment (ACP)
- Standard protocols AND custom integrations

## Architecture Patterns

### Pattern 1: MCP Tools with A2A Delegation

```
┌─────────────────────────────────────────────────────────┐
│                   Primary Agent                          │
│  - Uses MCP for tools                                   │
│  - Delegates to A2A specialists                         │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ MCP                          │ A2A
         ▼                              ▼
┌─────────────────┐           ┌─────────────────┐
│  Tool Server    │           │  Specialist     │
│  - Database     │           │  Agent          │
│  - File system  │           │  - Research     │
│  - APIs         │           │  - Coding       │
└─────────────────┘           └─────────────────┘
```

### Implementation

```python
class HybridAgent:
    def __init__(self):
        self.mcp_client = MCPClient()
        self.a2a_client = A2AClient()
    
    async def execute(self, task: str):
        # Decide: tool or delegation
        if self.needs_tool(task):
            return await self.use_mcp_tool(task)
        elif self.needs_specialist(task):
            return await self.delegate_a2a(task)
        else:
            return await self.handle_directly(task)
    
    async def use_mcp_tool(self, task: str):
        tool_name, args = self.parse_tool_call(task)
        return await self.mcp_client.call_tool(tool_name, args)
    
    async def delegate_a2a(self, task: str):
        # Find appropriate agent
        specialists = await self.a2a_client.discover_agents()
        agent = self.select_best(specialists, task)
        
        # Delegate
        task_result = await self.a2a_client.send_task(
            agent_url=agent["url"],
            message=task
        )
        return await self.a2a_client.wait_for_completion(task_result["id"])
```

### Pattern 2: Protocol Gateway

Unified interface that speaks multiple protocols:

```
┌─────────────────────────────────────────────────────────┐
│                   Protocol Gateway                       │
│  - Unified API                                          │
│  - Routes to appropriate protocol                        │
└─────────────────────────────────────────────────────────┘
         │              │              │
         │ MCP          │ A2A          │ HTTP
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │ Tools  │    │ Agents │    │  APIs  │
    └────────┘    └────────┘    └────────┘
```

```python
class ProtocolGateway:
    async def route(self, request: GatewayRequest):
        if request.target_type == "tool":
            return await self.route_mcp(request)
        elif request.target_type == "agent":
            return await self.route_a2a(request)
        elif request.target_type == "api":
            return await self.route_http(request)
    
    async def route_mcp(self, request):
        server = self.mcp_registry.find(request.target)
        return await self.mcp_client.call_tool(
            server=server,
            tool=request.action,
            args=request.params
        )
    
    async def route_a2a(self, request):
        agent = self.a2a_registry.find(request.target)
        return await self.a2a_client.send_task(
            agent_url=agent["url"],
            message=request.to_message()
        )
```

### Pattern 3: A2A Agent with MCP Backend

```python
# A2A-compliant agent that uses MCP internally

class ResearchAgent:
    def __init__(self):
        self.mcp_tools = MCPClient()  # Internal tools
    
    # A2A interface
    async def handle_task(self, task: A2ATask):
        query = task.messages[-1].parts[0].text
        
        # Use MCP tools internally
        search_results = await self.mcp_tools.call_tool(
            "search",
            {"query": query}
        )
        
        summary = await self.mcp_tools.call_tool(
            "summarize",
            {"text": search_results}
        )
        
        return A2AResponse(
            status="completed",
            message={"role": "agent", "parts": [{"text": summary}]}
        )

# Expose via A2A
@app.get("/.well-known/agent.json")
async def agent_card():
    return {"name": "Research Agent", ...}

@app.post("/tasks/send")
async def send_task(request):
    return await research_agent.handle_task(request)
```

## Protocol Adapters

### MCP to A2A Adapter

Expose MCP tools as an A2A agent:

```python
class MCPtoA2AAdapter:
    def __init__(self, mcp_server_config):
        self.mcp = MCPClient(mcp_server_config)
    
    def get_agent_card(self):
        tools = self.mcp.list_tools()
        return {
            "name": f"MCP Tools: {self.mcp.server_name}",
            "skills": [
                {"id": t.name, "description": t.description}
                for t in tools
            ]
        }
    
    async def handle_task(self, task: A2ATask):
        # Parse which tool to call
        message = task.messages[-1].parts[0].text
        tool_name, args = self.parse_tool_request(message)
        
        # Call MCP tool
        result = await self.mcp.call_tool(tool_name, args)
        
        return {"status": "completed", "result": result}
```

### A2A to MCP Adapter

Expose A2A agents as MCP tools:

```python
class A2AtoMCPAdapter:
    def __init__(self, a2a_agents: list[str]):
        self.agents = {url: A2AClient(url) for url in a2a_agents}
    
    def list_tools(self):
        tools = []
        for url, client in self.agents.items():
            card = client.get_agent_card()
            for skill in card["skills"]:
                tools.append(Tool(
                    name=f"{card['name']}_{skill['id']}",
                    description=skill["description"]
                ))
        return tools
    
    async def call_tool(self, name: str, args: dict):
        agent_name, skill = name.rsplit("_", 1)
        client = self.find_client(agent_name)
        
        task = await client.send_task(args["message"])
        return await client.wait_for_completion(task["id"])
```

## Files

- `hybrid_agent.py` - MCP + A2A agent
- `protocol_gateway.py` - Unified gateway
- `adapters.py` - Protocol adapters

## Key Takeaways

1. Real systems often need multiple protocols
2. Gateway pattern provides unified interface
3. Adapters bridge between protocols
4. A2A for external, MCP for internal is common

## What's Next?

Step 14: **Protocol Security** — securing multi-protocol systems.
