# Agentic Protocols Demos

Interactive demonstrations of agent protocols.

## Available Demos

### 1. MCP Server (`mcp_server_demo.py`)
Run a local MCP server with tools.

```bash
python demo/mcp_server_demo.py
```

### 2. MCP Client (`mcp_client_demo.py`)
Connect to an MCP server and call tools.

```bash
python demo/mcp_client_demo.py
```

### 3. A2A Agent (`a2a_agent_demo.py`)
Run an A2A-compliant agent server.

```bash
python demo/a2a_agent_demo.py
```

### 4. A2A Client (`a2a_client_demo.py`)
Discover and send tasks to A2A agents.

```bash
python demo/a2a_client_demo.py --agent-url http://localhost:8000
```

### 5. Protocol Comparison (`protocol_comparison_demo.py`)
See how the same tool is exposed via different protocols.

```bash
python demo/protocol_comparison_demo.py
```

### 6. Multi-Protocol Agent (`multi_protocol_demo.py`)
Agent that uses MCP for tools and A2A for delegation.

```bash
python demo/multi_protocol_demo.py
```

### 7. Protocol Gateway (`gateway_demo.py`)
Unified gateway routing to different protocols.

```bash
python demo/gateway_demo.py
```

## Running Demos

```bash
pip install -r requirements.txt
python demo/<demo_name>.py
```

## Recommended Order

1. Start with `mcp_server_demo.py` and `mcp_client_demo.py`
2. Move to `a2a_agent_demo.py` and `a2a_client_demo.py`
3. Compare with `protocol_comparison_demo.py`
4. Build on with `multi_protocol_demo.py`

---

These demos provide hands-on experience with real protocol implementations.
