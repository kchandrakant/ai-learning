# Agentic Protocols: A Step-by-Step Learning Journey

This course covers the emerging protocols that enable AI agents to communicate, use tools, and interoperate across systems. From MCP to A2A to ACP, understand how agents connect to the world.

---

## 🎯 Why Agentic Protocols Matter

AI agents need standardized ways to:
- **Discover and use tools** (MCP)
- **Communicate with other agents** (A2A)
- **Connect to cloud platforms** (ACP)
- **Access resources securely** (various auth protocols)

Without standards, every agent-tool and agent-agent integration is custom. Protocols enable ecosystems.

---

## 🎯 Prerequisites

- Basic understanding of HTTP, REST, JSON
- Familiarity with LLM APIs
- Completed Multi-Agent Systems course (recommended)

---

## 📚 Part 1: Tool & Context Protocols

How agents discover and use external capabilities.

### Step 1: The Tool Integration Problem
**Why it matters:** Every LLM provider has different function calling formats. Every tool has different APIs. Agents need a universal way to discover and use tools.

**What we'll cover:**
- The fragmentation problem
- Function calling variations (OpenAI, Anthropic, Google)
- Why we need tool protocols
- The protocol landscape overview

**Key insight:**
```
Without protocols:  N agents × M tools = N×M integrations
With protocols:     N agents × 1 protocol × M tools = N+M integrations
```

---

### Step 2: Model Context Protocol (MCP)
**Why it matters:** MCP (by Anthropic) is becoming the standard for tool and context integration. It's already supported by Claude, Cursor, and many IDEs.

**What we'll build:**
- MCP architecture (hosts, clients, servers)
- Resources, tools, and prompts
- Server implementation
- Client integration
- Transport layers (stdio, HTTP+SSE)

**MCP Components:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MCP Host  │────▶│  MCP Client │────▶│  MCP Server │
│  (Claude,   │     │  (Protocol  │     │  (Your tool │
│   Cursor)   │     │   handler)  │     │   provider) │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Key concepts:**
- Resources: File-like data the server exposes
- Tools: Functions the agent can call
- Prompts: Pre-built prompt templates
- Sampling: Server-initiated LLM requests

---

### Step 3: Building MCP Servers
**Why it matters:** To integrate your tools with MCP-compatible agents, you build MCP servers.

**What we'll build:**
- Python MCP server with FastMCP
- TypeScript MCP server
- Tool definitions with schemas
- Resource providers
- Error handling patterns

**Example server structure:**
```python
from mcp.server import Server
from mcp.types import Tool, Resource

server = Server("my-tools")

@server.tool()
async def search_database(query: str) -> str:
    """Search the internal database."""
    return await db.search(query)

@server.resource("config://settings")
async def get_settings() -> str:
    return json.dumps(settings)
```

---

### Step 4: MCP in Production
**Why it matters:** MCP servers need to be reliable, secure, and performant in production.

**What we'll cover:**
- Authentication patterns
- Rate limiting
- Logging and observability
- Deployment strategies
- Multi-server orchestration

---

## 📚 Part 2: Agent-to-Agent Protocols

How agents communicate with each other.

### Step 5: The Agent Communication Problem
**Why it matters:** Multi-agent systems need agents to talk to each other. Without standards, it's chaos.

**What we'll cover:**
- Why agent-to-agent communication is hard
- Message formats and semantics
- Discovery and routing
- Trust and verification

**Communication patterns:**
```
Direct:      Agent A ──────────────▶ Agent B
Broadcast:   Agent A ──┬──▶ Agent B
                       └──▶ Agent C
Hierarchical: Supervisor ──▶ Worker Agents
Peer-to-Peer: Agent A ◀──────────▶ Agent B
```

---

### Step 6: Agent2Agent Protocol (A2A)
**Why it matters:** Google's A2A protocol standardizes how agents discover each other and exchange tasks.

**What we'll build:**
- A2A architecture and concepts
- Agent Cards (discovery mechanism)
- Task lifecycle (send, receive, complete)
- Streaming and push notifications
- Multi-turn conversations

**A2A Flow:**
```
1. Discovery:  Client fetches Agent Card from /.well-known/agent.json
2. Task Send:  POST /tasks/send with message
3. Execution:  Agent processes, may stream updates
4. Completion: Agent returns result or artifact
```

**Key concepts:**
- Agent Card: JSON describing agent capabilities
- Tasks: Units of work with messages and artifacts
- Parts: Text, files, or structured data within messages
- Streaming: Real-time updates via SSE

---

### Step 7: Building A2A Agents
**Why it matters:** To participate in A2A ecosystems, your agents need to speak the protocol.

**What we'll build:**
- A2A server implementation
- Agent Card definition
- Task handling
- Artifact management
- Error handling

**Example Agent Card:**
```json
{
  "name": "Research Assistant",
  "description": "Helps with research tasks",
  "url": "https://agent.example.com",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {"name": "web_search", "description": "Search the web"},
    {"name": "summarize", "description": "Summarize documents"}
  ]
}
```

---

### Step 8: Agent Connect Protocol (ACP)
**Why it matters:** BeeAI's ACP focuses on cloud-native agent deployment and orchestration.

**What we'll cover:**
- ACP architecture
- Agent deployment model
- Communication patterns
- Integration with Kubernetes
- Comparison with A2A

---

## 📚 Part 3: Emerging Standards

The evolving landscape of agent protocols.

### Step 9: OpenAI's Approaches
**Why it matters:** OpenAI's massive user base means their patterns often become de facto standards.

**What we'll cover:**
- Function calling evolution
- Structured outputs
- Assistants API patterns
- Tool use best practices
- Realtime API for voice agents

---

### Step 10: LangChain & Framework Protocols
**Why it matters:** Frameworks define their own tool and agent interfaces that influence the ecosystem.

**What we'll cover:**
- LangChain tool interface
- LangGraph agent protocol
- CrewAI agent communication
- AutoGen conversation patterns
- Framework interoperability challenges

---

### Step 11: Semantic Kernel & Enterprise Patterns
**Why it matters:** Microsoft's Semantic Kernel brings enterprise patterns to agent development.

**What we'll cover:**
- Semantic Kernel plugin model
- Planner patterns
- Memory connectors
- Enterprise integration patterns

---

### Step 12: Protocol Comparison & Selection
**Why it matters:** Different protocols suit different use cases. Know when to use what.

**What we'll build:**
- Decision framework for protocol selection
- Interoperability strategies
- Bridge implementations
- Future-proofing your integrations

**Comparison:**
| Protocol | Focus | Best For |
|----------|-------|----------|
| MCP | Tool/context integration | IDE extensions, tool providers |
| A2A | Agent-to-agent tasks | Multi-agent orchestration |
| ACP | Cloud deployment | Enterprise agent platforms |
| Function Calling | LLM-native tools | Simple integrations |

---

## 📚 Part 4: Building Interoperable Systems

### Step 13: Multi-Protocol Architectures
**Why it matters:** Real systems often need multiple protocols working together.

**What we'll build:**
- MCP + A2A integration
- Protocol adapters and bridges
- Unified tool discovery
- Cross-protocol agent communication

---

### Step 14: Protocol Security
**Why it matters:** Agent protocols need secure authentication, authorization, and data handling.

**What we'll cover:**
- Authentication patterns per protocol
- Token management
- Data privacy considerations
- Audit and compliance

---

### Step 15: The Future of Agent Protocols
**Why it matters:** The landscape is evolving rapidly. Understand where it's heading.

**What we'll cover:**
- IETF and W3C standardization efforts
- Protocol convergence trends
- Emerging protocols to watch
- Building for extensibility

---

## 🗂️ Project Structure

```
agentic-protocols/
├── LEARNING_PATH.md
├── requirements.txt
├── verify_setup.py
│
├── 01_tool_integration_problem/
├── 02_mcp_fundamentals/
├── 03_building_mcp_servers/
├── 04_mcp_production/
├── 05_agent_communication/
├── 06_a2a_protocol/
├── 07_building_a2a_agents/
├── 08_acp_protocol/
├── 09_openai_patterns/
├── 10_framework_protocols/
├── 11_semantic_kernel/
├── 12_protocol_comparison/
├── 13_multi_protocol/
├── 14_protocol_security/
├── 15_future_protocols/
│
├── demo/
└── beyond/
```

---

## 🚀 Let's Begin!

Start with **Step 1: The Tool Integration Problem** to understand why these protocols exist.

---

## 📖 References

### Official Specifications
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [A2A Protocol](https://google.github.io/A2A/)
- [ACP Protocol](https://agentcommunicationprotocol.dev/)

### Implementation Resources
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [A2A Samples](https://github.com/google/A2A/tree/main/samples)

### Community
- MCP Discord
- Agent Protocol Working Groups
