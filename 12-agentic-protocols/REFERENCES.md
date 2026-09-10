# Agentic Protocols: Key References

A curated collection of specifications and papers on MCP, A2A, ACP, and agent communication standards.

---

## 📚 How to Use This Document

- **📖 Essential**: Core specifications everyone should read
- **🔧 Hands-on**: Implementation resources
- **🔬 Frontier**: Emerging standards

---

## Model Context Protocol (MCP)

### MCP Specification
**Anthropic, November 2024**

Open protocol for connecting AI models to tools and data sources.

- **Key innovations**: Standardized tool interface, resource management, prompts
- **Impact**: Adopted by Claude, Cursor, and many IDEs
- **Link**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)
- **Status**: 📖 Essential

---

### MCP Architecture Overview
**Anthropic**

Design principles and architecture of MCP.

- **Key topics**: Client-server model, transport layers, capability negotiation
- **Link**: [MCP Docs](https://modelcontextprotocol.io/docs/concepts/architecture)
- **Status**: 🔧 Hands-on

---

## Agent-to-Agent Protocol (A2A)

### A2A Specification
**Google, April 2025**

Protocol for peer-to-peer task delegation between agents.

- **Key innovations**: Agent Cards, capability-based discovery, task delegation
- **Impact**: Enables agent collaboration across vendors
- **Link**: [Google A2A](https://google.github.io/a2a-spec/)
- **Status**: 📖 Essential

---

### A2A Architecture
**Google**

Design of agent-to-agent communication.

- **Key topics**: Task lifecycle, Agent Cards, secure delegation
- **Link**: [A2A Docs](https://google.github.io/a2a-spec/architecture/)
- **Status**: 🔧 Hands-on

---

## Agent Communication Protocol (ACP)

### ACP Specification
**BeeAI/IBM, 2025**

REST-based protocol for enterprise agent messaging.

- **Key innovations**: Simple REST API, multimodal messaging, enterprise focus
- **Impact**: Lightweight alternative for internal systems
- **Link**: [agentcommunicationprotocol.dev](https://agentcommunicationprotocol.dev/)
- **Status**: 🔧 Hands-on

---

## Comparative Analysis

### A Comparative Study of AI Agent Communication Protocols
**arXiv:2505.02279, May 2025**

Analysis of MCP, A2A, ACP, and ANP.

- **Key value**: When to use which protocol, interoperability
- **Link**: [arXiv:2505.02279](https://arxiv.org/abs/2505.02279)
- **Status**: 📖 Essential

---

### Developer's Guide to AI Agent Protocols
**Google, 2025**

Practical guide to implementing agent protocols.

- **Key value**: Protocol selection, implementation patterns
- **Link**: [Google Developers Blog](https://developers.googleblog.com/developers-guide-to-ai-agent-protocols/)
- **Status**: 🔧 Hands-on

---

## Agent Network Protocol (ANP)

### ANP Specification
**2025**

Decentralized agent discovery and collaboration.

- **Key innovations**: W3C DIDs, JSON-LD graphs, agent marketplaces
- **Link**: [ANP Spec](https://github.com/agent-network-protocol/anp-spec)
- **Status**: 🔬 Frontier

---

## Protocol Integration Patterns

### MCP + A2A Integration
**2026**

Using MCP servers inside A2A agents.

- **Key pattern**: MCP for tools, A2A for agent collaboration
- **Link**: [a2a-mcp.org](https://a2a-mcp.org/)
- **Status**: 🔧 Hands-on

---

### Production Multi-Agent Architecture
**2026**

Reference architecture combining protocols.

- **Key insights**: MCP for tool access, ACP for internal messaging, A2A for cross-org
- **Link**: [neosalpha.com](https://neosalpha.com/blogs/ai-agent-protocols-acp-vs-mcp-vs-a2a/)
- **Status**: 📝 Reference

---

## Tool & API Standards

### OpenAPI Specification
**OpenAPI Initiative**

Standard for describing REST APIs.

- **Key value**: Tool descriptions for agents
- **Link**: [openapis.org](https://www.openapis.org/)
- **Status**: 📖 Essential

---

### JSON Schema
**json-schema.org**

Standard for describing JSON data structures.

- **Key value**: Tool parameter validation
- **Link**: [json-schema.org](https://json-schema.org/)
- **Status**: 📖 Essential

---

### AsyncAPI
**AsyncAPI Initiative**

Standard for event-driven APIs.

- **Key value**: Async tool interfaces
- **Link**: [asyncapi.com](https://www.asyncapi.com/)
- **Status**: 🔧 Hands-on

---

## Implementation Resources

### MCP SDKs

| Language | Repository | Link |
|----------|------------|------|
| TypeScript | @modelcontextprotocol/sdk | [npm](https://www.npmjs.com/package/@modelcontextprotocol/sdk) |
| Python | mcp | [PyPI](https://pypi.org/project/mcp/) |
| Rust | mcp-rust-sdk | [crates.io](https://crates.io/crates/mcp-rust-sdk) |

### A2A SDKs

| Language | Repository | Link |
|----------|------------|------|
| Python | a2a-sdk | [GitHub](https://github.com/google/a2a-sdk-python) |
| TypeScript | a2a-sdk | [GitHub](https://github.com/google/a2a-sdk-js) |

---

## MCP Server Examples

| Server | Description | Link |
|--------|-------------|------|
| filesystem | File system access | [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) |
| github | GitHub integration | [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/github) |
| postgres | Database access | [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) |
| puppeteer | Browser automation | [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer) |

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| MCP Documentation | Official MCP docs | [modelcontextprotocol.io](https://modelcontextprotocol.io/) |
| A2A Documentation | Official A2A docs | [google.github.io/a2a-spec](https://google.github.io/a2a-spec/) |
| ACP Documentation | Official ACP docs | [agentcommunicationprotocol.dev](https://agentcommunicationprotocol.dev/) |
| Awesome MCP Servers | Community MCP servers | [GitHub](https://github.com/punkpeye/awesome-mcp-servers) |

---

## Reading Order Recommendation

### If starting with agent protocols:
1. MCP Specification (understand tool access)
2. A2A Specification (understand agent collaboration)
3. Comparative study paper (understand when to use what)

### If building production systems:
1. MCP SDK + build a server
2. A2A Agent Card design
3. Protocol integration patterns

---

*Last updated: September 2026*
