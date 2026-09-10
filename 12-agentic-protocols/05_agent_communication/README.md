# Step 5: The Agent Communication Problem

## Why Agent-to-Agent?

MCP connects agents to tools. But what about connecting agents to agents?

**Use cases:**
- Research agent delegates to search agent
- Coding agent asks review agent
- Orchestrator coordinates specialists

## The Challenge

Without standards:

```
Agent A (Python/LangChain) ──?──▶ Agent B (TypeScript/AutoGen)
Agent C (Hosted service)   ──?──▶ Agent D (Local process)
```

Each pair needs custom integration code.

## Communication Patterns

### 1. Direct Messaging
```
Agent A ────────────────────────▶ Agent B
         "Please search for X"
```

### 2. Task Delegation
```
Agent A ──── Create Task ────────▶ Agent B
        ◀─── Task Result ─────────
```

### 3. Supervisor-Worker
```
Supervisor ──┬── Task 1 ──▶ Worker A
             ├── Task 2 ──▶ Worker B
             └── Task 3 ──▶ Worker C
             
             ◀── Results aggregated
```

### 4. Peer Collaboration
```
Agent A ◀────────────────────────▶ Agent B
        ◀────────────────────────▶ Agent C
```

## What Needs Standardization

### 1. Discovery
How do agents find each other?

```
Agent A: "I need a code review agent"
         ↓
    [Discovery Mechanism]
         ↓
    "Agent B at https://review.example.com"
```

### 2. Capabilities
What can an agent do?

```json
{
  "name": "Code Review Agent",
  "capabilities": ["code_review", "security_scan"],
  "input_formats": ["python", "javascript"],
  "max_file_size": "1MB"
}
```

### 3. Message Format
How do agents structure requests?

```json
{
  "task": "review_code",
  "payload": { "code": "...", "language": "python" },
  "context": { "project": "backend", "urgency": "normal" }
}
```

### 4. Lifecycle
How do tasks progress?

```
Created → Accepted → In Progress → Completed/Failed
```

### 5. Security
How do agents trust each other?

```
Identity → Authentication → Authorization → Audit
```

## Current Solutions

### Framework-Specific
- LangGraph: Graph-based orchestration
- CrewAI: Role-based agents with delegation
- AutoGen: Conversation-based multi-agent

**Problem:** Agents from different frameworks can't talk.

### Custom APIs
Build REST/gRPC APIs for each agent.

**Problem:** Every integration is custom.

### Emerging Protocols
- **A2A (Google):** Task-based agent communication
- **ACP (BeeAI):** Cloud-native agent orchestration

## The A2A Approach

```
┌─────────────────────────────────────────────────────────────┐
│  1. Discovery                                                │
│     GET /.well-known/agent.json                             │
│     → Returns Agent Card with capabilities                   │
├─────────────────────────────────────────────────────────────┤
│  2. Task Submission                                          │
│     POST /tasks/send                                         │
│     → Submit task with messages                              │
├─────────────────────────────────────────────────────────────┤
│  3. Execution                                                │
│     Agent processes task                                     │
│     → May stream updates                                     │
├─────────────────────────────────────────────────────────────┤
│  4. Completion                                               │
│     → Return result or artifact                              │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `communication_patterns.py` - Different agent communication patterns

## Key Takeaways

1. Agents need to communicate with other agents
2. Without standards, every integration is custom
3. Key needs: discovery, capabilities, message format, lifecycle
4. A2A and ACP are emerging solutions

## What's Next?

Step 6: **A2A Protocol** — Google's agent-to-agent standard.
