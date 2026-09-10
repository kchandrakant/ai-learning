# Step 8: Agent Connect Protocol (ACP)

## What is ACP?

ACP (Agent Connect Protocol) by BeeAI focuses on cloud-native agent deployment and orchestration, with strong emphasis on Kubernetes integration.

## ACP vs A2A

| Aspect | A2A | ACP |
|--------|-----|-----|
| Focus | Agent-to-agent tasks | Agent deployment & lifecycle |
| Discovery | Agent Cards at well-known URL | Service registry |
| Deployment | Agnostic | Kubernetes-native |
| Use case | Agent interoperability | Enterprise platforms |

## Core Concepts

### Agent Definition

```yaml
apiVersion: acp.beeai.dev/v1
kind: Agent
metadata:
  name: research-agent
  namespace: agents
spec:
  image: my-registry/research-agent:v1.0
  replicas: 3
  capabilities:
    - search
    - summarize
  resources:
    limits:
      memory: "1Gi"
      cpu: "500m"
  env:
    - name: LLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: llm-secrets
          key: api-key
```

### Agent Registry

Centralized discovery:

```python
# Register agent
registry.register({
    "name": "research-agent",
    "endpoint": "http://research-agent.agents.svc.cluster.local",
    "capabilities": ["search", "summarize"],
    "version": "1.0.0"
})

# Discover agents by capability
agents = registry.find_by_capability("search")
```

### Communication

```python
# ACP message format
message = {
    "type": "task",
    "source": "orchestrator-agent",
    "target": "research-agent",
    "payload": {
        "action": "search",
        "query": "transformer papers"
    },
    "correlation_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Kubernetes Integration

### Custom Resource Definitions

```yaml
# Install ACP CRDs
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: agents.acp.beeai.dev
spec:
  group: acp.beeai.dev
  versions:
    - name: v1
      served: true
      storage: true
  scope: Namespaced
  names:
    plural: agents
    singular: agent
    kind: Agent
```

### Agent Operator

The ACP operator manages agent lifecycles:

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  ACP Operator                          │  │
│  │  - Watches Agent CRDs                                  │  │
│  │  - Creates Deployments, Services                       │  │
│  │  - Manages agent registry                              │  │
│  │  - Handles scaling, updates                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│     │ Agent A  │   │ Agent B  │   │ Agent C  │            │
│     │ (Pod)    │   │ (Pod)    │   │ (Pod)    │            │
│     └──────────┘   └──────────┘   └──────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Agent Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Pending  │────▶│ Running  │────▶│ Stopping │
└──────────┘     └────┬─────┘     └──────────┘
                      │                 │
                      │                 ▼
                      │           ┌──────────┐
                      └──────────▶│ Stopped  │
                                  └──────────┘
```

## Health and Monitoring

```yaml
spec:
  healthCheck:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 10
    periodSeconds: 30
  
  metrics:
    enabled: true
    port: 9090
    path: /metrics
```

## Scaling

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 10
    metrics:
      - type: Resource
        resource:
          name: cpu
          targetAverageUtilization: 70
      - type: Custom
        custom:
          name: pending_tasks
          targetValue: 5
```

## Implementing ACP Agents

```python
from acp import Agent, Message

class ResearchAgent(Agent):
    name = "research-agent"
    capabilities = ["search", "summarize"]
    
    async def handle_message(self, message: Message):
        if message.payload["action"] == "search":
            result = await self.search(message.payload["query"])
            return self.reply(message, {"results": result})
        
        elif message.payload["action"] == "summarize":
            summary = await self.summarize(message.payload["text"])
            return self.reply(message, {"summary": summary})

# Run agent
if __name__ == "__main__":
    agent = ResearchAgent()
    agent.run(port=8080)
```

## When to Use ACP

**Good fit:**
- Enterprise agent platforms
- Kubernetes-native environments
- Need centralized agent management
- Complex scaling requirements

**Consider alternatives if:**
- Simple agent interactions (use A2A)
- Local/desktop deployment
- Serverless environments

## Files

- `acp_agent.py` - ACP agent implementation
- `agent_crd.yaml` - Kubernetes CRD example
- `deployment.yaml` - Complete deployment

## Key Takeaways

1. ACP is Kubernetes-native
2. Agents as Custom Resources
3. Centralized registry for discovery
4. Built-in lifecycle management
5. Best for enterprise platforms

## What's Next?

Step 9: **OpenAI Patterns** — de facto standards from OpenAI.
