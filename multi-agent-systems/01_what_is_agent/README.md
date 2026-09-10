# Step 1: What is an Agent?

## The Overloaded Term

"Agent" means different things to different people. Let's define it clearly.

## Agent vs Chatbot

```
Chatbot:
  Human: "What's the weather?"
  Bot:   "I don't have access to weather data."
  [End of interaction]

Agent:
  Human: "What's the weather in SF?"
  Agent: [Thinks] I need to get weather data
         [Acts] calls get_weather("San Francisco")
         [Observes] {"temp": 65, "sunny": true}
         [Responds] "It's 65°F and sunny in San Francisco"
```

**Key difference:** Agents can take actions to accomplish goals.

## The Agent Formula

```
Agent = LLM + Tools + Memory + Loop
```

| Component | Purpose | Example |
|-----------|---------|---------|
| **LLM** | Reasoning, planning, language | GPT-4, Claude |
| **Tools** | Actions in the world | API calls, code execution |
| **Memory** | Remember past interactions | Conversation history, vector DB |
| **Loop** | Keep going until done | ReAct, Plan-Execute |

## The Agentic Loop

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│   ┌──────────┐                                      │
│   │  Goal    │                                      │
│   └────┬─────┘                                      │
│        │                                            │
│        ▼                                            │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │ Observe  │───▶│  Think   │───▶│   Act    │    │
│   └────▲─────┘    └──────────┘    └────┬─────┘    │
│        │                               │          │
│        │         ┌──────────┐          │          │
│        └─────────│  Result  │◀─────────┘          │
│                  └────┬─────┘                      │
│                       │                            │
│                       ▼                            │
│              ┌─────────────────┐                   │
│              │  Goal complete? │                   │
│              └────────┬────────┘                   │
│                   Yes │ No                         │
│                       ▼                            │
│                  ┌─────────┐                       │
│                  │  Done   │                       │
│                  └─────────┘                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## When to Use Agents

✅ **Use agents when:**
- Task requires multiple steps
- Need to interact with external systems
- Goal achievement requires iteration
- Task scope isn't fully known upfront

❌ **Don't use agents when:**
- Simple, single-turn response suffices
- No external actions needed
- Deterministic workflow exists
- Latency/cost is critical

## Levels of Agency

```
Level 0: Static prompt → response (not an agent)
Level 1: Single tool call (function calling)
Level 2: Multi-step tool use (ReAct)
Level 3: Planning + execution (Plan-and-Solve)
Level 4: Self-reflection + correction (Reflexion)
Level 5: Multi-agent collaboration (Supervisor/Swarm)
```

## A Minimal Agent

```python
class MinimalAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
    
    def run(self, goal: str, max_steps: int = 10) -> str:
        messages = [{"role": "user", "content": goal}]
        
        for step in range(max_steps):
            # Think: Ask LLM what to do
            response = self.llm.chat(messages, tools=self.tools)
            
            if response.tool_calls:
                # Act: Execute tools
                for call in response.tool_calls:
                    result = self.tools[call.name].execute(call.args)
                    messages.append({
                        "role": "tool",
                        "content": str(result)
                    })
            else:
                # Done: Return final response
                return response.content
        
        return "Max steps reached"
```

## Files

- `what_is_agent.py` - Minimal agent implementation

## Key Takeaways

1. Agents = LLM + Tools + Memory + Loop
2. Agents can take actions; chatbots just respond
3. The core loop: Observe → Think → Act → Repeat
4. Use agents for multi-step, interactive tasks
5. Not everything needs to be an agent

## What's Next?

Step 2: **The ReAct Pattern** — the foundational reasoning + acting framework.
