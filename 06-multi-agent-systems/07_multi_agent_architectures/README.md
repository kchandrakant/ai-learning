# Step 7: Multi-Agent Architectures

## Why Multiple Agents?

Single agents struggle with:
- Complex tasks requiring diverse expertise
- Long-running workflows
- Tasks needing checks and balances

**Solution:** Specialized agents working together.

## Architecture Patterns

### 1. Supervisor Pattern

One "boss" agent delegates to worker agents:

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │   (Router)   │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Research│      │  Write  │      │  Code   │
    │  Agent  │      │  Agent  │      │  Agent  │
    └─────────┘      └─────────┘      └─────────┘
```

```python
class Supervisor:
    def __init__(self, workers: dict[str, Agent]):
        self.workers = workers
    
    def run(self, task: str) -> str:
        plan = self.create_plan(task)
        results = {}
        
        for step in plan:
            worker_name = step["worker"]
            subtask = step["task"]
            
            result = self.workers[worker_name].run(subtask)
            results[step["id"]] = result
        
        return self.synthesize(results)
    
    def create_plan(self, task: str) -> list[dict]:
        prompt = f"""
        Given this task, create a plan using these workers: {list(self.workers.keys())}
        
        Task: {task}
        
        Return a list of steps with worker and subtask.
        """
        return json.loads(self.llm(prompt))
```

### 2. Peer-to-Peer (Debate)

Agents discuss and refine ideas together:

```
    ┌─────────┐     debate      ┌─────────┐
    │ Agent A │◀───────────────▶│ Agent B │
    └────┬────┘                 └────┬────┘
         │                           │
         │         ┌─────────┐       │
         └────────▶│ Agent C │◀──────┘
                   │ (Judge) │
                   └─────────┘
```

```python
class DebateSystem:
    def __init__(self, agents: list[Agent], judge: Agent):
        self.agents = agents
        self.judge = judge
    
    def debate(self, topic: str, rounds: int = 3) -> str:
        history = []
        
        for round in range(rounds):
            for agent in self.agents:
                response = agent.respond(topic, history)
                history.append({
                    "agent": agent.name,
                    "round": round,
                    "response": response
                })
        
        # Judge synthesizes
        return self.judge.synthesize(topic, history)
```

### 3. Hierarchical

Multi-level management for complex tasks:

```
                    ┌─────────────┐
                    │     CEO     │
                    │   (Agent)   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────┴─────┐            ┌──────┴─────┐
        │ Engineering│            │  Research  │
        │  Manager   │            │  Manager   │
        └─────┬─────┘            └──────┬─────┘
              │                         │
      ┌───────┼───────┐         ┌───────┼───────┐
      │       │       │         │       │       │
    ┌─┴─┐   ┌─┴─┐   ┌─┴─┐     ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
    │Dev│   │Dev│   │QA │     │Res│   │Res│   │Res│
    └───┘   └───┘   └───┘     └───┘   └───┘   └───┘
```

### 4. Swarm

Agents share memory, work in parallel:

```
    ┌─────────────────────────────────────┐
    │         Shared Memory / State        │
    └─────────────────────────────────────┘
           ▲       ▲       ▲       ▲
           │       │       │       │
       ┌───┴───┬───┴───┬───┴───┬───┴───┐
       │       │       │       │       │
    ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐
    │Agent│ │Agent│ │Agent│ │Agent│ │Agent│
    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

```python
class Swarm:
    def __init__(self, agents: list[Agent], shared_state: dict):
        self.agents = agents
        self.state = shared_state
    
    async def run(self, task: str):
        # Agents work in parallel, reading/writing shared state
        tasks = [
            agent.work(task, self.state)
            for agent in self.agents
        ]
        
        await asyncio.gather(*tasks)
        return self.state["result"]
```

## Choosing an Architecture

| Architecture | Best For | Complexity |
|--------------|----------|------------|
| **Supervisor** | Clear subtask delegation | Low |
| **Peer-to-Peer** | Creative tasks, quality improvement | Medium |
| **Hierarchical** | Large, complex organizations | High |
| **Swarm** | Parallel exploration, brainstorming | Medium |

## Implementation in LangGraph

### Supervisor Pattern

```python
from langgraph.graph import StateGraph

class MultiAgentState(TypedDict):
    task: str
    current_agent: str
    results: dict
    final_answer: str

def supervisor(state: MultiAgentState) -> MultiAgentState:
    """Decide which agent to call next."""
    prompt = f"""
    Task: {state['task']}
    Completed: {list(state['results'].keys())}
    
    Which agent should handle the next step?
    Options: researcher, writer, coder, FINISH
    """
    next_agent = llm(prompt).strip().lower()
    return {"current_agent": next_agent}

def researcher(state: MultiAgentState) -> MultiAgentState:
    result = research_agent.run(state["task"])
    return {"results": {"research": result}}

def writer(state: MultiAgentState) -> MultiAgentState:
    result = writer_agent.run(state["task"], state["results"])
    return {"results": {"writing": result}}

# Build graph
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

def route(state):
    agent = state["current_agent"]
    if agent == "finish":
        return END
    return agent

graph.add_conditional_edges("supervisor", route)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.set_entry_point("supervisor")
```

## Agent Handoff

When one agent passes control to another:

```python
class AgentHandoff:
    def __init__(self, from_agent: str, to_agent: str, context: dict):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.context = context
        self.summary = self._create_summary()
    
    def _create_summary(self) -> str:
        """Summarize work done for handoff."""
        return llm(f"""
        Summarize this work for handoff to {self.to_agent}:
        {self.context}
        
        Focus on: what was done, what's needed next, key findings.
        """)
```

## Files

- `multi_agent_architectures.py` - All patterns implemented

## Key Takeaways

1. Supervisor: Boss delegates to specialists
2. Peer-to-peer: Agents debate and refine
3. Hierarchical: Multi-level for complex orgs
4. Swarm: Parallel work with shared state
5. Choose based on task complexity and structure

## What's Next?

Step 8: **Agent Communication** — effective information sharing between agents.
