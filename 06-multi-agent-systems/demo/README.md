# Multi-Agent Systems Demos

Interactive demonstrations that bring together concepts from the course modules.

## Available Demos

### 1. Single Agent Demo (`single_agent_demo.py`)
Build and test a basic ReAct agent:
- Tool definition and usage
- Observation-thought-action loop
- Error handling

```bash
python demo/single_agent_demo.py
```

### 2. LangGraph Agent (`langgraph_demo.py`)
Stateful agent with explicit control flow:
- State machine visualization
- Conditional routing
- Checkpointing and replay

```bash
python demo/langgraph_demo.py
```

### 3. Multi-Agent Supervisor (`supervisor_demo.py`)
Supervisor pattern with worker agents:
- Task delegation
- Result aggregation
- Handoff visualization

```bash
python demo/supervisor_demo.py
```

### 4. Agent Debate (`debate_demo.py`)
Peer-to-peer agent collaboration:
- Multiple perspectives on a topic
- Structured debate rounds
- Synthesis by judge agent

```bash
python demo/debate_demo.py
```

### 5. Workflow Orchestration (`workflow_demo.py`)
Complex multi-step workflow:
- Sequential and parallel execution
- Human approval gates
- Error recovery

```bash
python demo/workflow_demo.py
```

### 6. Research Assistant (`research_demo.py`)
Complete multi-agent research system:
- Researcher agent (web search, summarization)
- Writer agent (content creation)
- Editor agent (review and improvement)

```bash
python demo/research_demo.py --topic "AI agents in 2024"
```

### 7. Code Generation Agent (`code_demo.py`)
Coding agent with safety:
- Code generation
- Sandboxed execution
- Test verification

```bash
python demo/code_demo.py
```

## Running Demos

1. Ensure you've completed setup:
   ```bash
   pip install -r requirements.txt
   python verify_setup.py
   ```

2. Set your API keys:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. Run any demo:
   ```bash
   python demo/<demo_name>.py
   ```

## Demo Output

Each demo generates:
- Console output showing agent reasoning
- Trace logs for debugging
- Metrics summary (tokens, cost, latency)

## Building Your Own

Use these demos as templates for your own agents. Key patterns:
- Start simple, add complexity gradually
- Always include observability
- Test with edge cases

---

Ready to experiment? Start with `single_agent_demo.py` to see basic agent behavior.
