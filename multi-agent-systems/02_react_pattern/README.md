# Step 2: The ReAct Pattern

## Reasoning + Acting

ReAct interleaves reasoning (thinking out loud) with acting (using tools). This makes the agent's decision process transparent and debuggable.

## The Pattern

```
Thought: [What am I trying to do? What do I know?]
Action: [tool_name(arguments)]
Observation: [Result from tool]
Thought: [What did I learn? What next?]
Action: [next_tool(arguments)]
Observation: [Result]
... repeat ...
Thought: [I have enough information]
Action: finish(answer)
```

## Concrete Example

```
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first, then its population.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I know the capital is Paris. I need its population.
Action: search("population of Paris")
Observation: The population of Paris is approximately 2.1 million.

Thought: I have all the information needed to answer.
Action: finish("The population of Paris, the capital of France, is approximately 2.1 million.")
```

## Implementation

```python
import re
from typing import Callable

class ReActAgent:
    def __init__(self, llm, tools: dict[str, Callable]):
        self.llm = llm
        self.tools = tools
        self.tools["finish"] = lambda answer: answer  # Special finish tool
    
    def run(self, question: str, max_steps: int = 10) -> str:
        prompt = self._build_prompt(question)
        trajectory = []
        
        for step in range(max_steps):
            # Get next thought + action
            response = self.llm(prompt + "".join(trajectory))
            trajectory.append(response)
            
            # Parse action
            action_match = re.search(r"Action: (\w+)\((.*?)\)", response)
            if not action_match:
                continue
            
            tool_name, args = action_match.groups()
            
            # Check for finish
            if tool_name == "finish":
                return args.strip('"\'')
            
            # Execute tool
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](args)
                    observation = f"\nObservation: {result}\n"
                except Exception as e:
                    observation = f"\nObservation: Error - {e}\n"
            else:
                observation = f"\nObservation: Unknown tool '{tool_name}'\n"
            
            trajectory.append(observation)
        
        return "Max steps reached without answer"
    
    def _build_prompt(self, question: str) -> str:
        return f"""Answer the following question by reasoning step-by-step.

Available tools:
- search(query): Search for information
- calculate(expression): Evaluate a math expression
- finish(answer): Return the final answer

Use this format:
Thought: [your reasoning]
Action: tool_name("arguments")
Observation: [tool result - will be provided]

Question: {question}

"""
```

## Why ReAct Works

1. **Transparency**: See the agent's reasoning
2. **Debuggability**: Know exactly where things went wrong
3. **Grounding**: Actions are tied to explicit reasoning
4. **Self-correction**: Observations inform next steps

## ReAct vs Other Patterns

| Pattern | Approach | Best For |
|---------|----------|----------|
| **ReAct** | Interleaved reason + act | General tool use |
| **Plan-then-Execute** | Full plan first, then execute | Known task structure |
| **Chain-of-Thought** | Reasoning only, no actions | Pure reasoning tasks |
| **Reflexion** | ReAct + self-reflection | Learning from mistakes |

## Common Failure Modes

### 1. Action Loop
```
Thought: I need to search
Action: search("query")
Observation: No results
Thought: I need to search again  ← Stuck!
Action: search("query")
```

**Fix:** Add loop detection, force different approach after N failures.

### 2. Hallucinated Tools
```
Action: query_database("SELECT * FROM users")  ← Tool doesn't exist
```

**Fix:** Clearly document available tools, validate before execution.

### 3. Ignoring Observations
```
Observation: Error: API rate limited
Thought: Great, I have the data  ← Ignored error!
```

**Fix:** Prompt to explicitly acknowledge observation content.

## Enhanced ReAct

```python
class EnhancedReActAgent(ReActAgent):
    def __init__(self, llm, tools, max_retries=3):
        super().__init__(llm, tools)
        self.max_retries = max_retries
        self.action_history = []
    
    def run(self, question: str, max_steps: int = 10) -> str:
        # ... standard run logic ...
        
        # Add loop detection
        if self._is_stuck():
            self._inject_hint("Try a different approach")
        
        # Add observation validation
        if "error" in observation.lower():
            self._inject_hint("The last action failed. Handle the error.")
```

## Files

- `react_pattern.py` - Full ReAct implementation with examples

## Key Takeaways

1. ReAct = Thought → Action → Observation loop
2. Makes agent reasoning visible and debuggable
3. Ground actions in explicit reasoning
4. Watch for loops, hallucinated tools, ignored observations
5. The foundation for most agent frameworks

## What's Next?

Step 3: **Tool Design** — building effective tools for agents.
