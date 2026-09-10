# Step 6: Other Frameworks

## The Landscape

Multiple frameworks exist for building agents. Each has strengths and tradeoffs.

## Framework Comparison

| Framework | Philosophy | Best For | Learning Curve |
|-----------|------------|----------|----------------|
| **LangGraph** | Explicit state machines | Complex control flow | Medium |
| **CrewAI** | Role-based teams | Multi-agent collaboration | Low |
| **AutoGen** | Conversational agents | Research, exploration | Medium |
| **Semantic Kernel** | Enterprise integration | Microsoft ecosystem | Medium |
| **Haystack** | Pipeline-based | RAG + Agents | Medium |

## CrewAI

Role-based agent teams with defined processes.

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert at finding and analyzing tech trends",
    tools=[search_tool, web_scraper],
    llm=llm
)

writer = Agent(
    role="Tech Content Writer",
    goal="Write engaging content about AI discoveries",
    backstory="Skilled at making complex topics accessible",
    tools=[],
    llm=llm
)

# Define tasks
research_task = Task(
    description="Research the latest AI agent frameworks",
    expected_output="Detailed report on top 5 frameworks",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the research",
    expected_output="1000-word blog post",
    agent=writer,
    context=[research_task]  # Depends on research
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  # or Process.hierarchical
)

# Run
result = crew.kickoff()
```

**Strengths:**
- Intuitive role-based design
- Built-in collaboration patterns
- Easy to get started

**Weaknesses:**
- Less control over execution flow
- Limited customization

## AutoGen

Microsoft's framework for conversational multi-agent systems.

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    system_message="You are a helpful AI assistant."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # or "ALWAYS", "NEVER"
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers."
)
```

### Group Chat in AutoGen

```python
from autogen import GroupChat, GroupChatManager

# Multiple agents
coder = AssistantAgent(name="coder", ...)
reviewer = AssistantAgent(name="reviewer", ...)
tester = AssistantAgent(name="tester", ...)

# Group chat
groupchat = GroupChat(
    agents=[user_proxy, coder, reviewer, tester],
    messages=[],
    max_round=12
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="Create a web scraper with tests and code review."
)
```

**Strengths:**
- Great for code generation tasks
- Built-in code execution
- Flexible conversation patterns

**Weaknesses:**
- Can be unpredictable
- Complex debugging

## Semantic Kernel

Microsoft's SDK for AI orchestration, integrates well with .NET.

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Create kernel
kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat"))

# Define plugins (tools)
@kernel.function(name="search")
async def search(query: str) -> str:
    """Search for information."""
    return await search_api(query)

# Create agent
agent = kernel.create_agent(
    name="assistant",
    instructions="You are a helpful research assistant.",
    plugins=["search"]
)

# Run
response = await agent.invoke("Find information about quantum computing")
```

**Strengths:**
- Strong enterprise features
- Good Azure integration
- Planner capabilities

**Weaknesses:**
- Heavier abstraction
- Steeper learning curve

## Haystack

Pipeline-based framework, strong for RAG + Agents.

```python
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

# Build pipeline
pipe = Pipeline()
pipe.add_component("prompt", PromptBuilder(template="..."))
pipe.add_component("llm", OpenAIGenerator())
pipe.add_component("tool_router", ToolRouter())

pipe.connect("prompt", "llm")
pipe.connect("llm", "tool_router")

# Run
result = pipe.run({"query": "What is the weather?"})
```

**Strengths:**
- Great for RAG systems
- Modular, composable
- Good evaluation tools

**Weaknesses:**
- Less agent-native
- Primarily focused on retrieval

## When to Use What

```
Need explicit control flow?        → LangGraph
Want role-based teams quickly?     → CrewAI
Building code generation agent?    → AutoGen
Microsoft ecosystem integration?   → Semantic Kernel
RAG-heavy application?             → Haystack
Learning/prototyping?              → CrewAI or AutoGen
Production, complex logic?         → LangGraph
```

## Rolling Your Own

Sometimes frameworks add overhead. Consider building custom for:
- Simple, well-defined workflows
- Maximum control needed
- Performance-critical applications
- Learning purposes

```python
# Simple custom agent loop
class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def run(self, goal: str) -> str:
        messages = [{"role": "user", "content": goal}]
        
        while True:
            response = self.llm(messages, tools=self.tools)
            
            if response.tool_calls:
                for call in response.tool_calls:
                    result = self.execute(call)
                    messages.append({"role": "tool", "content": result})
            else:
                return response.content
```

## Files

- `framework_comparison.py` - Side-by-side examples

## Key Takeaways

1. No single "best" framework — depends on use case
2. LangGraph for control, CrewAI for simplicity
3. AutoGen excels at code generation
4. Consider building custom for simple cases
5. Evaluate based on your specific needs

## What's Next?

Step 7: **Multi-Agent Architectures** — supervisor, peer-to-peer, hierarchical, and swarm patterns.
