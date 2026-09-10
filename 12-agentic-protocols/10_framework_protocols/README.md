# Step 10: Framework Protocols

## The Framework Landscape

Each framework defines its own patterns for tools and agents.

## LangChain Tools

```python
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Simple decorator
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

# Structured with Pydantic
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Max results")

search_tool = StructuredTool.from_function(
    func=search_function,
    name="search",
    description="Search the web",
    args_schema=SearchInput
)
```

## LangGraph Agents

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Define state
class AgentState(TypedDict):
    messages: list
    next: str

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges(
    "tools",
    should_continue,
    {"continue": "agent", "end": END}
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"messages": [HumanMessage("Search for X")]})
```

## CrewAI Agents

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with attention to detail",
    tools=[search_tool],
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write compelling content",
    backstory="Experienced technical writer",
    llm=llm
)

# Define tasks
research_task = Task(
    description="Research the topic: {topic}",
    agent=researcher,
    expected_output="Detailed research notes"
)

write_task = Task(
    description="Write article based on research",
    agent=writer,
    expected_output="Polished article",
    context=[research_task]
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

# Run
result = crew.kickoff(inputs={"topic": "AI agents"})
```

## AutoGen Conversations

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a function to calculate fibonacci"
)
```

## Semantic Kernel (Microsoft)

```python
import semantic_kernel as sk
from semantic_kernel.functions import kernel_function

kernel = sk.Kernel()

# Native function
class MathPlugin:
    @kernel_function(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b

kernel.add_plugin(MathPlugin(), "math")

# Prompt function
summarize = kernel.add_function(
    prompt="Summarize: {{$input}}",
    plugin_name="text",
    function_name="summarize"
)

# Use
result = await kernel.invoke(summarize, input="Long text...")
```

## Framework Comparison

| Framework | Paradigm | Strengths | Use Case |
|-----------|----------|-----------|----------|
| LangChain | Composable chains | Flexibility, ecosystem | General purpose |
| LangGraph | State machines | Complex flows | Multi-step agents |
| CrewAI | Role-based crews | Collaboration | Multi-agent teams |
| AutoGen | Conversations | Multi-agent chat | Research, coding |
| Semantic Kernel | Plugins | Enterprise | Microsoft ecosystem |

## Interoperability Challenges

**Problem:** Agent built with CrewAI can't easily call a LangGraph agent.

**Solutions:**
1. Wrap as HTTP API
2. Use MCP as common interface
3. Adopt A2A for agent-to-agent

```python
# Wrap LangGraph agent as MCP server
@mcp.tool()
async def call_langgraph_agent(task: str) -> str:
    result = await langgraph_app.invoke({"messages": [task]})
    return result["messages"][-1].content
```

## Files

- `langchain_tools.py` - LangChain tool patterns
- `langgraph_agent.py` - LangGraph example
- `crewai_example.py` - CrewAI crew
- `interop.py` - Cross-framework integration

## Key Takeaways

1. Each framework has unique patterns
2. LangChain/LangGraph for flexibility
3. CrewAI for role-based collaboration
4. AutoGen for multi-agent conversations
5. Interoperability requires bridges (MCP, A2A)

## What's Next?

Step 11: **Semantic Kernel** — enterprise patterns.
