# Step 5: LangGraph Deep Dive

## What is LangGraph?

LangGraph is a framework for building stateful, multi-step agent applications using graph-based control flow.

**Key insight:** Agents are state machines. LangGraph makes that explicit.

## Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Concepts                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Graph = Nodes + Edges + State                             │
│                                                              │
│   Nodes:  Functions that transform state                    │
│   Edges:  Transitions between nodes                         │
│   State:  Shared data structure passed through graph        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Basic Graph Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Append-only
    next_step: str

# 2. Define Nodes (functions)
def call_llm(state: AgentState) -> AgentState:
    """Node that calls the LLM."""
    response = llm(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState) -> AgentState:
    """Node that executes a tool."""
    last_message = state["messages"][-1]
    result = execute_tool(last_message.tool_calls[0])
    return {"messages": [result]}

# 3. Build Graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("llm", call_llm)
graph.add_node("tool", call_tool)

# Add edges
graph.add_edge("tool", "llm")  # After tool, go to LLM

# Add conditional edge
def should_call_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    return END

graph.add_conditional_edges("llm", should_call_tool)

# Set entry point
graph.set_entry_point("llm")

# Compile
app = graph.compile()
```

## Visualizing the Graph

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
           ┌───────│   LLM   │◀──────┐
           │       └────┬────┘       │
           │            │            │
           │   has_tool_call?        │
           │      │         │        │
           │     YES        NO       │
           │      │         │        │
           │      ▼         ▼        │
           │ ┌─────────┐  ┌─────┐   │
           └─│  Tool   │  │ END │   │
             └────┬────┘  └─────┘   │
                  │                  │
                  └──────────────────┘
```

## State Management

### Reducers
Control how state updates are merged:

```python
from typing import Annotated
import operator

class AgentState(TypedDict):
    # Append new items to list
    messages: Annotated[list, operator.add]
    
    # Overwrite with latest value
    current_step: str
    
    # Custom reducer
    tool_calls: Annotated[list, merge_tool_calls]

def merge_tool_calls(existing: list, new: list) -> list:
    """Custom reducer that deduplicates."""
    seen = {tc.id for tc in existing}
    return existing + [tc for tc in new if tc.id not in seen]
```

## Conditional Routing

```python
def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent")
    
    if intent == "search":
        return "search_node"
    elif intent == "calculate":
        return "calculator_node"
    elif intent == "chitchat":
        return "chitchat_node"
    else:
        return "clarify_node"

graph.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "search_node": "search",
        "calculator_node": "calculator",
        "chitchat_node": "chitchat",
        "clarify_node": "clarify"
    }
)
```

## Human-in-the-Loop

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add persistence for human approval
memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory, interrupt_before=["dangerous_action"])

# Run until interrupt
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config)

if result.get("__interrupt__"):
    # Ask human for approval
    approved = get_human_approval(result)
    
    if approved:
        # Continue from checkpoint
        result = app.invoke(None, config)
    else:
        # Cancel
        pass
```

## Subgraphs

Compose complex agents from simpler graphs:

```python
# Research subgraph
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_web)
research_graph.add_node("summarize", summarize_results)
# ... build research graph ...
research_app = research_graph.compile()

# Main graph uses subgraph as a node
main_graph = StateGraph(MainState)

def research_node(state: MainState) -> MainState:
    """Node that runs the research subgraph."""
    research_result = research_app.invoke({
        "query": state["research_query"]
    })
    return {"research_summary": research_result["summary"]}

main_graph.add_node("research", research_node)
```

## Error Handling

```python
def safe_tool_call(state: AgentState) -> AgentState:
    """Tool node with error handling."""
    try:
        result = execute_tool(state["pending_tool"])
        return {
            "messages": [result],
            "error": None
        }
    except ToolError as e:
        return {
            "messages": [f"Tool error: {e}"],
            "error": str(e)
        }

def route_after_tool(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "llm"

graph.add_conditional_edges("tool", route_after_tool)
```

## Complete ReAct Agent in LangGraph

```python
from langgraph.prebuilt import create_react_agent

# Simple way
tools = [search_tool, calculator_tool]
agent = create_react_agent(llm, tools)

# Or build custom
class ReActState(TypedDict):
    messages: Annotated[list, operator.add]
    
def call_model(state: ReActState):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: ReActState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: ReActState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(ReActState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
```

## Files

- `langgraph_basics.py` - LangGraph fundamentals
- `langgraph_react.py` - ReAct agent with LangGraph

## Key Takeaways

1. LangGraph makes agent control flow explicit
2. State = shared data, Nodes = functions, Edges = transitions
3. Conditional edges enable dynamic routing
4. Checkpointing enables human-in-the-loop
5. Subgraphs compose complex agents from simple ones

## What's Next?

Step 6: **Other Frameworks** — comparing CrewAI, AutoGen, and more.
