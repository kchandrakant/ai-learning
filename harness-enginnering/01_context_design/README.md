# Module 1: Context Design

## Overview

Context design is the most underappreciated surface in agentic systems. Most practitioners spend hours on prompt wording and minutes on context structure. This is backwards. **The structure of what the model sees—and when it sees it—determines coherence across long tasks more than any individual instruction.**

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Eliminate the orientation tax** by injecting environment maps at session start
2. **Optimize KV-cache hit rates** to reduce costs by up to 10x
3. **Implement the todo.md pattern** to prevent goal drift in long-running tasks
4. **Preserve error evidence** for effective recovery
5. **Write effective instruction files** under 60 lines
6. **Build append-only context managers** that maintain cache coherence

---

## Key Concepts

### 1. The Orientation Tax

**The Problem**: Every time an agent starts a task in an unfamiliar environment, it wastes compute figuring out what's available. It explores directories, probes tools, reads configuration files.

```
Without Environment Map:
Turn 1: list_directory("/")
Turn 2: list_directory("/src")
Turn 3: list_directory("/src/components")
Turn 4: read_file("/src/config.json")
Turn 5: [finally starts actual task]  ← 5 turns wasted

With Environment Map:
Turn 1: [starts actual task immediately]  ← 0 turns wasted
```

**The Solution**: `LocalContextMiddleware`

Inject a workspace map at session start that includes:
- Directory structure (top 3 levels)
- Available tools
- Environment variables (with secrets redacted)

```python
# Conceptual structure
class LocalContextMiddleware:
    def build_context(self, workspace_root: str) -> str:
        return f"""## Environment Map

### Workspace
{self._scan_directory(workspace_root, max_depth=3)}

### Tools
{self._list_tool_names()}

### Environment
{self._safe_env_scan()}  # Redacts secrets
"""
```

### 2. KV-Cache Optimization

**The Economics**: 
- Cached tokens: **$0.30/MTok**
- Uncached tokens: **$3.00/MTok**
- That's a **10x cost differential**

KV-cache works by reusing computed representations of identical token prefixes. Break the prefix—by inserting a timestamp, changing serialization order, or routing to a different server—and you pay full price.

**The Four Rules of KV-Cache Hygiene**:

#### Rule 1: No Second-Precision Timestamps
```python
# ❌ BREAKS CACHE ON EVERY CALL
system_prompt = f"""You are an AI assistant.
Current time: {datetime.now().isoformat()}"""

# ✅ CORRECT - Date-level granularity only
system_prompt = """You are an AI assistant."""
user_message = f"[Session date: {date.today().isoformat()}] {user_query}"
```

#### Rule 2: Append-Only Context
```python
# ❌ BREAKS CACHE - Inserting/reordering
context.insert(0, new_message)
context[5] = modified_message

# ✅ PRESERVES CACHE - Append only
context.append(new_message)
```

#### Rule 3: Deterministic Serialization
```python
# ❌ BREAKS CACHE - Non-deterministic order
json.dumps(data)  # Dict order may vary

# ✅ PRESERVES CACHE - Same bytes every time
json.dumps(data, sort_keys=True, separators=(',', ':'))
```

#### Rule 4: Sticky Session Routing
In distributed deployments, route the same session to the same inference server. A session bouncing between servers loses its cache every call.

### 3. The todo.md Pattern

**The Problem**: Across 50 tool calls, agents drift. This isn't hallucination—it's a structural property of transformer attention. The goal stated at turn 1 gets geometrically diluted by turn 47.

**The Solution**: Recite the goal into the end of the context window—exactly where attention is strongest.

```markdown
## Task Management Protocol

You MUST maintain a todo.md file throughout this task:
1. At task start: Create todo.md with the complete goal and initial subtasks
2. Before every action: Read todo.md, check if you are on track
3. After every completed subtask: Update todo.md to mark it done
4. If uncertain: Re-read todo.md before continuing

The todo.md is your working memory. Keep it current.
```

**The Difference**:
```
Without todo.md:
Turn 1:  Goal: Refactor authentication module
Turn 20: Fixing unrelated CSS in login page
Turn 35: Rewriting the entire user model
Turn 49: Returns "task complete" — auth module untouched

With todo.md:
Turn 1:  Goal: Refactor authentication module → todo.md created
Turn 20: Reads todo.md → recognizes drift → returns to auth module
Turn 49: todo.md shows all subtasks complete → correct completion
```

### 4. Preserve Error Evidence

**Counterintuitive**: When an agent fails, the instinct is to clean up before retrying. Don't.

**Why**: The agent will make the same mistake on the next attempt because it has no information about why the last one failed.

```python
# ❌ WRONG: Clean slate retry
def retry_with_clean_slate(agent, task):
    agent.clear_context()
    agent.delete_failed_outputs()
    return agent.run(task)  # Will likely fail the same way

# ✅ CORRECT: Preserve evidence
def retry_with_error_context(agent, task, failed_attempt):
    error_summary = {
        "failed_at": failed_attempt.last_action,
        "error": failed_attempt.error,
        "files_modified": failed_attempt.modified_files,
        "avoid": "Do not repeat the same approach"
    }
    agent.append_to_context(f"## Previous Attempt Failed\n{json.dumps(error_summary)}")
    return agent.run(task)
```

### 5. The Instruction File: Under 60 Lines

**Research Finding**: LLM-generated instruction files actively hurt performance and cost 20%+ more tokens. Human-written, concise files outperform them.

**The Sweet Spot**: Under 60 lines.

```markdown
# ❌ LLM-Generated (bloated, ~200 lines)
# Agent Instructions

## Overview and Context
You are an AI assistant designed to help with Python backend development tasks.
You have been configured to work with a FastAPI-based web application that uses
SQLAlchemy for database interactions. Your role is to assist developers by...
[continues for 200 lines]

# ✅ Human-Written (30 lines, specific)
# Agent Instructions

## Role
Code review agent for Python backend services.

## Environment
- Python 3.11, FastAPI, SQLAlchemy
- Tests: pytest, run with `pytest tests/`
- Linting: ruff, run with `ruff check .`

## Conventions
- All functions need type annotations
- Error handling: raise specific exceptions, never bare `except`
- Database operations: always use context managers

## When stuck
1. Read the relevant test file first
2. Check existing patterns in similar files
3. If still blocked, add a TODO comment and move on

## What NOT to do
- Never modify migration files
- Never change pyproject.toml without explicit instruction
```

### 6. Append-Only Context Management

Build context managers that never mutate the prefix:

```python
class AppendOnlyContext:
    def __init__(self, system_prompt: str, compact_after: int = 80):
        self.system_prompt = system_prompt  # Never mutated
        self.turns: list[dict] = []
        self.compact_after = compact_after

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        # Never insert, delete, or reorder—append only
        
        if len(self.turns) >= self.compact_after:
            self._compact()

    def _compact(self) -> None:
        """Summarize oldest turns to free context budget."""
        keep_last = 20
        if len(self.turns) <= keep_last:
            return
        
        to_summarize = self.turns[:-keep_last]
        summary = self._summarize(to_summarize)
        
        self.turns = [
            {"role": "system", "content": f"[Summary of {len(to_summarize)} prior turns]\n{summary}"}
        ] + self.turns[-keep_last:]

    def to_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.turns
```

---

## Context Design Checklist

- [ ] Environment map injected at session start
- [ ] No timestamps in system prompt (date-level granularity only)
- [ ] Append-only context management
- [ ] Deterministic JSON serialization (`sort_keys=True`)
- [ ] `todo.md` protocol in system prompt
- [ ] Error evidence preserved on retry
- [ ] Instruction file under 60 lines
- [ ] No redundant context—every paragraph earns its tokens

---

## Exercises

### Exercise 1.1: Build an Environment Map
Create a function that generates an environment map for your workspace:
- Scan directory structure (max 3 levels deep)
- List available tools
- Include environment variables with secrets redacted

### Exercise 1.2: Implement Append-Only Context
Build an `AppendOnlyContext` class that:
- Maintains cache-coherent message history
- Compacts when approaching context limits
- Never mutates the prefix

### Exercise 1.3: Write Your Instruction File
For a task you want to build:
1. Write an instruction file under 60 lines
2. Include: role, environment, conventions, stuck protocols, prohibitions
3. Remove any redundant or obvious information

### Exercise 1.4: todo.md Protocol
Design a `todo.md` protocol for a specific task:
1. Define the initial structure
2. Specify update rules
3. Add the protocol to your system prompt

---

## Metrics to Track

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Orientation turns | 0-1 | Environment map eliminates exploration |
| Cache hit rate | >80% | 10x cost difference |
| Goal drift incidents | 0 | todo.md prevents drift |
| Retry success rate | >80% | Error evidence enables recovery |

---

## Key Takeaways

1. **Structure > Words**: How you organize context matters more than individual word choices

2. **KV-cache is your friend**: Follow the four rules to get 10x cost reduction

3. **todo.md is the model's working memory**: Recite goals to fight attention dilution

4. **Errors are information**: Preserve them for effective recovery

5. **60 lines max**: Concise, human-written instruction files outperform verbose ones

---

## Next Module

Continue to **[Module 2: Tool Selection](../02_tool_selection/README.md)** to learn how to design the agent's capabilities.

---

## References

- Manus. "Context Engineering for AI Agents" - KV-cache economics
- LangChain. "Deep Agents Harness Documentation"
- ETH Zurich. Research on instruction file length and LLM-generated instructions
