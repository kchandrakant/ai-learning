# Module 6: Case Studies

## Overview

Theory is useful; practice is proof. This module examines real-world harness engineering implementations from leading AI teams. Each case study demonstrates specific principles from previous modules applied at production scale.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Analyze real harness architectures** and identify key design decisions
2. **Apply lessons from Manus** about iterative harness development
3. **Understand GitHub Copilot's** embedding router approach
4. **Extract insights from Claude Code's** leaked architecture
5. **Replicate Vercel's** tool consolidation methodology

---

## Case Study 1: Manus — Four Rebuilds in Six Months

### Background

Manus is one of the most rigorously engineered agentic systems publicly documented. They underwent **four complete architectural rebuilds in six months**—what they called "stochastic gradient descent" at the system level.

### Key Metrics Revealed

| Metric | Value |
|--------|-------|
| Tool calls per task | ~50 |
| Input:output token ratio | ~100:1 |
| Cached token cost | $0.30/MTok |
| Uncached token cost | $3.00/MTok |

### Core Philosophy

> "If model progress is the rising tide, we want Manus to be the boat, not the pillar stuck to the seabed."

**Translation**: Build a harness that floats on any model, not one that depends on specific model quirks.

### Key Innovations

#### 1. todo.md as Working Memory

Manus pioneered the todo.md pattern for goal recitation:

```markdown
# Current Task
Refactor authentication module

## Progress
- [x] Read existing auth.py
- [x] Identify functions to refactor
- [ ] Implement new token validation
- [ ] Update tests
- [ ] Run integration suite

## Current Step
Implementing new token validation in auth.py
```

The agent reads and updates this file **before every action**, ensuring the goal stays in the attention window.

#### 2. Logits Masking for Tool Gating

Rather than changing tool definitions (which breaks KV-cache), Manus uses logits masking to control which tools are available per phase:

```
Planning phase:  [search, read, analyze] available
Building phase:  [write, execute, test] available
Verification:    [test, validate, review] available
```

The tool definitions never change; only the decoding constraints change.

#### 3. Error Evidence Preservation

When tasks fail, Manus preserves the complete error context:

```python
# Not just the error message
error_context = {
    "failed_action": "pytest tests/",
    "error_type": "AssertionError",
    "error_message": "Expected 200, got 404",
    "stack_trace": "...",
    "files_touched": ["api/routes.py", "tests/test_api.py"],
    "hypothesis": "Route path mismatch",
    "avoid": ["Modifying the test assertion", "Ignoring the error"]
}
```

### Lessons Learned

1. **The harness is the durable investment**: Models change; good harness patterns persist
2. **50 tool calls is normal**: Design for multi-step, not one-shot
3. **Cache economics dominate**: 10x cost difference between cached/uncached
4. **Iteration velocity matters**: Be willing to rebuild when patterns don't work

---

## Case Study 2: GitHub Copilot — Embedding Router

### Background

GitHub Copilot needed to support 40+ tools across diverse domains (code navigation, documentation, testing, PR management) without hitting the tool cliff.

### The Problem

- 40 tools = performance degradation
- Tools had meaningful semantic differences (couldn't collapse to bash)
- Needed full capability coverage

### The Solution: Embedding Router

```
┌─────────────────────────────────────────────────────────────┐
│                    COPILOT TOOL ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Task: "Fix the failing test in auth module"                │
│                          │                                   │
│                          ▼                                   │
│                 ┌─────────────────┐                          │
│                 │ Embedding Router│                          │
│                 │ (all-MiniLM)    │                          │
│                 └────────┬────────┘                          │
│                          │                                   │
│    ┌─────────────────────┼─────────────────────┐            │
│    ▼                     ▼                     ▼            │
│ ┌──────────┐      ┌──────────┐          ┌──────────┐       │
│ │ 13 Core  │  +   │ top-5    │    =     │ 18 Total │       │
│ │ Tools    │      │ Dynamic  │          │ Tools    │       │
│ └──────────┘      └──────────┘          └──────────┘       │
│                                                              │
│ Core: file_read, file_write, bash, search, git_status...   │
│ Dynamic: test_run, coverage_report, debug_step...           │
└─────────────────────────────────────────────────────────────┘
```

### Results

| Metric | Static 13 | With Router |
|--------|-----------|-------------|
| Task coverage | 69% | **94.5%** |
| SWE-bench | baseline | **+2-5 points** |
| Latency per call | baseline | **-400ms** |

### Implementation Pattern

```python
class CopilotToolRouter:
    def __init__(self):
        self.core_tools = [
            # Always available
            "file_read", "file_write", "file_search",
            "bash_execute", "git_status", "git_diff",
            "codebase_search", "symbol_lookup",
            "run_tests", "get_diagnostics",
            "explain_code", "suggest_fix", "create_pr"
        ]
        
        self.extended_tools = [
            # Dynamically added based on task
            "coverage_report", "debug_step", "profile_code",
            "dependency_graph", "security_scan", "docs_search",
            "slack_notify", "jira_update", "confluence_search",
            # ... 30+ more
        ]
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._index_tools()
    
    def get_tools(self, task: str) -> list[str]:
        # Embed the task
        task_vec = self.embedder.encode(task)
        
        # Find most relevant extended tools
        scores = cosine_similarity([task_vec], self.tool_embeddings)[0]
        top_5_idx = scores.argsort()[-5:][::-1]
        
        dynamic = [self.extended_tools[i] for i in top_5_idx]
        
        return self.core_tools + dynamic  # Max 18 tools
```

### Lessons Learned

1. **13 core tools handle most cases**: Identify your universal primitives
2. **Dynamic routing preserves capability**: Full coverage without tool cliff
3. **Embedding similarity works**: Simple approach, effective results
4. **Measure task coverage**: 69% → 94.5% is a massive improvement

---

## Case Study 3: Claude Code — The Leaked Architecture

### Background

In April 2026, a debug source map was accidentally bundled into Claude Code npm package v2.1.88, exposing ~512,000 lines of TypeScript—including the system prompt assembly logic.

### What the Leak Revealed

#### 1. Dynamic System Prompt Assembly

The system prompt is **not a static string**. It assembles dynamically from ~40 conditional sections:

```
System Prompt Assembly:
├── Base instructions (always)
├── Tool definitions (~50, each conditional)
├── User preferences (from CLAUDE.md)
├── Workspace context (from repo analysis)
├── Session state (current task, history)
└── Permission constraints (from settings.json)
```

#### 2. Cache Boundary Marker

A key architectural feature: the prompt includes a **cache boundary marker** that separates:
- Globally cacheable content (system instructions, tool definitions)
- Session-specific content (current task, recent history)

```
[GLOBAL CACHEABLE CONTENT]
System instructions...
Tool definitions...
Base permissions...

--- CACHE BOUNDARY ---

[SESSION SPECIFIC]
Current task: Fix the bug in auth.py
Recent actions: [...]
Current state: [...]
```

This maximizes cache hits on the stable prefix while allowing dynamic session content.

#### 3. Aggressive Compaction

The code revealed "about a dozen different methods for compaction, offloading, and summarizing" conversation history. Key insight: **only the 5 most recent function results are kept in context**.

```python
# Simplified version of the compaction logic
def compact_function_results(history: list[dict]) -> list[dict]:
    recent_results = []
    other_turns = []
    
    for turn in reversed(history):
        if turn.get("type") == "function_result":
            if len(recent_results) < 5:
                recent_results.append(turn)
            else:
                # Summarize older results
                other_turns.append(summarize_result(turn))
        else:
            other_turns.append(turn)
    
    return list(reversed(other_turns + recent_results))
```

#### 4. Numeric Constraints

Hard limits, not suggestions:
- ≤25 words between tool calls
- ≤100 words for final responses (unless task demands more)

```
# From the leaked prompt
Between tool calls, keep your response under 25 words.
Final responses should be under 100 words unless the user's
request specifically requires longer output.
```

### Lessons Learned

1. **Dynamic assembly is the norm**: Static prompts are for demos, not production
2. **Cache boundaries are explicit**: Design your prompt with caching in mind
3. **5 recent results is enough**: Aggressive compaction works
4. **Numeric constraints work**: The model respects hard limits

---

## Case Study 4: Vercel — Tool Consolidation

### Background

Vercel's AI team built an agent with 15+ specialized tools. Performance was disappointing. Their solution was radical: collapse to a single bash tool.

### The Journey

#### Before: 15+ Specialized Tools

```python
tools = [
    ReadFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
    RunCommandTool(),
    CheckSyntaxTool(),
    FormatCodeTool(),
    RunTestsTool(),
    InstallPackageTool(),
    SearchInFileTool(),
    CreateDirectoryTool(),
    DeleteFileTool(),
    MoveFileTool(),
    GitStatusTool(),
    GitCommitTool(),
    GitPushTool(),
]
```

**Hypothesis**: Dedicated tools provide clearer affordances.

**Reality**: 15 options fragment attention; model struggles with selection.

#### After: 1 Bash Tool

```python
tools = [
    BashTool(
        allowed_commands=["ls", "cat", "grep", "echo", "mkdir", ...],
        timeout=30,
        sandbox="docker"
    )
]
```

### Results

| Metric | 15+ Tools | 1 Bash Tool | Change |
|--------|-----------|-------------|--------|
| Task time | 274s | 77s | **3.5x faster** |
| Success rate | 80% | 100% | **+20 points** |
| Token usage | baseline | -37% | **37% cheaper** |

### Why It Worked

1. **The model knows bash**: LLMs have extensive training on shell commands
2. **Composition is natural**: `cat file.py | grep "def"` is more natural than separate tools
3. **Fewer decisions**: 1 tool = 0 selection overhead
4. **Universal primitive**: Bash can express any file/process operation

### When NOT to Consolidate

Vercel's approach worked because their tools were **file and process operations**—things bash handles naturally. Don't consolidate:

- Tools with different authentication contexts
- Tools that need different sandboxing
- Tools with complex parameter schemas
- Domain-specific tools (Stripe, GitHub API, etc.)

### Lessons Learned

1. **More tools ≠ more capability**: Often the opposite
2. **Ask "could this be bash?"**: For file/process ops, usually yes
3. **Measure before/after**: The numbers tell the story
4. **Universal primitives win**: Leverage what the model already knows

---

## Case Study 5: LangChain Terminal Bench — The Reasoning Sandwich

### Background

LangChain's Terminal Bench 2.0 evaluation revealed counterintuitive results about reasoning depth allocation.

### The Experiment

Tested three reasoning strategies on the same benchmark:

| Strategy | Description | Score |
|----------|-------------|-------|
| `xhigh` throughout | Maximum reasoning every turn | 53.9% |
| `high` only | Reduced reasoning | 63.6% |
| Reasoning sandwich | xhigh → high → xhigh | **66.5%** |

### The Counterintuitive Finding

**Maximum reasoning was the worst performer.**

Why? Latency spikes cascade into timeouts in time-bounded environments. The extra thinking time at every step doesn't add value—it just burns clock.

### The Sandwich Pattern

```
Task Start
    │
    ▼
┌─────────────────┐
│    PLANNING     │  ← xhigh reasoning (complex decomposition)
│   ~10K tokens   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IMPLEMENTATION │  ← high reasoning (executing known plan)
│   ~5K tokens    │
│   (multiple     │
│    turns)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VERIFICATION   │  ← xhigh reasoning (validating edge cases)
│   ~10K tokens   │
└────────┬────────┘
         │
         ▼
Task Complete
```

### Implementation

```python
class ReasoningSandwich:
    PHASE_CONFIG = {
        "planning": {
            "thinking_budget": 10000,
            "max_turns": 3,
        },
        "implementation": {
            "thinking_budget": 5000,
            "max_turns": 30,
        },
        "verification": {
            "thinking_budget": 10000,
            "max_turns": 5,
        },
    }
    
    def get_config(self, phase: str) -> dict:
        return self.PHASE_CONFIG.get(phase, self.PHASE_CONFIG["implementation"])
```

### Lessons Learned

1. **More thinking ≠ better results**: Diminishing returns are real
2. **Phase-appropriate reasoning**: Match depth to task complexity
3. **Time budgets matter**: Latency spikes cause cascading failures
4. **Measure empirically**: Intuition about "more is better" was wrong

---

## Synthesis: Common Patterns

Across all case studies, several patterns emerge:

### 1. The Model is the Commodity

Every team that achieved strong results focused on the harness, not model selection. Same model, different harness = dramatically different outcomes.

### 2. Cache Economics Dominate

10x cost difference between cached and uncached tokens. Every team that published numbers emphasized cache optimization.

### 3. Tool Count Has a Cliff

GitHub Copilot: 40 → 13 core + dynamic
Vercel: 15 → 1
The pattern: fewer tools, better performance.

### 4. Dynamic Assembly is Standard

No production harness uses static prompts. Context is assembled dynamically based on task, state, and permissions.

### 5. Explicit Constraints Work

Numeric limits (25 words between tools, 5 recent results) are more effective than suggestions. The model respects hard boundaries.

---

## Exercises

### Exercise 6.1: Analyze Your Own Harness

Apply the case study framework to your own agent:
1. What's your tool count? Is there a consolidation opportunity?
2. What's your cache hit rate? Where is the prefix breaking?
3. Are you using phase-appropriate reasoning?
4. Do you have numeric constraints or just suggestions?

### Exercise 6.2: Implement the Sandwich

Take an existing agent and implement the reasoning sandwich:
1. Identify your phases (planning, building, verifying)
2. Assign reasoning budgets to each
3. Measure before/after on your benchmark

### Exercise 6.3: Tool Audit

Perform a Vercel-style tool audit:
1. List all your tools
2. For each, ask: "Could this be bash?"
3. Consolidate where possible
4. Measure the impact

### Exercise 6.4: Cache Boundary Design

Design a cache-aware prompt structure:
1. Identify globally cacheable content
2. Identify session-specific content
3. Place the boundary between them
4. Measure cache hit rate improvement

---

## Key Takeaways

1. **Real teams iterate aggressively**: Manus rebuilt 4 times in 6 months

2. **Embedding routers scale tool access**: GitHub Copilot went from 69% to 94.5% coverage

3. **Dynamic assembly is production standard**: Claude Code assembles from 40+ conditional sections

4. **Tool consolidation works**: Vercel's 15→1 produced 100% success rate

5. **Reasoning has optimal allocation**: The sandwich beats maximum reasoning by 12+ points

---

## What's Next

You've completed the learning path. Here's how to continue:

1. **Apply one pattern this week**: Pick the highest-leverage change for your situation

2. **Measure everything**: You can't improve what you don't measure

3. **Iterate continuously**: The best harnesses are built through repeated refinement

4. **Share your learnings**: The field is young; your discoveries help everyone

---

## References

- Manus. "Context Engineering for AI Agents"
- GitHub. "How We're Making GitHub Copilot Smarter with Fewer Tools"
- Vercel. "We Removed 80% of Our Agent's Tools"
- LangChain. "Evaluating deepagents-cli on Terminal Bench 2.0"
- "How Claude Code Builds a System Prompt" - dbreunig.com (analysis of leaked architecture)
