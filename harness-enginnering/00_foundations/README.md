# Module 0: Foundations

## Overview

Before building harnesses, you need to understand *why* they exist. This module explores the fundamental limitations of LLMs that make harness engineering necessary, and introduces the core concepts that underpin everything that follows.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain why LLMs alone aren't sufficient** for reliable autonomous agents
2. **Identify the five failure modes** that harnesses are designed to prevent
3. **Distinguish between** prompt engineering, context engineering, and harness engineering
4. **Describe the components** of a complete agent harness
5. **Understand the agent formula**: `Agent = Model + Harness`

---

## Key Concepts

### Why Agents Fail Without Harnesses

LLMs have fundamental properties that make them unreliable for autonomous work:

| Property | Problem | Harness Solution |
|----------|---------|------------------|
| **Stateless** | No memory between calls | State management, memory systems |
| **Non-deterministic** | Different outputs for same input | Verification gates, assertions |
| **Context-limited** | Fixed window size | Compaction, progressive disclosure |
| **Overconfident** | Claims completion prematurely | Pre-completion checklists |
| **No self-evaluation** | Can't verify own work | External validation loops |

### The Five Agent Failure Modes

Without a harness, agents fail in predictable ways:

1. **One-shot failure**: Agent tries to complete everything in one context window, runs out of tokens mid-task, leaves work half-built

2. **Premature success declaration**: Agent claims "done!" without verifying anything works

3. **Context rot**: Window fills with tool outputs and history; model loses sight of original instructions

4. **Hallucinated tool calls**: Agent calls functions with wrong parameter types or references non-existent APIs

5. **Lost state on failure**: Network timeout or error wipes progress; next session starts from zero

### The Three Eras Model

```
Era 1: Prompt Engineering
├── What: Instructions, role, format, constraints
├── Solves: One-shot tasks, formatting, tone
└── Ceiling: No tools, no memory, no multi-step

Era 2: Context Engineering  
├── What: Dynamic context assembly, RAG, retrieval
├── Solves: Grounding in knowledge, relevant information
└── Ceiling: Still stateless, no verification, no recovery

Era 3: Harness Engineering
├── What: Tools, state, verification, safety, recovery
├── Solves: Production-grade autonomous operation
└── Ceiling: [Current frontier of research]
```

### What's IN the Harness

The harness encompasses everything except the model:

```
┌─────────────────────────────────────────────────────────────┐
│                         HARNESS                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Context    │  │    Tools     │  │    State     │       │
│  │  Assembly    │  │  & Sandbox   │  │  Management  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Verification │  │  Guardrails  │  │ Observability│       │
│  │    Loops     │  │  & Approvals │  │   & Logs     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │   Recovery   │  │    Budget    │                         │
│  │    Paths     │  │   Controls   │                         │
│  └──────────────┘  └──────────────┘                         │
│                                                              │
│                    ┌──────────────┐                          │
│                    │    MODEL     │                          │
│                    │  (commodity) │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## The Evidence: Same Model, Different Results

The most compelling case for harness engineering comes from benchmark data:

### SWE-bench Results
- Basic scaffold: **23%** success rate
- Optimized 250-turn scaffold: **45%+** success rate
- Same model, 22-point difference

### LangChain Terminal Bench
- Team moved from Top 30 to Top 5
- Jump from **52.8%** to **66.5%**
- No model changes—only harness improvements

### Vercel Case Study
- Collapsed 15+ tools to 1 bash tool
- Task completion: 274 seconds → 77 seconds
- Success rate: 80% → 100%
- Token usage: 37% reduction

---

## Harness vs. Benchmark Runner

A common confusion: evaluation harnesses vs. benchmark runners.

| Aspect | Benchmark Runner | Evaluation Harness |
|--------|------------------|-------------------|
| **Evaluates** | A model | A system (agent, RAG, workflow) |
| **Inputs** | Static academic datasets | Live traces, production slices |
| **Scoring** | Accuracy vs. ground truth | Multi-method: LLM-judge, code, embeddings |
| **Unit of work** | Single prompt/response | Spans, traces, trajectories, sessions |
| **Closes the loop** | No. Produces a report. | Yes. Triggers alerts, queues, gates. |

---

## The Martin Fowler Framework: Guides + Sensors

ThoughtWorks provides a useful framework for thinking about harness controls:

### Guides (Feedforward)
Controls that constrain behavior *before* execution:
- System prompts with explicit rules
- Tool permission matrices
- Pre-completion checklists
- Response format specifications

### Sensors (Feedback)
Controls that evaluate behavior *after* execution:
- Test suites
- LLM-as-judge evaluation
- Output validation
- Metric tracking

### Key Insight
> You need both. Feedback-only produces agents that keep repeating mistakes. Feedforward-only encodes rules but never verifies they worked. Together, they form a steering loop.

Each control can be:
- **Computational** (deterministic): tests, linters, type checkers—fast, cheap, reliable
- **Inferential** (semantic): AI code review, LLM-as-judge—slower, expensive, richer

---

## Exercises

### Exercise 0.1: Identify the Failure Mode
For each scenario below, identify which of the five failure modes is occurring:

1. An agent is asked to refactor a module. After 40 turns, it has rewritten the entire user model instead of the authentication module it was supposed to fix.

2. An agent runs a database migration, reports "Migration complete!", but the migration actually failed due to a network timeout.

3. An agent tries to call `stripe.refunds.create()` with a customer_id that doesn't exist because it pulled the ID from a similarly-named field.

4. An agent crashes mid-task after completing step 2 of 5. When restarted, it begins from step 1 again.

5. An agent is given a complex task. By turn 47, its context is full of tool outputs, and it has lost track of the original goal.

### Exercise 0.2: Classify the Control
For each harness feature, classify it as Guide or Sensor, and Computational or Inferential:

1. A linter that blocks commits with syntax errors
2. A system prompt rule: "Always run tests before marking complete"
3. An LLM-as-judge that evaluates code readability
4. A permission matrix that blocks destructive database operations
5. A test suite that validates the agent's output

### Exercise 0.3: Map the Problem
Think of an agent task you want to build (or have built). List:
1. The potential failure modes for this specific task
2. What guides could prevent each failure
3. What sensors could detect each failure

---

## Key Takeaways

1. **LLMs have structural limitations** that make them unreliable for autonomous work without external systems

2. **The harness is everything except the model**: tools, state, verification, safety, recovery, observability

3. **Same model + better harness = dramatically better results** (22-point swings on benchmarks)

4. **Harness controls are both feedforward (guides) and feedback (sensors)**—you need both

5. **The model is a commodity; the harness is the moat**

---

## Next Module

Continue to **[Module 1: Context Design](../01_context_design/README.md)** to learn how to engineer what the model sees.

---

## References

- Fowler, M. & Boeckeler, B. (2026). "Harness Engineering" - ThoughtWorks
- Hashimoto, M. (2026). "My AI Adoption Journey" - Origin of the term "harness engineering"
- LangChain. "Improving Deep Agents with Harness Engineering"
- OpenAI. "A Practical Guide to Building AI Agents"
