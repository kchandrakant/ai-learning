# Agent System Design Learning Path

This guide walks through designing and building production-grade AI agent systems from scratch, starting with foundational patterns, then exploring advanced self-improvement techniques and future directions.

---

## 🎯 What is Agent System Design?

**Agent system design** is the discipline of architecting the complete operating environment that surrounds an AI model—everything except the model weights themselves. This includes tool orchestration, state management, verification loops, policy enforcement, error recovery, and human escalation workflows that transform a language model into a reliable agent.

> "Swap the model for a competitor and output quality shifts 10–15%. Change the system design and you change whether the agent works at all."
> — From "The Three Eras of AI Agent Engineering"

### The Core Insight

The SWE-bench data is unambiguous: a basic scaffold achieves 23% on software engineering tasks. An optimized system running the **same model** achieves 45%+. That's a 22-point swing—the kind of gap that would cost millions in compute and months of research if you tried to close it by switching models.

**The model is a commodity. The system design is the moat.**

---

## The Three Eras of AI Agent Engineering

The progression from prompt engineering to agent system design is nested, not sequential:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT SYSTEM DESIGN                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 CONTEXT ENGINEERING                        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              PROMPT ENGINEERING                      │  │  │
│  │  │  • Task instructions                                 │  │  │
│  │  │  • Role definition                                   │  │  │
│  │  │  • Output format                                     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  • What the model sees                                    │  │
│  │  • Retrieved documents, code, memory                      │  │
│  │  • Dynamic context assembly                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│  • Tool access & sandboxing                                     │
│  • State & memory management                                    │
│  • Verification loops                                           │
│  • Guardrails & approvals                                       │
│  • Observability & recovery                                     │
└─────────────────────────────────────────────────────────────────┘
```

| Era | Focus | Ceiling |
|-----|-------|---------|
| **Prompt Engineering** | Instructions, role, format | One-shot tasks; no tools, no memory |
| **Context Engineering** | What the model sees | RAG, retrieval; still stateless |
| **Agent System Design** | How the agent operates | Production-grade agentic systems |

---

## The Agent Formula

```
Agent = Model + System Design
```

The system design controls:
- **Context Design** — What the model knows before it starts
- **Tool Selection** — Which tools are available and how they're managed
- **Constraint Management** — Time budgets, verification gates, loop detection
- **Production Hardening** — Sandboxing, memory compaction, observability

---

## 📚 Part 1: Core Agent System Components

Core patterns for building reliable agent systems.

### Module 0: Foundations
**Why agent system design matters**

- Why LLMs alone aren't enough for reliable agents
- The five failure modes good system design prevents
- Agent architectures: single-turn vs multi-turn vs long-horizon
- The Martin Fowler framework: Guides + Sensors

---

### Module 1: Context Design
**Engineering what the model sees**

- The orientation tax and how to eliminate it
- KV-cache optimization: the silent 10x cost multiplier
- The `todo.md` pattern: preventing goal drift
- Preserving error evidence for recovery
- Instruction files: keeping them under 60 lines
- Append-only context management

**Key insight:** Structure > words. How you organize context matters more than individual word choices.

---

### Module 2: Tool Selection & Design
**Designing the agent's capabilities**

- The tool cliff effect: why more tools = worse performance
- Tool rationalization: Vercel's 15→1 case study (80% → 100% success)
- Embedding-based tool routing (GitHub Copilot's approach)
- Logits masking for phase-specific tool gating
- Tool audit frameworks

**Key insight:** Ask "could this be a bash command?" for every specialized tool.

---

### Module 3: Constraint Management
**Making agents reliable through boundaries**

- The reasoning sandwich pattern (xhigh → high → xhigh)
- Loop detection middleware
- Self-verification gates
- Time budget injection
- The four-stage structure: Plan → Build → Verify → Fix

**Key insight:** Maximum reasoning everywhere is the WORST strategy. Allocate reasoning depth appropriately.

---

### Module 4: Error Recovery & Resilience
**Handling failures gracefully**

- Error classification: recoverable vs terminal
- Retry strategies with exponential backoff
- Checkpoint and resume patterns
- Graceful degradation strategies
- Fallback chains (model routing on failure)
- Error evidence preservation for debugging

**Key insight:** Plan for failure. Every tool call can fail; every model response can be malformed.

**Recovery patterns:**
```
Try → Fail → Classify error → Retry with fix?
                            → Fallback to simpler approach?
                            → Escalate to human?
                            → Abort gracefully?
```

---

### Module 5: Human-in-the-Loop Design
**Integrating human oversight effectively**

- When to pause for human input
- Approval gates for high-risk actions
- Async approval patterns (deferred execution)
- Human feedback incorporation
- Escalation policies and routing
- UI/UX for human-agent collaboration

**Key insight:** The goal isn't full autonomy—it's appropriate autonomy. Know when to ask.

**Approval patterns:**
```
Pre-approval:   "Can I do X?" → User approves → Agent acts
Post-review:    Agent acts → User reviews → Confirm/Revert
Guardrail:      Agent acts → System checks → Block if risky
```

---

### Module 6: Cost Management & Token Economics
**Making agents economically viable**

- Token counting and cost attribution
- Cost-aware model routing
- Caching strategies (semantic, exact match)
- Context pruning vs quality tradeoffs
- Budget injection ("you have $X remaining")
- Cost dashboards and alerting

**Key insight:** A 10x cheaper solution that's 90% as good often wins.

**Cost levers:**
```
Model selection:     GPT-4 vs GPT-3.5 vs local = 100x cost difference
Context management:  Shorter context = lower cost
Caching:             Cache hits = zero marginal cost
Batching:            Amortize fixed costs
```

---

### Module 7: Testing Agent Systems
**Beyond unit tests: testing non-deterministic systems**

- Unit testing tool implementations
- Integration testing agent flows
- Behavioral testing (assert on actions, not text)
- Trajectory testing (was the path reasonable?)
- Regression testing for agents
- Mocking LLM responses
- Property-based testing for agents

**Key insight:** Test behavior, not text. "Did it call the right tool with right args?" not "Did it say the right words?"

**Testing pyramid for agents:**
```
        /\
       /  \  End-to-end (few, expensive)
      /----\
     /      \ Behavioral evals (core layer)
    /--------\
   /          \ Tool & integration tests
  /------------\
 /              \ Unit tests (many, fast)
```

---

### Module 8: Evaluation & Benchmarking
**Measuring agent system quality**

- End-to-end benchmarks vs behavioral evaluations
- Building behavioral eval suites
- The three-stage evaluation architecture
- Metrics that matter: cache hit rate, error rate, loop detections
- CI/CD integration for agent quality gates
- A/B testing agent systems

**Key insight:** Cost per success is more meaningful than success rate alone.

---

### Module 9: Production Hardening
**Making it work in the real world**

- Sandboxing strategies: allowlist, Docker, cloud sandboxes
- Memory compaction techniques
- Observability instrumentation (traces, metrics, logs)
- Graceful shutdown and state persistence
- Multi-region deployment considerations

**Key insight:** Never run untrusted code without isolation.

---

### Module 10: Case Studies
**Learning from real implementations**

- Manus: Four architectural rebuilds in six months
- GitHub Copilot's agentic harness & embedding router
- Claude Code's architecture (dynamic prompt assembly)
- Vercel's tool consolidation journey (15→1 tools)
- Cursor's context engineering patterns
- LangChain's Terminal Bench optimizations

**Key insight:** The best systems are built through repeated iteration, not upfront design.

---

## 📚 Part 2: Self-Improving Agent Systems

Advanced patterns where agent systems optimize themselves.

### Module 11: Self-Improving Systems
**Building systems that get better automatically**

- Recursive self-improvement (RSI) concepts
- Agentic Context Engineering (ACE): structured playbooks
- Meta Context Engineering (MCE): bi-level optimization
- Self-System: propose-evaluate-accept loops
- Agentic System Engineering (ASE): observability-driven evolution
- Meta-System: optimizing the optimizer
- STOP: Self-Taught Optimizer

**Key insight:** Once system design becomes executable code, strong agents can search the design space.

---

### Module 12: Research Benchmarks
**Evaluating agent system design progress**

- SWE-bench family (Original, Verified, Pro, Live, Multilingual)
- Terminal-Bench for CLI tasks
- PaperBench: research replication
- RE-Bench: research engineering (humans vs AI comparison)
- MLE-bench: ML engineering (Kaggle competitions)
- KernelBench: GPU kernel optimization
- Evaluation protocol design and pitfalls

**Key insight:** Cost per success is more meaningful than success rate alone.

---

## 🚀 Demo: Agent System Design in Action

Hands-on demonstrations bringing together all components.

### Included Demos
1. **Basic Agent System** — Minimal working system
2. **Context Engineering** — Environment maps, todo.md, compaction
3. **Tool Routing** — Embedding-based selection
4. **Constraint Enforcement** — Reasoning sandwich, loop detection
5. **Behavioral Evaluation** — Writing and running evals
6. **Self-Improving System** — Propose-evaluate-accept loop
7. **End-to-End Coding Agent** — Complete implementation

```bash
# Run any demo
python demo/<demo_name>.py --verbose
```

---

## 🔮 Beyond: The Future of Agent System Design

Emerging patterns and future directions (2024-2027+).

### Covered Advances
| Evolution | Year | Key Innovation |
|-----------|------|----------------|
| Self-Improving Systems | 2026 | Propose-evaluate-accept loops |
| Meta-System Design | 2026 | System that searches design space |
| Agentic Context Engineering | 2025 | Structured, itemized playbooks |
| Evolutionary System Search | 2025 | LLM-guided system code evolution |
| Joint Weight+System Optimization | 2026 | Unified improvement loop |
| Continual Systems | 2026 | Online adaptation during deployment |
| Standardized Interfaces | 2027+ | Industry-wide protocols |

### Open Challenges
- Weak and fuzzy evaluators
- Memory lifecycle management
- Reward hacking prevention
- Long-term success optimization
- Capability internalization (what stays in system vs. model?)

---

## 🗂️ Project Structure

```
agent-system-design/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Dependencies
├── verify_setup.py           # Environment verification
│
├── 00_foundations/           # Why agent system design matters
├── 01_context_design/        # KV-cache, todo.md, environment maps
├── 02_tool_selection/        # Tool cliff, embedding routers
├── 03_constraints/           # Reasoning sandwich, loop detection
├── 04_error_recovery/        # Retry strategies, fallbacks, resilience
├── 05_human_in_loop/         # Approval gates, escalation, async consent
├── 06_cost_management/       # Token economics, caching, model routing
├── 07_testing/               # Behavioral tests, trajectory tests, mocking
├── 08_evaluation/            # Benchmarks, metrics, A/B testing
├── 09_production/            # Sandboxing, compaction, observability
├── 10_case_studies/          # Manus, Copilot, Claude Code, Vercel, Cursor
├── 11_self_improving/        # ACE, MCE, Self-System, Meta-System
├── 12_benchmarks/            # SWE-bench, PaperBench, RE-Bench
│
├── demo/                     # Hands-on demonstrations
│   └── README.md
│
└── beyond/               # Future directions & emerging patterns
    └── README.md
```

---

## 📅 Recommended Learning Order

```
Week 1: Foundations
├── Module 0: Foundations
└── Module 1: Context Design

Week 2: Core Mechanics
├── Module 2: Tool Selection & Design
└── Module 3: Constraint Management

Week 3: Resilience
├── Module 4: Error Recovery & Resilience
└── Module 5: Human-in-the-Loop Design

Week 4: Economics & Quality
├── Module 6: Cost Management & Token Economics
└── Module 7: Testing Agent Systems

Week 5: Production
├── Module 8: Evaluation & Benchmarking
├── Module 9: Production Hardening
└── Module 10: Case Studies

Week 6: Advanced
├── Module 11: Self-Improving Systems
├── Module 12: Research Benchmarks
└── Explore beyond/
```

---

## ✅ Prerequisites

- Python 3.10+
- Familiarity with LLM APIs (OpenAI, Anthropic, or similar)
- Basic understanding of async programming
- Git for version control

---

## 🚀 Setup

```bash
# Navigate to the repository
cd agent-system-design

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

---

## 📖 Key References

### Foundational Articles
- [What is Agent System Design?](https://www.designveloper.com/blog/what-is-harness-engineering/) - Designveloper
- [The Definitive Guide to Agent System Design](https://engineeratheart.medium.com/the-definitive-guide-to-agent-harness-engineering-5f5edf25fd73) - Medium
- [From Prompts to Systems: The Three Eras](https://mohamed-hendawy.medium.com/from-prompts-to-harnesses-the-three-eras-of-ai-agent-engineering-fbd0e6168b21) - Medium
- [The Anatomy of Agent System Design](https://developers.googleblog.com/the-anatomy-of-harness-engineering-how-to-evaluate-iterate-and-guard-ai-coding-agents/) - Google Developers
- [Agent System Design for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) - Lilian Weng (Lil'Log)

### Implementation Resources
- [Improving Deep Agents with System Design](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering) - LangChain
- [Building Reliable Agents with Memory and Compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction) - OpenAI
- [Agent System Design](https://www.oreilly.com/radar/agent-harness-engineering/) - O'Reilly

### GitHub Resources
- [awesome-agent-system-design](https://github.com/ai-boost/awesome-harness-engineering) - Curated list
- [agent-system-design-guide](https://github.com/nexu-io/harness-engineering-guide) - Open guide
- [benchmark-systems](https://github.com/strands-labs/benchmark-harnesses) - Benchmark implementations

---

## Attribution

This learning path synthesizes insights from ThoughtWorks (Martin Fowler), Mitchell Hashimoto, Anthropic, OpenAI, LangChain, Manus, Vercel, GitHub Copilot teams, and Lilian Weng's comprehensive survey. Content was rephrased for compliance with licensing restrictions.

---

## 🎯 Let's Begin!

When you're ready, start with **Module 0: Foundations**.

I'll explain the concepts, then we'll implement them together—you can ask questions, suggest changes, and we'll make sure you understand each piece before moving on.

```
📁 00_foundations/README.md  ← Start here
```
