# Harness Engineering Learning Path

This guide walks through building and understanding AI agent harnesses from scratch, starting with foundational patterns, then exploring advanced self-improvement techniques and future evolutions.

---

## 🎯 What is Harness Engineering?

**Harness engineering** is the discipline of designing the operating layer that surrounds an AI agent—everything except the model itself. The harness gives an agent the ability to use tools, preserve state, verify work, follow policies, recover from errors, and escalate risky actions instead of only generating text.

> "Swap the model for a competitor and output quality shifts 10–15%. Change the harness and you change whether the system works at all."
> — From "The Three Eras of AI Agent Engineering"

### The Core Insight

The SWE-bench data is unambiguous: a basic scaffold achieves 23% on software engineering tasks. An optimized scaffold running the **same model** achieves 45%+. That's a 22-point swing—the kind of gap that would cost millions in compute and months of research if you tried to close it by switching models.

**The model is a commodity. The harness is the moat.**

---

## The Three Eras of AI Agent Engineering

The progression from prompt engineering to harness engineering is nested, not sequential:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARNESS ENGINEERING                          │
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
| **Harness Engineering** | How the agent operates | Production-grade agentic systems |

---

## The Agent Formula

```
Agent = Model + Harness
```

The harness controls:
- **Context Design** — What the model knows before it starts
- **Tool Selection** — Which tools are available and how they're managed
- **Constraint Management** — Time budgets, verification gates, loop detection
- **Production Hardening** — Sandboxing, memory compaction, observability

---

## 📚 Part 1: The Foundational Harness

Core patterns for building reliable agent harnesses.

### Module 0: Foundations
**Why harness engineering matters**

- Why LLMs alone aren't enough for reliable agents
- The five failure modes harnesses prevent
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

### Module 2: Tool Selection
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

### Module 4: Evaluation Harness
**Testing and benchmarking agent systems**

- End-to-end benchmarks vs behavioral evaluations
- Building behavioral eval suites
- The three-stage evaluation architecture
- Metrics that matter: cache hit rate, error rate, loop detections
- CI/CD integration for agent quality gates

**Key insight:** Test behavior, not text. Assert on tool calls and actions.

---

### Module 5: Production Hardening
**Making it work in the real world**

- Sandboxing strategies: allowlist, Docker, cloud sandboxes
- Memory compaction techniques
- Observability instrumentation
- Recovery mechanisms and graceful degradation
- Cost management and token economics

**Key insight:** Never run untrusted code without isolation.

---

### Module 6: Case Studies
**Learning from real implementations**

- Manus: Four architectural rebuilds in six months
- GitHub Copilot's agentic harness & embedding router
- Claude Code's leaked architecture (dynamic prompt assembly)
- Vercel's tool consolidation journey
- LangChain's Terminal Bench optimizations

**Key insight:** The best harnesses are built through repeated iteration, not upfront design.

---

## 📚 Part 2: Self-Improving Harnesses

Advanced patterns where harnesses optimize themselves.

### Module 7: Self-Improving Harnesses
**Building harnesses that get better automatically**

- Recursive self-improvement (RSI) concepts
- Agentic Context Engineering (ACE): structured playbooks
- Meta Context Engineering (MCE): bi-level optimization
- Self-Harness: propose-evaluate-accept loops
- Agentic Harness Engineering (AHE): observability-driven evolution
- Meta-Harness: optimizing the optimizer
- STOP: Self-Taught Optimizer

**Key insight:** Once harness design becomes executable code, strong agents can search the design space.

---

### Module 8: Research Benchmarks
**Evaluating harness engineering progress**

- SWE-bench family (Original, Verified, Pro, Live, Multilingual)
- Terminal-Bench for CLI tasks
- PaperBench: research replication
- RE-Bench: research engineering (humans vs AI comparison)
- MLE-bench: ML engineering (Kaggle competitions)
- KernelBench: GPU kernel optimization
- Evaluation protocol design and pitfalls

**Key insight:** Cost per success is more meaningful than success rate alone.

---

## 🚀 Demo: Harness Engineering in Action

Hands-on demonstrations bringing together all components.

### Included Demos
1. **Basic Agent Harness** — Minimal working harness
2. **Context Engineering** — Environment maps, todo.md, compaction
3. **Tool Routing** — Embedding-based selection
4. **Constraint Enforcement** — Reasoning sandwich, loop detection
5. **Behavioral Evaluation** — Writing and running evals
6. **Self-Improving Harness** — Propose-evaluate-accept loop
7. **End-to-End Coding Agent** — Complete implementation

```bash
# Run any demo
python demo/<demo_name>.py --verbose
```

---

## 🔮 Evolutions: The Future of Harness Engineering

Emerging patterns and future directions (2024-2027+).

### Covered Evolutions
| Evolution | Year | Key Innovation |
|-----------|------|----------------|
| Self-Improving Harnesses | 2026 | Propose-evaluate-accept loops |
| Meta-Harness | 2026 | Harness that searches harness space |
| Agentic Context Engineering | 2025 | Structured, itemized playbooks |
| Evolutionary Harness Search | 2025 | LLM-guided harness code evolution |
| Joint Weight+Harness Optimization | 2026 | Unified improvement loop |
| Continual Harness | 2026 | Online adaptation during deployment |
| Standardized Interfaces | 2027+ | Industry-wide protocols |

### Open Challenges
- Weak and fuzzy evaluators
- Memory lifecycle management
- Reward hacking prevention
- Long-term success optimization
- Capability internalization (what stays in harness vs. model?)

---

## 🗂️ Project Structure

```
harness-engineering/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Dependencies
├── verify_setup.py           # Environment verification
│
├── 00_foundations/           # Why harness engineering matters
├── 01_context_design/        # KV-cache, todo.md, environment maps
├── 02_tool_selection/        # Tool cliff, embedding routers
├── 03_constraints/           # Reasoning sandwich, loop detection
├── 04_evaluation/            # Behavioral evals, metrics
├── 05_production/            # Sandboxing, compaction, observability
├── 06_case_studies/          # Manus, Copilot, Claude Code, Vercel
├── 07_self_improving/        # ACE, MCE, Self-Harness, Meta-Harness
├── 08_benchmarks/            # SWE-bench, PaperBench, RE-Bench
│
├── demo/                     # Hands-on demonstrations
│   └── README.md
│
└── evolutions/               # Future directions & emerging patterns
    └── README.md
```

---

## 📅 Recommended Learning Order

```
Week 1: Foundations
├── Module 0: Foundations
└── Module 1: Context Design

Week 2: Core Mechanics
├── Module 2: Tool Selection
└── Module 3: Constraint Management

Week 3: Reliability
├── Module 4: Evaluation Harness
└── Module 5: Production Hardening

Week 4: Real World
├── Module 6: Case Studies
└── Run demos 1-5

Week 5: Advanced
├── Module 7: Self-Improving Harnesses
└── Run demo 6 (self-improvement)

Week 6: Research Frontier
├── Module 8: Research Benchmarks
└── Explore evolutions/
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
cd harness-engineering

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
- [What is Harness Engineering?](https://www.designveloper.com/blog/what-is-harness-engineering/) - Designveloper
- [The Definitive Guide to Agent Harness Engineering](https://engineeratheart.medium.com/the-definitive-guide-to-agent-harness-engineering-5f5edf25fd73) - Medium
- [From Prompts to Harnesses: The Three Eras](https://mohamed-hendawy.medium.com/from-prompts-to-harnesses-the-three-eras-of-ai-agent-engineering-fbd0e6168b21) - Medium
- [The Anatomy of Harness Engineering](https://developers.googleblog.com/the-anatomy-of-harness-engineering-how-to-evaluate-iterate-and-guard-ai-coding-agents/) - Google Developers
- [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) - Lilian Weng (Lil'Log)

### Implementation Resources
- [Improving Deep Agents with Harness Engineering](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering) - LangChain
- [Building Reliable Agents with Memory and Compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction) - OpenAI
- [Agent Harness Engineering](https://www.oreilly.com/radar/agent-harness-engineering/) - O'Reilly

### GitHub Resources
- [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) - Curated list
- [harness-engineering-guide](https://github.com/nexu-io/harness-engineering-guide) - Open guide
- [benchmark-harnesses](https://github.com/strands-labs/benchmark-harnesses) - Benchmark implementations

---

## Attribution

This learning path synthesizes insights from ThoughtWorks (Martin Fowler), Mitchell Hashimoto (origin of the term "harness engineering"), Anthropic, OpenAI, LangChain, Manus, Vercel, GitHub Copilot teams, and Lilian Weng's comprehensive survey. Content was rephrased for compliance with licensing restrictions.

---

## 🎯 Let's Begin!

When you're ready, start with **Module 0: Foundations**.

I'll explain the concepts, then we'll implement them together—you can ask questions, suggest changes, and we'll make sure you understand each piece before moving on.

```
📁 00_foundations/README.md  ← Start here
```
