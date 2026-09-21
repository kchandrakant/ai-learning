# Agent System Design: Beyond the Basics (2024-2027+)

The foundational harness patterns are established, but researchers continue to discover improvements. These advances represent the frontier of harness engineering—where the field is heading.

---

## Overview of Advances

| Evolution | Year | Problem Solved | Key Innovation |
|-----------|------|----------------|----------------|
| Self-Improving Harnesses | 2026 | Manual iteration | Propose-evaluate-accept loops |
| Meta-Harness | 2026 | Optimizing optimizers | Harness that searches harness space |
| Agentic Context Engineering | 2025 | Context collapse | Structured, itemized context playbooks |
| Evolutionary Harness Search | 2025 | Manual design | LLM-guided evolution of harness code |
| Joint Weight+Harness Optimization | 2026 | Separate optimization | Unified improvement loop |
| Continual Harness | 2026 | Offline-only improvement | Online adaptation during deployment |

---

## Evolution 1: Self-Improving Harnesses

### The Problem
Harness iteration is manual: run benchmark → analyze failures → edit harness → repeat. This is slow and doesn't scale.

### The Solution
Automate the loop: the harness proposes edits to itself based on observed failures.

```
┌──────────────────────────────────────────────────────────┐
│                 SELF-HARNESS LOOP                        │
│                                                          │
│   Execute → Mine Weaknesses → Propose Edits → Validate   │
│      ↑                                              │    │
│      └──────────────── Accept/Reject ───────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Key Insight
**Bounded proposal context**: The proposer only sees editable surfaces, failure patterns, passing behaviors, and previous attempts. This prevents reward hacking.

### Papers
- Zhang et al. (2026). "Self-Harness: Harnesses That Improve Themselves"
- Lin et al. (2026). "Agentic Harness Engineering (AHE)"

---

## Evolution 2: Meta-Harness

### The Problem
Self-harness optimizes within a fixed architecture. What if the architecture itself is suboptimal?

### The Solution
A harness that searches over harness architectures—optimizing the optimizer.

```python
# Meta-Harness produces harness candidates on Pareto frontier
meta_harness = MetaHarness(
    objectives=["success_rate", "cost", "latency"],
    search_budget=100,
)
best_harnesses = meta_harness.search(benchmark)
```

### Key Insight
**The proposer is itself a coding agent**. It reads execution history from the file system and edits harness source code directly.

### Papers
- Lee et al. (2026). "Meta-Harness: End-to-End Optimization of Model Harnesses"

---

## Evolution 3: Agentic Context Engineering (ACE)

### The Problem
Context grows unboundedly. Full prompt rewrites cause brevity bias and lose information.

### The Solution
Treat context as an **evolving playbook** of structured items, not a blob.

```python
# ACE uses itemized bullets, not prose
context_items = [
    ("ID001", "Always run tests before completion"),
    ("ID002", "File auth.py contains authentication logic"),
    ("ID003", "Use pytest for testing, not unittest"),
]

# Deterministic merge rules prevent drift
curator.merge(new_items)  # Add/update, never full rewrite
```

### Key Insight
**Deterministic merge logic** prevents context collapse. Items are added/updated, never rewritten wholesale.

### Papers
- Zhang et al. (2025). "Agentic Context Engineering"
- Ye et al. (2026). "Meta Context Engineering (MCE)"

---

## Evolution 4: Evolutionary Harness Search

### The Problem
The design space for harnesses is vast. Manual search is inefficient.

### The Solution
Apply evolutionary algorithms: maintain a population of harness configurations, mutate via LLM, select by fitness.

```
┌──────────────────────────────────────────────────────────┐
│               DARWIN GÖDEL MACHINE                        │
│                                                          │
│  Population of harnesses                                 │
│       ↓                                                  │
│  Select parent (weighted by fitness, diversity)          │
│       ↓                                                  │
│  Parent proposes edits to its own harness code           │
│       ↓                                                  │
│  Evaluate child on benchmark                             │
│       ↓                                                  │
│  Add to population if good enough                        │
└──────────────────────────────────────────────────────────┘
```

### Key Insight
**Self-modification**: The agent examines its own evaluation logs and proposes improvements to its own harness. The harness is the genome being evolved.

### Results
- SWE-bench Verified: 20% → 50% (from simple initial configs)
- Discovers novel patterns human engineers missed

### Papers
- Zhang et al. (2025). "Darwin Gödel Machine"
- Novikov et al. (2025). "AlphaEvolve"
- Hu et al. (2025). "Automated Design of Agentic Systems (ADAS)"

---

## Evolution 5: Joint Weight + Harness Optimization

### The Problem
Harness evolution and model training are separate. But they affect each other—maybe they should be optimized together.

### The Solution
A feedback loop that decides whether to update harness or model weights based on failure analysis.

```python
class SIA:  # Self-Improving AI
    def improve(self, failure):
        if failure.is_harness_addressable():
            self.update_harness(failure)
        else:
            self.update_weights(failure)  # Fine-tune on failure case
```

### Key Insight
**Different failures need different fixes**. Some failures are best addressed by harness changes (tool permissions, prompts); others need model capability improvements.

### Papers
- Hebbar et al. (2026). "SIA: Self-Improving AI with Harness & Weight Updates"

---

## Evolution 6: Continual Harness

### The Problem
Current harnesses improve offline (between deployments). Production data is wasted.

### The Solution
Online adaptation: the harness improves while deployed, learning from real interactions.

```python
class ContinualHarness:
    def on_task_complete(self, trajectory, reward):
        # Low reward? Learn from this failure
        if reward < threshold:
            self.update_from_trajectory(trajectory)
```

### Key Insight
**Incremental, safe updates**: Changes are small and regression-tested against held-out data before deployment.

### Papers
- Karten et al. (2026). "Continual Harness: Online Adaptation for Self-Improving Foundation Agents"

---

## Evolution 7: Standardized Harness Interfaces

### The Problem
Every team builds harnesses differently. Tools, context protocols, and evaluation aren't interoperable.

### The Emerging Solution
Industry standardization of interfaces, similar to how MCP is standardizing tool access.

```
Future Standards (?):
├── Tool Interface Protocol (MCP)
├── Context Interface Protocol
├── Memory Interface Protocol  
├── Evaluation Interface Protocol
└── Safety/Permission Protocol
```

### Key Insight
**Interoperability enables ecosystems**: Shared tool definitions, evaluation benchmarks, and harness components.

### Status
Emerging. MCP is an early example. More standardization expected 2027+.

---

## Open Challenges

These are unsolved problems that future advances will address:

### 1. Fuzzy Evaluation
**Problem**: Many tasks lack clear verifiers. Research taste, code maintainability, long-term impact—how do you score these?

**Current approaches**: LLM-as-judge, human feedback, proxy metrics

**Needed**: Principled evaluation for open-ended tasks

### 2. Memory Lifecycle
**Problem**: When should working memory become long-term? How to forget gracefully?

**Current approaches**: Recency-based pruning, importance scoring

**Needed**: Principled memory consolidation and forgetting

### 3. Reward Hacking
**Problem**: Self-improving harnesses optimize whatever signal they're given. Gaming is possible.

**Current approaches**: Held-out evaluation, permission restrictions, human oversight

**Needed**: Robust evaluation that can't be gamed

### 4. Long-Term Success
**Problem**: Current optimization focuses on immediate task success, not downstream effects (maintainability, tech debt).

**Current approaches**: None—this is mostly ignored

**Needed**: Multi-horizon reward models

### 5. Capability Internalization
**Problem**: Should harness techniques become model capabilities? When?

**Observation**: Just as instruction-following was internalized into models, harness patterns may be too.

**Open question**: What should stay external vs. become model capability?

---

## Modern Harness Architecture Recipe

A cutting-edge harness (2027-style) combines multiple advances:

```python
class ModernHarness:
    def __init__(self, model, tools, benchmark):
        # Foundation: Context engineering (ACE-style)
        self.context = ACEContextManager(
            structured=True,
            merge_rules="deterministic",
        )
        
        # Reliability: Constraints from Module 3
        self.constraints = ConstraintManager(
            reasoning_sandwich=True,
            loop_detection=True,
            verification_gate=True,
        )
        
        # Efficiency: Tool routing from Module 2
        self.tool_router = EmbeddingRouter(
            core_tools=tools[:10],
            extended_tools=tools[10:],
        )
        
        # Self-improvement: From Evolution 1
        self.self_improver = SelfHarnessLoop(
            editable_surfaces=["system_prompt", "tools", "middleware"],
            validation_split=benchmark.split(0.2),
        )
        
        # Observability: From Module 5
        self.metrics = HarnessMetrics()
    
    def run(self, task):
        # Assemble context
        context = self.context.build(task)
        
        # Select tools
        tools = self.tool_router.get_tools(task)
        
        # Execute with constraints
        result = self.execute_with_constraints(context, tools, task)
        
        # Learn from result
        self.self_improver.observe(result)
        
        return result
    
    def evolve(self):
        """Periodic self-improvement cycle"""
        proposals = self.self_improver.propose_edits()
        for proposal in proposals:
            if self.self_improver.validate(proposal):
                self.self_improver.apply(proposal)
```

---

## Summary: Foundational vs. Evolved

| Aspect | Foundational (2024) | Evolved (2027) |
|--------|---------------------|----------------|
| Context | Static prompt | ACE-style structured playbook |
| Improvement | Manual iteration | Self-improving loops |
| Architecture | Fixed design | Evolutionary search |
| Optimization | Harness only | Joint harness + weights |
| Deployment | Offline updates | Continual online learning |
| Interfaces | Custom per team | Standardized protocols |

---

## Files in this Directory

When we implement these advances, they'll be organized as:

```
beyond/
├── self_harness.py           # Self-improving harness loop
├── meta_harness.py           # Meta-level harness optimization
├── ace_context.py            # Agentic Context Engineering
├── evolutionary_search.py    # Darwin Gödel Machine pattern
├── joint_optimization.py     # Harness + weight updates
├── continual_harness.py      # Online adaptation
└── modern_harness.py         # Complete modern architecture
```

---

## References

### Self-Improvement
- Zhang et al. (2026). "Self-Harness: Harnesses That Improve Themselves"
- Lin et al. (2026). "Agentic Harness Engineering (AHE)"
- Lee et al. (2026). "Meta-Harness: End-to-End Optimization of Model Harnesses"

### Context Engineering
- Zhang et al. (2025). "Agentic Context Engineering (ACE)"
- Ye et al. (2026). "Meta Context Engineering (MCE)"

### Evolutionary Search
- Zhang et al. (2025). "Darwin Gödel Machine"
- Novikov et al. (2025). "AlphaEvolve"
- Hu et al. (2025). "Automated Design of Agentic Systems (ADAS)"
- Zhang et al. (2025). "AFlow: Automating Agentic Workflow Generation"

### Joint & Continual
- Hebbar et al. (2026). "SIA: Self-Improving AI with Harness & Weight Updates"
- Karten et al. (2026). "Continual Harness: Online Adaptation"

### Survey
- Weng, Lilian (2026). "Harness Engineering for Self-Improvement"
