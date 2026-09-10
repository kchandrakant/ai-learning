# Module 7: Self-Improving Harnesses

## Overview

The previous modules taught you how to build harnesses. This module teaches you how to build **harnesses that improve themselves**. Self-improving harnesses represent the frontier of harness engineering—systems where the harness becomes an optimization target, not just a static scaffold.

> "Once harness design becomes an executable search space, a strong coding agent can exploit the same design space human engineers use."
> — Lee et al., Meta-Harness (2026)

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand recursive self-improvement (RSI)** and its application to harnesses
2. **Implement context engineering frameworks** like ACE and MCE
3. **Build self-improving loops** with propose-evaluate-accept patterns
4. **Apply evolutionary search** to harness optimization
5. **Design observability-driven harness evolution** systems

---

## Key Concepts

### 1. The Self-Improvement Vision

The concept of recursive self-improvement (RSI) dates back to I.J. Good (1965), who defined an "ultraintelligent machine" as a system that can improve the machinery that produces its intelligence.

**Modern RSI in Harnesses**:
```
┌─────────────────────────────────────────────────────────────┐
│                  SELF-IMPROVEMENT LOOP                       │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Execute  │───▶│ Evaluate │───▶│ Improve  │             │
│   │  Tasks   │    │ Results  │    │ Harness  │             │
│   └──────────┘    └──────────┘    └────┬─────┘             │
│        ▲                               │                    │
│        │                               │                    │
│        └───────────────────────────────┘                    │
│                                                              │
│   The harness improves itself based on execution feedback    │
└─────────────────────────────────────────────────────────────┘
```

**The Progression of What Gets Optimized**:
```
instruction prompts → structured context → workflow → harness code → optimizer code
```

As models become more intelligent, we move toward more complex targets and generic methods.

### 2. Harness Layer vs. Core Intelligence

A key finding from research (Lin et al., 2026):

| Capability | Finding |
|------------|---------|
| **Harness Updating** | Flat across models (9B to Opus) — small models can propose good harness edits |
| **Harness Benefit** | Non-monotonic — middle-tier models benefit most from harness improvements |

**Implication**: Harness improvement enables better deployment, but core intelligence remains the foundation. A weak model cannot improve its own harness effectively (see STOP results with GPT-3.5 vs GPT-4).

### 3. Agentic Context Engineering (ACE)

ACE (Zhang et al., 2025) treats context as an **evolving playbook** rather than a growing prompt.

**Three Components**:

```
┌─────────────────────────────────────────────────────────────┐
│                         ACE FRAMEWORK                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐                                             │
│  │ GENERATOR  │  Produces task trajectories using           │
│  │            │  bullet-point references                    │
│  └─────┬──────┘                                             │
│        │                                                     │
│        ▼                                                     │
│  ┌────────────┐                                             │
│  │ REFLECTOR  │  Distills insights from successful          │
│  │            │  and failed trajectories                    │
│  └─────┬──────┘                                             │
│        │                                                     │
│        ▼                                                     │
│  ┌────────────┐                                             │
│  │  CURATOR   │  Updates structured context with            │
│  │            │  incremental, itemized entries              │
│  └────────────┘                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Choice**: The curator outputs structured, itemized bullets `(identifier, description)` merged with deterministic logic—not full prompt rewrites. This prevents context collapse and brevity bias.

```python
# ACE-style context item
class ContextItem:
    identifier: str      # Unique ID for the insight
    description: str     # The actual insight
    source: str          # Which trajectory it came from
    confidence: float    # How reliable this insight is

class ACECurator:
    def __init__(self):
        self.context_items: dict[str, ContextItem] = {}
    
    def update(self, new_items: list[ContextItem]) -> None:
        """Merge new items with deterministic logic."""
        for item in new_items:
            if item.identifier in self.context_items:
                # Update if higher confidence
                existing = self.context_items[item.identifier]
                if item.confidence > existing.confidence:
                    self.context_items[item.identifier] = item
            else:
                self.context_items[item.identifier] = item
    
    def deduplicate(self) -> None:
        """Periodically remove redundant items."""
        # Semantic deduplication using embeddings
        pass
    
    def to_prompt(self) -> str:
        """Generate prompt section from items."""
        lines = ["## Learned Context"]
        for item in sorted(self.context_items.values(), key=lambda x: -x.confidence):
            lines.append(f"- [{item.identifier}] {item.description}")
        return "\n".join(lines)
```

### 4. Meta Context Engineering (MCE)

MCE (Ye et al., 2026) separates:
- **Mechanism** (how to manage context) — the skill
- **Artifact** (what's in context) — the content

**Bi-Level Optimization**:

```
Inner loop: Find optimal context c* given skill s
    c_s* = argmax J_train(c_s; s)

Outer loop: Find optimal skill s* that maximizes validation performance
    s* = argmax J_val(c_s*)
```

**Skill Definition**: A skill `s` defines a context function `c_s = (ρ_s, F_s)`:
- `ρ_s` = static components (prompts, knowledge bases, code libraries)
- `F_s` = dynamic operators (search, selection, filtering, formatting)

```python
class MCESkill:
    """A skill that defines how to construct context."""
    
    def __init__(self, skill_id: str):
        self.skill_id = skill_id
        self.static_components: dict[str, str] = {}  # ρ_s
        self.dynamic_operators: list[Callable] = []   # F_s
    
    def generate_context(self, task: str, history: list) -> str:
        """Apply the skill to generate context for a task."""
        context_parts = []
        
        # Add static components
        for name, content in self.static_components.items():
            context_parts.append(f"## {name}\n{content}")
        
        # Apply dynamic operators
        for operator in self.dynamic_operators:
            result = operator(task, history)
            if result:
                context_parts.append(result)
        
        return "\n\n".join(context_parts)


class MCEEvolver:
    """Meta-level agent that evolves skills."""
    
    def __init__(self):
        self.skill_history: list[tuple[MCESkill, float, float]] = []  # (skill, train_score, val_score)
    
    def crossover(self, task: str) -> MCESkill:
        """Create new skill via agentic crossover over prior skills."""
        # Select high-performing parent skills
        parents = self._select_parents()
        
        # Generate new skill combining elements from parents
        new_skill = self._generate_child(task, parents)
        
        return new_skill
    
    def _select_parents(self, k: int = 3) -> list[MCESkill]:
        """Select top-k skills by validation score."""
        sorted_skills = sorted(self.skill_history, key=lambda x: -x[2])
        return [s[0] for s in sorted_skills[:k]]
```

### 5. Self-Harness: Propose-Evaluate-Accept Loop

Self-Harness (Zhang et al., 2026) uses a three-stage loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-HARNESS LOOP                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: WEAKNESS MINING                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Execute with current harness h_t                    │   │
│  │ • Collect execution traces                            │   │
│  │ • Cluster failures into verifier-grounded patterns    │   │
│  │ • Identify root causes (not just surface errors)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  Stage 2: HARNESS PROPOSAL                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Provide bounded proposal context:                   │   │
│  │   - Editable surfaces of current harness              │   │
│  │   - Verifier-grounded failure patterns                │   │
│  │   - Records of passing behaviors to preserve          │   │
│  │   - Previously attempted edits                        │   │
│  │ • Generate diverse, distinct edit candidates          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  Stage 3: PROPOSAL VALIDATION                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Evaluate on held-in data (weakness resolved?)       │   │
│  │ • Evaluate on held-out data (regressions?)            │   │
│  │ • Accept only if no regression on both                │   │
│  │ • Merge accepted → h_{t+1}; log rejected              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Failure Record Structure**:
```python
@dataclass
class FailureRecord:
    """Rich failure information for root cause analysis."""
    
    # Surface-level
    terminal_cause: str           # e.g., "timeout", "assertion_error"
    error_message: str
    
    # Causal analysis
    causal_mechanism: str         # What agent behavior caused this?
    abstract_pattern: str         # What general pattern does this expose?
    
    # Context
    task_id: str
    trajectory_summary: str
    relevant_harness_components: list[str]
    
    # Classification
    is_addressable: bool          # Can harness changes fix this?
    is_task_specific: bool        # Or is it inherent task difficulty?
```

### 6. Agentic Harness Engineering (AHE)

AHE (Lin et al., 2026) introduces **three observability pillars**:

#### Pillar 1: Component Observability
Every editable harness component has a file-system representation:

| Component | Description |
|-----------|-------------|
| System prompt | Base instructions |
| Tool description | What tools are available |
| Tool implementation | How tools work |
| Middleware | Hooks around model/tool calls |
| Skill | Reusable capabilities |
| Sub-agent config | Delegation settings |
| Long-term memory | Persistent state |

Each failure pattern maps to one component for targeted edits.

#### Pillar 2: Experience Observability
```
Raw Trajectories (k per harness)
        │
        ▼
Per-Task Analysis Reports (Agent Debugger)
        │
        ▼
Benchmark Overview (Aggregated)
        │
        ▼
Evolve Agent (reads and decides edits)
```

Layered access is more token-efficient than dumping all traces.

#### Pillar 3: Decision Observability
Every edit is a **falsifiable claim** with:
- Failure evidence name
- Inferred root cause
- Targeted fix
- Predicted impact (expected fixes + at-risk regressions)

**Critical Constraints**:
1. Edits only apply to harness workspace (not verifier, not model config)
2. Edits must be evidence-driven with manifesto entries

This prevents reward hacking (e.g., disabling verifier, raising budget).

### 7. Meta-Harness: Optimizing the Optimizer

Meta-Harness (Lee et al., 2026) takes it one level deeper: **a harness for optimizing harnesses**.

```
┌─────────────────────────────────────────────────────────────┐
│                     META-HARNESS LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              HARNESS PROPOSER                        │    │
│  │         (itself a coding agent)                      │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         CANDIDATE HARNESS POOL                       │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │    │
│  │  │   H1   │ │   H2   │ │   H3   │ │   H4   │       │    │
│  │  │ score  │ │ score  │ │ score  │ │ score  │       │    │
│  │  └────────┘ └────────┘ └────────┘ └────────┘       │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PARETO FRONTIER                         │    │
│  │    (best harnesses across multiple objectives)       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: The execution history is accessible via file system. The agent uses `grep` or `cat` to read through it instead of cramming everything into context.

Each proposed harness is a **dictionary in the file system** containing:
- Source code
- Scores
- Rollout trajectories
- State updates

### 8. STOP: Self-Taught Optimizer

STOP (Zelikman et al., 2023) is recursive scaffolding improvement where an improver improves itself.

**The Setup**:
- Improver `I` takes a solution `s`, utility `u`, and model `M`
- Returns improved solution: `s' = I(u, s; M)`

**Meta-Utility**:
```
û(I) = (1/|D|) * E[(u(I(u,s; M)))]  over tasks D
```

**Recursive Update**:
```
I_t = I_{t-1}(û, I_{t-1}; M)
```

The improver uses itself to improve itself!

**Discovered Strategies** (emergently learned):
- Genetic algorithms
- Decomposing and improving parts
- Multi-armed prompt bandits
- Simulated annealing
- Varying temperature
- Beam/tree search

**Cautionary Finding**: STOP improved with GPT-4 but **degraded** with GPT-3.5 and Mixtral. Recursive structure alone isn't enough—the base model must be capable enough.

---

## Workflow Optimization

### ADAS: Automated Design of Agentic Systems

ADAS (Hu et al., 2025) formulates agent design as an optimization problem:

```python
class ADASMetaSearch:
    def __init__(self):
        # Initialize archive with simple agents
        self.archive = [
            CoTAgent(),
            SelfRefineAgent(),
        ]
    
    def search(self, max_iterations: int):
        for _ in range(max_iterations):
            # 1. Meta-agent generates high-level description
            description = self.meta_agent.describe_new_workflow()
            
            # 2. Implement in code
            code = self.meta_agent.implement(description)
            
            # 3. Self-refine for novelty (2 iterations)
            code = self.meta_agent.refine(code, criterion="novelty")
            code = self.meta_agent.refine(code, criterion="novelty")
            
            # 4. Evaluate
            score = self.evaluate(code)
            
            # 5. Add successful candidates to archive
            if score > self.threshold:
                self.archive.append(code)
```

### AFlow: MCTS for Workflow Optimization

AFlow (Zhang et al., 2025) represents workflows as graphs and uses Monte Carlo Tree Search:

```
Workflow = Graph(
    nodes = LLM-invoking actions,
    edges = logical operations in code
)

MCTS Loop:
1. Initialize tree with template workflow W_0
2. Select node (mixture of score + exploration)
3. Expand: LLM proposes modified workflow
4. Execute and evaluate new workflow
5. Add back if improved (within N rounds budget)
6. Repeat until top-k average plateaus
```

**Results**: AFlow outperforms manual methods and ADAS on QA, code, and math tasks.

---

## Implementation Patterns

### Pattern 1: File System as Harness State

```python
class HarnessFileSystem:
    """Harness state persisted in file system for observability."""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        
        # Standard structure
        self.dirs = {
            "harness": workspace / "harness",        # Editable harness code
            "runs": workspace / "runs",              # Execution traces (read-only)
            "proposals": workspace / "proposals",    # Edit proposals
            "accepted": workspace / "accepted",      # Merged edits
            "rejected": workspace / "rejected",      # Failed edits with reasons
        }
    
    def save_harness_component(self, name: str, content: str):
        """Save a harness component (system_prompt, tools, etc.)."""
        path = self.dirs["harness"] / f"{name}.md"
        path.write_text(content)
    
    def log_proposal(self, proposal: dict):
        """Log a harness edit proposal with manifesto."""
        proposal_id = f"prop_{datetime.now().isoformat()}"
        path = self.dirs["proposals"] / f"{proposal_id}.json"
        path.write_text(json.dumps(proposal, indent=2))
        return proposal_id
    
    def accept_proposal(self, proposal_id: str, results: dict):
        """Move accepted proposal and apply edit."""
        src = self.dirs["proposals"] / f"{proposal_id}.json"
        dst = self.dirs["accepted"] / f"{proposal_id}.json"
        
        # Add results to proposal
        proposal = json.loads(src.read_text())
        proposal["results"] = results
        dst.write_text(json.dumps(proposal, indent=2))
        
        # Apply the edit to harness
        self._apply_edit(proposal["edit"])
        
        src.unlink()
```

### Pattern 2: Bounded Proposal Context

```python
class BoundedProposalContext:
    """Context for harness proposals—constrained to prevent reward hacking."""
    
    READ_ONLY = {"runs", "verifier", "model_config"}
    EDITABLE = {"system_prompt", "tools", "middleware", "skills", "memory"}
    
    def build_context(
        self,
        failure_patterns: list[FailureRecord],
        passing_behaviors: list[str],
        previous_attempts: list[dict],
    ) -> str:
        context = []
        
        # 1. Editable surfaces
        context.append("## Editable Harness Components")
        for component in self.EDITABLE:
            content = self.read_component(component)
            context.append(f"### {component}\n```\n{content}\n```")
        
        # 2. Failure patterns (verifier-grounded)
        context.append("## Failure Patterns to Address")
        for fp in failure_patterns:
            if fp.is_addressable and not fp.is_task_specific:
                context.append(f"- {fp.abstract_pattern}: {fp.causal_mechanism}")
        
        # 3. Passing behaviors to preserve
        context.append("## Behaviors to Preserve")
        for behavior in passing_behaviors[:10]:  # Limit for token efficiency
            context.append(f"- {behavior}")
        
        # 4. Previous attempts (avoid repeating failures)
        context.append("## Previously Attempted Edits")
        for attempt in previous_attempts[-5:]:
            context.append(f"- {attempt['edit']}: {attempt['outcome']}")
        
        return "\n\n".join(context)
```

### Pattern 3: Regression-Safe Validation

```python
class RegressionSafeValidator:
    """Validate proposals against held-in and held-out data."""
    
    def __init__(self, held_in: list, held_out: list):
        self.held_in = held_in    # Tasks where weakness should be fixed
        self.held_out = held_out  # Tasks that should not regress
    
    def validate(self, old_harness, new_harness, proposal: dict) -> ValidationResult:
        # Run on held-in: did we fix the weakness?
        held_in_old = self.evaluate(old_harness, self.held_in)
        held_in_new = self.evaluate(new_harness, self.held_in)
        
        improvement = held_in_new.score - held_in_old.score
        weakness_fixed = improvement > 0
        
        # Run on held-out: did we break anything?
        held_out_old = self.evaluate(old_harness, self.held_out)
        held_out_new = self.evaluate(new_harness, self.held_out)
        
        regression = held_out_old.score - held_out_new.score
        has_regression = regression > 0.01  # Small tolerance
        
        # Accept only if fixed and no regression
        accepted = weakness_fixed and not has_regression
        
        return ValidationResult(
            accepted=accepted,
            improvement=improvement,
            regression=regression,
            details={
                "held_in_delta": improvement,
                "held_out_delta": -regression,
                "predicted_impact": proposal.get("predicted_impact"),
            }
        )
```

---

## Safety Considerations

### Reward Hacking Risks

Self-improving harnesses can learn to game their evaluators:

| Risk | Example | Mitigation |
|------|---------|------------|
| Disable verifier | Edit code to skip tests | Verifier is read-only |
| Swap model | Use stronger model for eval | Model config is read-only |
| Raise budget | Increase token/time limits | Budget is read-only |
| Overfit to held-in | Memorize test cases | Use held-out validation |
| Game metrics | Optimize proxy, not goal | Human review at checkpoints |

### Permission Control Pattern

```python
class HarnessPermissions:
    """Strict permission control for self-improving harnesses."""
    
    # What the evolving agent can modify
    EDITABLE = {
        "harness/system_prompt.md",
        "harness/tools/*.py",
        "harness/middleware/*.py",
        "harness/skills/*.md",
        "harness/memory/*.json",
    }
    
    # What must remain untouched
    READ_ONLY = {
        "verifier/*",
        "model_config.yaml",
        "budget_limits.yaml",
        "runs/*",
        "eval_data/*",
    }
    
    def check_edit(self, path: str) -> bool:
        """Reject edits to protected paths."""
        for pattern in self.READ_ONLY:
            if fnmatch(path, pattern):
                return False
        
        for pattern in self.EDITABLE:
            if fnmatch(path, pattern):
                return True
        
        return False  # Default deny
```

---

## Exercises

### Exercise 7.1: Implement ACE Curator
Build an ACE-style curator that:
1. Accepts trajectory results (success/failure)
2. Extracts insights as structured items
3. Merges with existing context using deterministic rules
4. Deduplicates periodically

### Exercise 7.2: Build Self-Harness Loop
Implement a simplified self-harness loop:
1. Weakness mining from execution traces
2. Bounded proposal generation
3. Held-in/held-out validation
4. Accept/reject logic

### Exercise 7.3: Observability Pillars
Design the file-system structure for an AHE-style harness:
1. Component observability (what's editable)
2. Experience observability (layered trace access)
3. Decision observability (edit manifestos)

### Exercise 7.4: Permission Matrix
Create a permission system that:
1. Allows harness edits
2. Blocks verifier/model/budget changes
3. Logs all edit attempts
4. Alerts on suspicious patterns

---

## Key Takeaways

1. **Self-improvement is achievable**: Harnesses can optimize themselves through structured loops

2. **Observability is critical**: Every edit must be traceable, falsifiable, and evidence-driven

3. **Permission control prevents gaming**: The evaluator must live outside the optimization loop

4. **Base model matters**: Recursive improvement requires sufficient base capability

5. **File system is the interface**: Persistent state enables recovery, debugging, and evolution

---

## Next Module

Continue to **[Module 8: Evolutionary Search](../08_evolutionary/README.md)** to learn how evolutionary algorithms optimize harnesses.

---

## References

- Zhang et al. (2025). "Agentic Context Engineering (ACE)"
- Ye et al. (2026). "Meta Context Engineering (MCE)"
- Zhang et al. (2026). "Self-Harness: Harnesses That Improve Themselves"
- Lin et al. (2026). "Agentic Harness Engineering (AHE)"
- Lee et al. (2026). "Meta-Harness: End-to-End Optimization of Model Harnesses"
- Zelikman et al. (2023). "Self-Taught Optimizer (STOP)"
- Hu et al. (2025). "Automated Design of Agentic Systems (ADAS)"
- Zhang et al. (2025). "AFlow: Automating Agentic Workflow Generation"
- Weng, Lilian (2026). "Harness Engineering for Self-Improvement"
