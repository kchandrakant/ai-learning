# Module 3: Constraint Management

## Overview

Constraints sound like limitations. In harness engineering, **they are performance architecture**. This module teaches you how to use constraints strategically to improve agent reliability, prevent failures, and optimize costs.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement the reasoning sandwich pattern** to optimize thinking depth allocation
2. **Build loop detection middleware** to catch and break infinite retry cycles
3. **Create self-verification gates** to prevent premature completion
4. **Inject time budget warnings** to enable graceful degradation
5. **Apply the four-stage structure** (Plan → Build → Verify → Fix)

---

## Key Concepts

### 1. The Reasoning Sandwich Pattern

**The Counterintuitive Finding**: Maximum reasoning throughout **hurts** performance.

LangChain's Terminal Bench 2.0 data:

| Reasoning Strategy | Score |
|-------------------|-------|
| `xhigh` throughout (maximum reasoning) | 53.9% |
| `high` only (reduced reasoning) | 63.6% |
| **Reasoning Sandwich** (xhigh → high → xhigh) | **66.5%** |

Maximum reasoning everywhere = **worst result**.

**Why**: Maxing reasoning at every step creates latency spikes that cascade into task failures, especially in time-bounded environments.

**The Sandwich**:
- **xhigh** reasoning at **Planning** phase (complex problem decomposition)
- **high** reasoning at **Implementation** phase (executing known approach)
- **xhigh** reasoning at **Verification** phase (validating correctness, edge cases)

```python
class ReasoningSandwichScheduler:
    PHASE_REASONING = {
        "planning":       "xhigh",  # Complex problem decomposition
        "discovery":      "xhigh",  # Understanding codebase/environment
        "implementation": "high",   # Executing known approach
        "testing":        "high",   # Running and checking tests
        "verification":   "xhigh",  # Validating correctness, edge cases
        "completion":     "high",   # Writing summaries, cleanup
    }

    def get_reasoning_level(self, current_phase: str) -> str:
        return self.PHASE_REASONING.get(current_phase, "high")

    def build_api_params(self, current_phase: str) -> dict:
        level = self.get_reasoning_level(current_phase)
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000 if level == "xhigh" else 5000
            }
        }
```

**The Difference**:
```
Without Sandwich (xhigh everywhere):
Planning:        xhigh (good)
File read 1:     xhigh (overkill, +800ms)
File read 2:     xhigh (overkill, +800ms)
Code edit 1:     xhigh (overkill, +800ms)
Timeout rate: 46%

With Sandwich:
Planning:        xhigh (worth it)
File reads:      high  (fast, sufficient)
Code edits:      high  (executing known plan)
Verification:    xhigh (worth it)
Timeout rate: 12%
Success rate:    66.5%
```

### 2. Loop Detection Middleware

**The Problem**: Production agents loop. The mechanism varies:
- Edit-test-edit-test cycle that's not converging
- Repeatedly tries the same broken approach
- Without detection, runs until budget exhaustion

**The Pattern**:

```python
class LoopDetectionMiddleware:
    def __init__(
        self,
        window: int = 5,
        threshold: int = 3,
        expected_multi_edit: set[str] | None = None
    ):
        self.window = window
        self.threshold = threshold
        self.action_history: list[dict] = []
        # Files where multiple edits are expected (e.g., TDD cycles)
        self.expected_multi_edit = expected_multi_edit or set()

    def check_and_intervene(self, current_action: dict) -> tuple[bool, str]:
        self.action_history.append(current_action)
        
        if len(self.action_history) < self.window:
            return False, ""
        
        recent = self.action_history[-self.window:]
        target_files = [a.get("file_path") for a in recent if "file_path" in a]
        
        for f in set(target_files):
            if f in self.expected_multi_edit:
                continue  # Normal TDD pattern - not a loop
            
            if target_files.count(f) >= self.threshold:
                return True, f"""
You have edited {f} {target_files.count(f)} times without progress.
Stop. Re-read todo.md. Try a different approach.
"""
        
        return False, ""
```

**The Difference**:
```
Without Loop Detection:
Turn 22: edit auth.py — test fails (TypeError line 47)
Turn 23: edit auth.py — test fails (TypeError line 47)
Turn 24: edit auth.py — test fails (TypeError line 47)
Turn 25: edit auth.py — test fails (same error)
...
Turn 50: budget exhausted — task incomplete

With Loop Detection:
Turn 22: edit auth.py — test fails
Turn 23: edit auth.py — test fails
Turn 24: loop detected → intervention: "re-read todo.md, try different approach"
Turn 25: agent reads error carefully → identifies root cause → different fix
Turn 28: tests pass
```

### 3. Self-Verification Gates

**The Problem**: Agents declare completion too early. The model reaches a state that *looks* like completion from the perspective of the last few turns and exits.

**The Solution**: Force a dedicated verification pass before exit.

```python
PRE_COMPLETION_CHECKLIST = """
Before marking this task complete, you MUST verify:

## Functional Verification
- [ ] Run all relevant tests: `pytest tests/ -v`
- [ ] Verify the primary happy path works end-to-end
- [ ] Test at least two edge cases:
  - Empty/null inputs
  - Boundary values
  - Error conditions

## Code Quality
- [ ] No syntax errors (run `ruff check .`)
- [ ] Type annotations present on all new functions
- [ ] No hardcoded values that should be constants

## Completeness
- [ ] Every item in todo.md is marked done or explicitly deferred
- [ ] No TODO comments left without tracking notes
- [ ] All modified files are in a clean, committed state

Only after completing ALL checks should you call task_complete().
If any check fails, fix the issue before proceeding.
"""

def inject_pre_completion_check(agent):
    original_complete = agent.tools["task_complete"]
    
    def verified_complete(**kwargs):
        # Inject checklist and force verification pass
        agent.append_to_context(PRE_COMPLETION_CHECKLIST)
        result = agent.run_single_turn(
            "Complete ALL items in the checklist above. "
            "Reply with: VERIFIED or FAILED:<reason>"
        )
        
        if "FAILED:" in str(result):
            raise RuntimeError(f"Pre-completion check failed: {result}")
        
        return original_complete(**kwargs)
    
    agent.tools["task_complete"] = verified_complete
```

### 4. Time Budget Injection

**The Finding**: Agents behave differently when they know they're running out of time. This is a real pattern—models that receive a turn budget behave differently than models that don't.

```python
class BudgetAwareContextInjector:
    def __init__(self, max_turns: int, warn_at: list[int] = [10, 5, 2]):
        self.max_turns = max_turns
        self.warn_at = warn_at
        self.current_turn = 0

    def get_budget_message(self) -> str | None:
        remaining = self.max_turns - self.current_turn
        self.current_turn += 1
        
        if remaining not in self.warn_at:
            return None
        
        if remaining <= 2:
            return """
# BUDGET WARNING: {remaining} turns remaining.
Immediately wrap up current work. If not complete:
1. Commit any working changes
2. Write a handoff note in todo.md describing exact state
3. List what remains to be done
Do NOT start new work. Consolidate now.
"""
        elif remaining <= 5:
            return f"""
Budget notice: {remaining} turns remaining.
Prioritize: finish current subtask, run tests, verify output.
Do not start new features or refactors.
"""
        else:
            return f"Note: {remaining} turns remaining in this session."
```

**Why It Works**: 
- At 10 turns, agent knows to prioritize
- At 5 turns, agent stops new work
- At 2 turns, agent focuses on clean handoff

The 2-turn warning is most important—the agent needs to either complete or hand off cleanly, not attempt one more change that will be left broken.

### 5. The Four-Stage Structure

LangChain's Terminal Bench findings support explicit stage structure:

```
┌─────────────────────────────────────────────────────────────┐
│                    FOUR-STAGE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  PLAN   │───▶│  BUILD  │───▶│ VERIFY  │───▶│   FIX   │  │
│  │         │    │         │    │         │    │         │  │
│  │ xhigh   │    │ high    │    │ xhigh   │    │ high    │  │
│  │ reason  │    │ reason  │    │ reason  │    │ reason  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│  LocalContext   LoopDetect    Checklist     ErrorEvidence  │
│  + todo.md                    Injection     Preservation   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Value**: The stages aren't magical—they give you clean **harness injection points**.

| Stage | Reasoning | Harness Injections |
|-------|-----------|-------------------|
| Plan | xhigh | LocalContextMiddleware, todo.md creation |
| Build | high | Loop detection active |
| Verify | xhigh | Pre-completion checklist |
| Fix | high | Error evidence preserved |

```python
class FourStagePipeline:
    def __init__(self, agent):
        self.agent = agent
        self.reasoning = ReasoningSandwichScheduler()
        self.loop_detector = LoopDetectionMiddleware()
        self.budget = BudgetAwareContextInjector(max_turns=50)

    def run(self, task: str):
        # PLAN phase
        self.agent.set_reasoning(self.reasoning.get_reasoning_level("planning"))
        self.agent.inject_context(self.local_context.build_context())
        plan = self.agent.run_turn("Create a plan for: " + task)
        
        # BUILD phase
        self.agent.set_reasoning(self.reasoning.get_reasoning_level("implementation"))
        while not self.agent.claims_complete():
            # Check for loops
            is_loop, intervention = self.loop_detector.check_and_intervene(
                self.agent.last_action
            )
            if is_loop:
                self.agent.inject_context(intervention)
            
            # Inject budget warnings
            budget_msg = self.budget.get_budget_message()
            if budget_msg:
                self.agent.inject_context(budget_msg)
            
            self.agent.run_turn()
        
        # VERIFY phase
        self.agent.set_reasoning(self.reasoning.get_reasoning_level("verification"))
        self.agent.inject_context(PRE_COMPLETION_CHECKLIST)
        verified = self.agent.run_turn("Complete verification checklist")
        
        if not verified:
            # FIX phase
            self.agent.set_reasoning(self.reasoning.get_reasoning_level("implementation"))
            # Preserve error evidence and retry
            self.agent.run_turn("Fix the verification failures")
```

---

## Constraint Management Checklist

- [ ] Reasoning sandwich configured (xhigh at Plan/Verify, high at Build/Fix)
- [ ] Loop detection active with appropriate thresholds
- [ ] Pre-completion verification gate installed
- [ ] Time budget warnings injected (10, 5, 2 turn warnings)
- [ ] Four-stage structure enforced
- [ ] Retry policy defined (max attempts, backoff, escalation)
- [ ] Hard budget ceiling (max turns per task)

---

## Retry Policies

Different failures need different handling:

```python
class RetryPolicy:
    def __init__(self):
        self.max_retries = {
            "timeout": 2,        # Network issues - worth retrying
            "rate_limit": 3,     # Back off and retry
            "tool_error": 1,     # Different approach needed
            "validation": 0,     # Don't retry - fix the issue
        }
        self.backoff_seconds = {
            "timeout": 5,
            "rate_limit": 30,
            "tool_error": 0,
        }

    def should_retry(self, error_type: str, attempt: int) -> tuple[bool, float]:
        max_attempts = self.max_retries.get(error_type, 0)
        if attempt >= max_attempts:
            return False, 0
        
        backoff = self.backoff_seconds.get(error_type, 0)
        return True, backoff * (attempt + 1)  # Linear backoff
```

---

## Exercises

### Exercise 3.1: Implement Reasoning Sandwich
Build a reasoning scheduler that:
1. Accepts the current phase as input
2. Returns the appropriate reasoning level
3. Builds the API parameters for the model call

### Exercise 3.2: Build Loop Detection
Implement loop detection middleware that:
1. Tracks the last N actions
2. Detects repeated edits to the same file
3. Injects intervention prompts
4. Allows whitelisted files (for TDD patterns)

### Exercise 3.3: Create Verification Gate
Design a pre-completion checklist for your task type:
1. Define functional verification steps
2. Define quality checks
3. Define completeness checks
4. Implement the gate that enforces the checklist

### Exercise 3.4: Design Budget Warnings
Create budget warning messages for:
1. 10 turns remaining
2. 5 turns remaining
3. 2 turns remaining
4. 0 turns (timeout)

What actions should the agent take at each stage?

---

## Anti-Patterns to Avoid

### ❌ Infinite Retry Without Escalation
```python
# BAD: Will run forever
while True:
    try:
        result = agent.run(task)
        break
    except:
        continue  # No limit, no escalation
```

### ❌ No Loop Detection
```python
# BAD: Agent can edit same file 100 times
while not done:
    agent.run_turn()  # No loop checking
```

### ❌ Maximum Reasoning Everywhere
```python
# BAD: Slowest and worst performing
for turn in task:
    agent.run_turn(reasoning="xhigh")  # Even for simple file reads
```

### ❌ No Budget Awareness
```python
# BAD: Agent doesn't know when to wrap up
agent.run(task, max_turns=50)  # No warnings, sudden cutoff
```

---

## Key Takeaways

1. **Maximum reasoning hurts**: The reasoning sandwich (xhigh → high → xhigh) beats maximum reasoning by 12+ points

2. **Loops are predictable**: Detect them with simple heuristics and inject reconsideration prompts

3. **Completion is not claimed, it's verified**: Force a checklist pass before task_complete()

4. **Budget awareness changes behavior**: Agents prioritize differently when they know time is limited

5. **Four stages = clean injection points**: Plan, Build, Verify, Fix each get appropriate constraints

---

## Next Module

Continue to **[Module 4: Evaluation Harness](../04_evaluation/README.md)** to learn how to test and benchmark agent systems.

---

## References

- LangChain. "Evaluating deepagents-cli on Terminal Bench 2.0" - Reasoning sandwich data
- Claude Code system prompt (April 2026) - Numeric constraints pattern
- Manus. "Context Engineering for AI Agents" - Error recovery patterns
