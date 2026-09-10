# Module 4: Evaluation Harness

## Overview

Testing an agent requires a different mindset than typical QA. If your integration tests expect the model to generate the exact output every run, you'll end up with a flaky test suite. This module teaches you how to build evaluation harnesses that measure what matters: **reliable behavior, not exact outputs**.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Distinguish between** end-to-end benchmarks and behavioral evaluations
2. **Write behavioral evals** that assert on actions, not text
3. **Build the three-stage evaluation architecture** (inputs → execution → actions)
4. **Track the metrics that predict harness health**
5. **Integrate evaluations into CI/CD pipelines**

---

## Key Concepts

### 1. Two Types of Evaluation

#### End-to-End Benchmarks
- Measure final outcomes on standardized tasks
- Examples: SWE-bench, Terminal-Bench, GAIA
- Good for: comparing models, tracking capability
- Limitation: don't tell you *why* something failed

#### Behavioral Evaluations
- Measure discrete, observable actions
- Assert on *what the agent did*, not *what text it generated*
- Good for: iteration, regression detection, debugging
- Limitation: require upfront design of expected behaviors

**The Google Developers insight**:
> End-to-end benchmarks are the de facto for evaluating model performance, but behavioral evaluations are a better measure of confidence on whether the behaviors you expect actually happen.

### 2. The Paradigm Shift: Report Cards vs. Behavioral Guideposts

**Old Way**: Evaluate agents like students taking exams
- Hand the agent a codebase
- Give it a time limit
- Measure: how many tests pass?

**When the score drops, what went wrong?**
- Did the model get overconfident on ambiguous prompts?
- Did it forget to verify the test suite before submitting?
- Did it hallucinate a CLI flag?

End-to-end benchmarks don't answer these questions.

**New Way**: Integration tests for agent behavior
- Measure discrete, observable actions
- Build a baseline of expected behaviors
- Iterate on the harness to achieve the baseline

**Behavioral eval examples**:
- When given an underspecified prompt, does the agent ask a clarifying question?
- When modifying a build file, does it run the validator before completing?
- When generating documentation, does it provide canonical links?

### 3. Writing Behavioral Evals

Behavioral evals assert on **intermediate execution steps**, not final string equality:

```python
import pytest

@pytest.mark.asyncio
async def test_agent_uses_web_search_for_live_data():
    """Assert that the agent consults ground truth rather than guessing."""
    
    async with Agent(config) as agent:
        response = await agent.chat("What's the weather in Mountain View?")
        tools_called = [call.name async for call in response.tool_calls]
    
    # Assert behavior, not output prose
    assert "web_search" in tools_called, (
        "Agent answered from memory without consulting live search."
    )


@pytest.mark.asyncio
async def test_agent_runs_tests_before_completion():
    """Assert that the agent validates its work."""
    
    async with Agent(config) as agent:
        response = await agent.chat("Fix the bug in auth.py")
        tools_called = [call.name async for call in response.tool_calls]
    
    # Verify the agent ran tests
    assert "run_tests" in tools_called, (
        "Agent claimed completion without running tests."
    )
    
    # Verify tests were run BEFORE completion
    test_index = tools_called.index("run_tests")
    complete_index = tools_called.index("task_complete") if "task_complete" in tools_called else len(tools_called)
    assert test_index < complete_index, (
        "Agent ran tests after claiming completion."
    )


@pytest.mark.asyncio  
async def test_agent_asks_clarification_on_ambiguous_prompt():
    """Assert that the agent doesn't guess on underspecified tasks."""
    
    async with Agent(config) as agent:
        # Deliberately ambiguous prompt
        response = await agent.chat("Fix the bug")
        
    # Should ask for clarification, not just start editing
    assert response.asks_clarification or "which" in response.text.lower(), (
        "Agent started work on ambiguous prompt without clarification."
    )
```

### 4. The Three-Stage Evaluation Architecture

A mature evaluation harness has three stages:

```
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION HARNESS                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: INPUTS          Stage 2: EXECUTION                │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │ What gets        │    │ How it gets      │               │
│  │ evaluated        │───▶│ scored           │               │
│  │                  │    │                  │               │
│  │ • Traces         │    │ • LLM-as-Judge   │               │
│  │ • Test cases     │    │ • Code checks    │               │
│  │ • Production     │    │ • Embedding sim  │               │
│  │   slices         │    │ • Custom funcs   │               │
│  └──────────────────┘    └──────────────────┘               │
│                                   │                          │
│                                   ▼                          │
│                    Stage 3: ACTIONS                          │
│                    ┌──────────────────┐                      │
│                    │ What happens     │                      │
│                    │ next             │                      │
│                    │                  │                      │
│                    │ • Alerts         │                      │
│                    │ • Annotation     │                      │
│                    │ • CI/CD gates    │                      │
│                    │ • Experiments    │                      │
│                    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

#### Stage 1: Define Inputs

Inputs operate at four levels of granularity:

| Level | What It Measures | Example Question |
|-------|------------------|------------------|
| **Spans** | Single unit of work (one LLM call, tool call) | Did this specific step succeed? |
| **Traces** | Full chain for one request | Did the end-to-end workflow deliver? |
| **Trajectories** | The path an agent took | Did the agent reason correctly? |
| **Sessions** | Multi-turn interactions | Did the system maintain coherence? |

#### Stage 2: Run Evaluation Methods

| Method | When to Use | Example |
|--------|-------------|---------|
| **LLM-as-Judge** | Open-ended quality (relevance, tone) | "Is this response helpful?" |
| **Code Checks** | Fixed shape answers | JSON schema validation |
| **Embedding Similarity** | Fuzzy equivalence | Semantic match to reference |
| **Custom Functions** | Domain-specific | Call internal API, run classifier |

#### Stage 3: Act on Results

**This is where most evaluation workflows break**: they produce scores without connecting them to action.

| Action | Trigger | Value |
|--------|---------|-------|
| **Annotation Queues** | Low-confidence outputs | Human labels → better evals |
| **Alerts** | Metric thresholds crossed | Catch regressions early |
| **CI/CD Gates** | PR/deploy events | Prevent quality drops |
| **Experiment Workflows** | Failing evals | Automated improvement loops |

### 5. The Three Metrics That Predict Harness Health

```python
@dataclass
class HarnessMetrics:
    tool_calls: int = 0
    tool_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    loop_detections: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls > 0 else 0.0
```

| Metric | Target | What It Means |
|--------|--------|---------------|
| **Cache hit rate** | >80% | Below 50% = prefix is breaking somewhere |
| **Tool error rate** | <10% per tool | Above 30% = redesign or remove tool |
| **Loop detections per run** | 0 in steady state | >2 per run = todo.md not preventing drift |

### 6. Building a Behavioral Test Suite

**The Three-Step Loop**:

1. **Pick one failure mode**: Find a recent mistake your agent made
2. **Write flexible assertions**: Single-turn for simple tasks, outcome-based for complex
3. **Automate batch evaluations**: Track aggregate pass rates over time

```python
class BehavioralTestSuite:
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        self.results = []
    
    async def run_test(self, test_case: dict) -> TestResult:
        agent = self.agent_factory()
        
        response = await agent.chat(test_case["prompt"])
        tools_called = [call.name for call in response.tool_calls]
        
        # Check required tools
        for required in test_case.get("required_tools", []):
            if required not in tools_called:
                return TestResult(
                    passed=False,
                    reason=f"Missing required tool: {required}"
                )
        
        # Check forbidden tools
        for forbidden in test_case.get("forbidden_tools", []):
            if forbidden in tools_called:
                return TestResult(
                    passed=False,
                    reason=f"Used forbidden tool: {forbidden}"
                )
        
        # Check tool ordering
        if "tool_order" in test_case:
            expected_order = test_case["tool_order"]
            actual_order = [t for t in tools_called if t in expected_order]
            if actual_order != expected_order:
                return TestResult(
                    passed=False,
                    reason=f"Wrong tool order: {actual_order} vs {expected_order}"
                )
        
        return TestResult(passed=True)
    
    async def run_batch(self, test_cases: list[dict], runs_per_case: int = 3):
        """Run multiple times to account for non-determinism."""
        for case in test_cases:
            case_results = []
            for _ in range(runs_per_case):
                result = await self.run_test(case)
                case_results.append(result)
            
            pass_rate = sum(r.passed for r in case_results) / len(case_results)
            self.results.append({
                "case": case["name"],
                "pass_rate": pass_rate,
                "threshold": case.get("threshold", 0.8)
            })
        
        return self.results
```

### 7. CI/CD Integration

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'tools/**'
      - 'harness/**'

jobs:
  behavioral-evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run behavioral tests
        run: |
          pytest evals/behavioral/ -v --tb=short
      
      - name: Run batch evaluation
        run: |
          python evals/batch_eval.py --runs=5 --threshold=0.8
      
      - name: Check metrics
        run: |
          python evals/check_metrics.py \
            --cache-hit-rate-min=0.8 \
            --error-rate-max=0.1 \
            --loop-detections-max=2

  regression-check:
    runs-on: ubuntu-latest
    needs: behavioral-evals
    steps:
      - name: Compare to baseline
        run: |
          python evals/regression_check.py \
            --baseline=main \
            --current=${{ github.sha }} \
            --max-regression=0.05
```

---

## Evaluation Methods Deep Dive

### LLM-as-Judge

Use when you need to evaluate open-ended quality:

```python
JUDGE_PROMPT = """
You are evaluating an AI agent's response.

Task given to agent: {task}
Agent's response: {response}

Evaluate on a scale of 1-5:
1. Relevance: Does the response address the task?
2. Completeness: Did the agent do everything needed?
3. Safety: Did the agent follow proper procedures?

Respond with JSON:
{{"relevance": X, "completeness": X, "safety": X, "reasoning": "..."}}
"""

async def llm_judge(task: str, response: str) -> dict:
    judge_response = await llm.generate(
        JUDGE_PROMPT.format(task=task, response=response)
    )
    return json.loads(judge_response)
```

### Code-Based Checks

Use for deterministic, structural validation:

```python
def check_response_format(response: dict) -> bool:
    """Validate response has required structure."""
    required_fields = ["status", "result", "metadata"]
    return all(field in response for field in required_fields)

def check_tool_call_params(tool_call: dict, schema: dict) -> bool:
    """Validate tool call parameters against schema."""
    try:
        jsonschema.validate(tool_call["params"], schema)
        return True
    except jsonschema.ValidationError:
        return False
```

### Embedding Similarity

Use for fuzzy equivalence when exact match is too strict:

```python
from sentence_transformers import SentenceTransformer

def embedding_similarity(response: str, reference: str, threshold: float = 0.8) -> bool:
    """Check if response is semantically similar to reference."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode([response, reference])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity >= threshold
```

---

## Exercises

### Exercise 4.1: Write Behavioral Evals
For your agent task, write three behavioral evals:
1. A tool-must-be-called assertion
2. A tool-ordering assertion
3. A clarification-required assertion

### Exercise 4.2: Build a Metrics Dashboard
Implement a metrics collector that tracks:
1. Cache hit rate
2. Tool error rate per tool
3. Loop detections per run
4. Average turns to completion

### Exercise 4.3: CI/CD Integration
Create a GitHub Actions workflow that:
1. Runs behavioral tests on PR
2. Compares metrics to baseline
3. Blocks merge if regression >5%

### Exercise 4.4: LLM-as-Judge Evaluator
Build an LLM-as-judge evaluator for your task:
1. Define the rubric (what dimensions to evaluate)
2. Write the judge prompt
3. Parse and aggregate results
4. Set pass/fail thresholds

---

## Evaluation Checklist

- [ ] Behavioral evals defined for key failure modes
- [ ] Three-stage architecture implemented (inputs → execution → actions)
- [ ] Metrics tracked: cache hit rate, error rate, loop detections
- [ ] CI/CD integration with regression blocking
- [ ] Batch evaluation for non-determinism handling
- [ ] LLM-as-judge for open-ended quality
- [ ] Code checks for structural validation

---

## Key Takeaways

1. **Test behavior, not text**: Assert on tool calls and actions, not output strings

2. **End-to-end and behavioral are complementary**: Macro benchmarks verify destination; micro evals enable safe iteration

3. **Three stages**: Inputs (what to evaluate) → Execution (how to score) → Actions (what happens next)

4. **Three metrics predict health**: Cache hit rate, tool error rate, loop detections

5. **Batch for non-determinism**: Run multiple times, track aggregate pass rates

---

## Next Module

Continue to **[Module 5: Production Hardening](../05_production/README.md)** to learn how to make harnesses work in the real world.

---

## References

- Arize. "What is an Evaluation Harness?"
- Google Developers. "The Anatomy of Harness Engineering"
- FutureAGI. "How to Evaluate AI Agents"
- ACL Digital. "How to Actually Test an Autonomous Agent"
