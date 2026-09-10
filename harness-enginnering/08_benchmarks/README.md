# Module 9: Research Benchmarks

## Overview

How do you know if your harness is actually good? This module covers the benchmarks used to evaluate harness engineering research—from coding agent benchmarks to research automation benchmarks. Understanding these benchmarks helps you measure progress and compare approaches.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Select appropriate benchmarks** for different harness capabilities
2. **Understand SWE-bench variants** and their differences
3. **Use research automation benchmarks** like PaperBench and RE-Bench
4. **Design evaluation protocols** that avoid common pitfalls
5. **Interpret benchmark results** correctly

---

## Key Concepts

### 1. The Benchmark Landscape

```
┌─────────────────────────────────────────────────────────────┐
│                   HARNESS BENCHMARKS                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CODING AGENTS                                               │
│  ├── SWE-bench (Original, Verified, Pro, Live)              │
│  ├── Terminal-Bench                                          │
│  └── Polyglot                                                │
│                                                              │
│  RESEARCH AUTOMATION                                         │
│  ├── PaperBench (replicate ICML papers)                     │
│  ├── CORE-Bench (computational reproducibility)             │
│  ├── RE-Bench (research engineering)                        │
│  └── ScienceAgentBench (data-driven discovery)              │
│                                                              │
│  ML ENGINEERING                                              │
│  ├── MLE-bench (Kaggle competitions)                        │
│  └── KernelBench (GPU kernel optimization)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. SWE-bench Family

SWE-bench has become the de-facto standard for coding agent evaluation. But there are **five distinct variants**—comparing scores across them is methodological malpractice.

| Variant | Tasks | Difficulty | Use Case |
|---------|-------|------------|----------|
| **Original** | 2,294 | Mixed | Historical comparison |
| **Verified** | 500 | Human-validated | Standard reporting |
| **Pro** | 1,000 | Harder | Advanced agents |
| **Multilingual** | 350 | 8 languages | Cross-language |
| **Live** | Rolling | Current issues | Real-world testing |

**SWE-bench Verified** is the current standard for reporting because:
- Tasks are human-validated for solvability
- Deterministic pass/fail from unit tests
- No rubric subjectivity or cherry-picking

```python
# SWE-bench evaluation structure
@dataclass
class SWEBenchTask:
    repo: str              # GitHub repository
    issue: str             # Issue description
    base_commit: str       # Starting point
    test_patch: str        # Tests that must pass
    gold_patch: str        # Reference solution (not given to agent)


def evaluate_swebench(agent: Agent, task: SWEBenchTask) -> bool:
    """Standard SWE-bench evaluation."""
    # 1. Clone repo at base commit
    workspace = clone_repo(task.repo, task.base_commit)
    
    # 2. Agent attempts to fix the issue
    agent_patch = agent.solve(task.issue, workspace)
    
    # 3. Apply agent's patch
    apply_patch(workspace, agent_patch)
    
    # 4. Apply test patch
    apply_patch(workspace, task.test_patch)
    
    # 5. Run tests
    result = run_tests(workspace)
    
    return result.all_passed
```

### 3. Terminal-Bench

Terminal-Bench evaluates agents on **terminal-based tasks**—system administration, DevOps, and CLI workflows.

**Key Characteristics**:
- Tasks require shell command composition
- Includes system configuration, debugging, automation
- Measures both correctness and efficiency

**LangChain's Terminal-Bench 2.0 Results** (from their harness engineering research):

| Configuration | Score |
|--------------|-------|
| Basic harness | 52.8% |
| Optimized harness (same model) | 66.5% |
| Improvement from harness alone | +13.7 points |

This benchmark demonstrated that harness engineering can achieve gains equivalent to model improvements.

### 4. Research Automation Benchmarks

#### PaperBench
**Task**: Replicate 20 ICML 2024 Spotlight and Oral papers from scratch.

```python
@dataclass
class PaperBenchTask:
    paper_id: str
    paper_pdf: str
    rubrics: list[str]       # 8,316 total, co-developed with authors
    evaluation_criteria: dict  # Contribution understanding, code, experiments
```

**Key Findings**:
- Best model at time (Claude 3.5 Sonnet): ~21%
- Does not outperform ML PhDs
- Decomposes replication into gradable subtasks

#### RE-Bench (Research Engineering)
**Task**: 7 challenging, open-ended ML research-engineering environments.

| Environment Example | Description |
|---------------------|-------------|
| Kernel optimization | Optimize a GPU kernel |
| Scaling experiment | Run scaling-law experiments |
| Fine-tune GPT-2 | Fine-tune for QA task |
| Fix embedding | Debug embedding issue |

**Human vs AI Comparison**:
- At 2 hours: AI agents scored **4× higher** than humans
- At 8 hours: Humans exceeded agents
- At 32 hours: Humans significantly better

**Implication**: Current agents are good at quick tasks but struggle with extended research that requires iterative exploration.

#### CORE-Bench
**Task**: Evaluate computational reproducibility of published research.

- 270 tasks from 90 papers
- Spans computer science, social science, medicine
- Reproduces results from provided code and data
- Best agent (GPT-4o): 21% on hardest tier

#### ScienceAgentBench
**Task**: Data-driven scientific discovery.

- 102 tasks from 44 peer-reviewed publications
- Four disciplines: math, chemistry, biology, geography
- Tasks: data processing, model development, analysis, visualization

### 5. ML Engineering Benchmarks

#### MLE-bench
**Task**: Offline Kaggle competitions.

```python
@dataclass  
class MLEBenchTask:
    competition_id: str
    description: str
    train_data: Path
    test_data: Path
    submission_format: str
    leaderboard_baseline: float  # Public leaderboard reference


def evaluate_mlebench(agent: Agent, task: MLEBenchTask) -> float:
    """Evaluate on Kaggle-style competition."""
    # Agent trains model and generates predictions
    submission = agent.compete(
        task.description,
        task.train_data,
        task.submission_format
    )
    
    # Score against hidden test set
    score = grade_submission(submission, task.test_data)
    
    # Compare to human baseline (Kaggle bronze medal)
    return score, score >= task.leaderboard_baseline
```

**Results**: o1-preview with AIDE scaffolding reached bronze-medal level in 16.9% of competitions.

#### KernelBench
**Task**: Write fast and correct GPU kernels.

- 250 PyTorch tasks
- Metric: `fast_p` = % of kernels that are correct AND faster than baseline
- Tests correctness (functional) and performance (speed)

---

## Benchmark Selection Guide

| Goal | Recommended Benchmark | Why |
|------|----------------------|-----|
| General coding agent | SWE-bench Verified | Industry standard, human-validated |
| Terminal/DevOps | Terminal-Bench 2.0 | CLI-focused, efficiency metrics |
| Research automation | PaperBench | End-to-end research capability |
| Quick research tasks | RE-Bench (2hr) | Time-bounded comparison |
| ML engineering | MLE-bench | Real Kaggle competitions |
| GPU optimization | KernelBench | Performance-focused |
| Cross-language | SWE-bench Multilingual | 8 programming languages |

---

## Evaluation Protocol Design

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Cherry-picking** | Report best of N runs | Report mean ± std over fixed N |
| **Cross-variant comparison** | Compare SWE-bench Original to Verified | Compare only within same variant |
| **Contamination** | Model trained on benchmark data | Use held-out or live benchmarks |
| **Single run** | High variance from non-determinism | Run 3-5 times, report aggregate |
| **Ignoring cost** | Compare success rate only | Report cost per successful task |

### Proper Evaluation Protocol

```python
class EvaluationProtocol:
    """Rigorous evaluation protocol for harness research."""
    
    def __init__(
        self,
        benchmark: Benchmark,
        num_runs: int = 3,
        timeout_seconds: int = 3600,
    ):
        self.benchmark = benchmark
        self.num_runs = num_runs
        self.timeout = timeout_seconds
    
    def evaluate(self, harness: Harness) -> EvaluationReport:
        all_results = []
        
        for task in self.benchmark.tasks:
            task_results = []
            
            for run_idx in range(self.num_runs):
                # Set random seed for reproducibility
                set_seed(task.id * 1000 + run_idx)
                
                result = self._run_single(harness, task)
                task_results.append(result)
            
            all_results.append(TaskResults(
                task_id=task.id,
                success_rate=sum(r.success for r in task_results) / len(task_results),
                mean_cost=np.mean([r.cost for r in task_results]),
                mean_latency=np.mean([r.latency for r in task_results]),
                std_cost=np.std([r.cost for r in task_results]),
            ))
        
        return EvaluationReport(
            benchmark_name=self.benchmark.name,
            harness_name=harness.name,
            num_runs=self.num_runs,
            overall_success_rate=np.mean([r.success_rate for r in all_results]),
            overall_success_std=np.std([r.success_rate for r in all_results]),
            total_cost=sum(r.mean_cost for r in all_results),
            task_results=all_results,
        )
    
    def _run_single(self, harness: Harness, task: Task) -> RunResult:
        start_time = time.time()
        tokens_used = 0
        
        try:
            with timeout(self.timeout):
                result = harness.solve(task)
                success = self.benchmark.verify(task, result)
                tokens_used = harness.get_tokens_used()
        except TimeoutError:
            success = False
        except Exception as e:
            success = False
            logging.error(f"Task {task.id} failed: {e}")
        
        return RunResult(
            success=success,
            latency=time.time() - start_time,
            cost=self._compute_cost(tokens_used),
        )
```

### Reporting Template

```markdown
## Evaluation Results

**Benchmark**: SWE-bench Verified (500 tasks)
**Harness**: MyHarness v1.2
**Model**: Claude 3.5 Sonnet
**Runs per task**: 3

### Primary Metrics
| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 45.2% | ±2.1% |
| Cost per Task | $0.82 | ±$0.15 |
| Median Latency | 127s | - |

### Comparison to Baselines
| Harness | Success Rate | Cost | 
|---------|--------------|------|
| Baseline (no harness) | 23.0% | $1.45 |
| Basic harness | 38.5% | $0.95 |
| **MyHarness (ours)** | **45.2%** | **$0.82** |

### Ablations
| Component Removed | Success Rate | Delta |
|-------------------|--------------|-------|
| Full harness | 45.2% | - |
| - todo.md | 41.3% | -3.9% |
| - loop detection | 43.1% | -2.1% |
| - reasoning sandwich | 42.8% | -2.4% |
```

---

## Cost-Aware Evaluation

Success rate alone doesn't tell the full story. A harness that achieves 50% at $10/task is worse than one achieving 45% at $0.50/task.

```python
@dataclass
class CostAwareMetrics:
    success_rate: float
    cost_per_task: float
    cost_per_success: float  # More meaningful!
    
    @classmethod
    def compute(cls, results: list[RunResult]) -> "CostAwareMetrics":
        successes = sum(1 for r in results if r.success)
        total_cost = sum(r.cost for r in results)
        
        return cls(
            success_rate=successes / len(results),
            cost_per_task=total_cost / len(results),
            cost_per_success=total_cost / max(1, successes),
        )


def compare_harnesses(harness_a: Harness, harness_b: Harness, benchmark: Benchmark):
    """Cost-aware comparison of two harnesses."""
    results_a = evaluate(harness_a, benchmark)
    results_b = evaluate(harness_b, benchmark)
    
    metrics_a = CostAwareMetrics.compute(results_a)
    metrics_b = CostAwareMetrics.compute(results_b)
    
    print(f"Harness A: {metrics_a.success_rate:.1%} success, ${metrics_a.cost_per_success:.2f}/success")
    print(f"Harness B: {metrics_b.success_rate:.1%} success, ${metrics_b.cost_per_success:.2f}/success")
    
    # Which is better?
    # If budgets are equal, compare total successes achievable
    budget = 1000  # $1000 budget
    
    successes_a = budget / metrics_a.cost_per_success
    successes_b = budget / metrics_b.cost_per_success
    
    print(f"\nWith ${budget} budget:")
    print(f"  Harness A: {successes_a:.0f} expected successes")
    print(f"  Harness B: {successes_b:.0f} expected successes")
```

---

## Creating Your Own Benchmarks

When existing benchmarks don't fit your use case:

```python
class CustomBenchmark:
    """Template for creating domain-specific benchmarks."""
    
    def __init__(self):
        self.tasks = self._load_tasks()
        self.verifiers = self._build_verifiers()
    
    def _load_tasks(self) -> list[Task]:
        """Load or generate tasks."""
        tasks = []
        
        # Example: Load from historical issues/tickets
        for issue in load_historical_issues():
            if self._is_suitable(issue):
                tasks.append(Task(
                    id=issue.id,
                    description=issue.description,
                    context=issue.context,
                    gold_solution=issue.resolution,
                ))
        
        return tasks
    
    def _is_suitable(self, issue) -> bool:
        """Filter for benchmark-suitable tasks."""
        return (
            issue.has_clear_resolution and
            issue.is_automatically_verifiable and
            not issue.is_duplicate and
            issue.complexity in ["medium", "hard"]
        )
    
    def _build_verifiers(self) -> dict[str, Callable]:
        """Build verification functions for each task."""
        verifiers = {}
        
        for task in self.tasks:
            if task.has_tests:
                verifiers[task.id] = lambda result: run_tests(result)
            else:
                # Use LLM-as-judge for tasks without tests
                verifiers[task.id] = lambda result: llm_judge(
                    task.description,
                    task.gold_solution,
                    result
                )
        
        return verifiers
    
    def verify(self, task: Task, result: str) -> bool:
        """Verify a result for a task."""
        return self.verifiers[task.id](result)
```

---

## Exercises

### Exercise 9.1: Benchmark Analysis
Take results from a harness paper and:
1. Identify which benchmark variant was used
2. Check if proper evaluation protocol was followed
3. Calculate cost per success

### Exercise 9.2: Run SWE-bench Evaluation
Set up SWE-bench Verified evaluation:
1. Clone the SWE-bench repository
2. Run 10 tasks with a simple harness
3. Report success rate, cost, latency

### Exercise 9.3: Design Custom Benchmark
Create a benchmark for your domain:
1. Define 20+ tasks
2. Build automatic verifiers
3. Document difficulty distribution

### Exercise 9.4: Cost-Aware Comparison
Compare two harness configurations:
1. Run both on same benchmark
2. Compute cost per success
3. Determine which is better under budget constraints

---

## Key Takeaways

1. **Use the right variant**: SWE-bench has 5 variants; don't mix them

2. **Report properly**: Mean ± std over multiple runs, include cost

3. **Cost matters**: Cost per success is more meaningful than success rate alone

4. **Humans vs. AI**: Agents beat humans at short tasks, lose at longer ones

5. **Verification is key**: Benchmarks need automatic, deterministic verification

---

## Next Module

Continue to **[Module 10: Future Directions](../10_future/README.md)** to explore emerging research directions and open challenges.

---

## References

- SWE-bench: https://www.swebench.com/
- PaperBench: Starace et al. (2025)
- RE-Bench: Wijk et al. (2025)
- CORE-Bench: Siegel et al. (2024)
- MLE-bench: Chan et al. (2024)
- KernelBench: Ouyang et al. (2025)
- ScienceAgentBench: Chen et al. (2025)
