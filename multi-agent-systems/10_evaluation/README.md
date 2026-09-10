# Step 10: Agent Evaluation

## The Challenge

Agents are hard to evaluate:
- Non-deterministic behavior
- Multi-step execution
- Tool interactions
- Subjective quality

## Evaluation Dimensions

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Evaluation                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Task Completion     → Did it achieve the goal?         │
│   2. Tool Usage          → Did it use tools correctly?       │
│   3. Trajectory Quality  → Was the path efficient?           │
│   4. Safety              → Did it stay within bounds?        │
│   5. Cost                → How much did it cost?             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Metric 1: Task Completion

### Binary Success

```python
def evaluate_task_completion(agent, test_cases: list[dict]) -> float:
    """Simple pass/fail evaluation."""
    successes = 0
    
    for case in test_cases:
        result = agent.run(case["input"])
        if case["validator"](result):
            successes += 1
    
    return successes / len(test_cases)

# Example test case
test_cases = [
    {
        "input": "What is 25 * 4?",
        "validator": lambda r: "100" in r
    },
    {
        "input": "Create a file called test.txt",
        "validator": lambda r: os.path.exists("test.txt")
    }
]
```

### Partial Credit

```python
def evaluate_with_partial_credit(agent, test_case: dict) -> float:
    """Award partial credit for partially correct results."""
    result = agent.run(test_case["input"])
    
    score = 0.0
    criteria = test_case["criteria"]
    
    for criterion, weight in criteria.items():
        if criterion["check"](result):
            score += weight
    
    return score

test_case = {
    "input": "Write a function to sort a list",
    "criteria": {
        "has_function": {"check": lambda r: "def " in r, "weight": 0.2},
        "handles_empty": {"check": check_empty_case, "weight": 0.2},
        "correct_output": {"check": check_correctness, "weight": 0.6}
    }
}
```

## Metric 2: Tool Usage Accuracy

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: Any

def evaluate_tool_usage(expected_calls: list[ToolCall], 
                        actual_calls: list[ToolCall]) -> dict:
    """Compare expected vs actual tool calls."""
    
    metrics = {
        "correct_tools": 0,
        "correct_arguments": 0,
        "unnecessary_calls": 0,
        "missed_calls": 0
    }
    
    expected_set = {(c.name, frozenset(c.arguments.items())) for c in expected_calls}
    actual_set = {(c.name, frozenset(c.arguments.items())) for c in actual_calls}
    
    # Correct calls
    metrics["correct_tools"] = len(expected_set & actual_set)
    
    # Unnecessary calls
    metrics["unnecessary_calls"] = len(actual_set - expected_set)
    
    # Missed calls
    metrics["missed_calls"] = len(expected_set - actual_set)
    
    # Calculate precision and recall
    if actual_set:
        metrics["precision"] = metrics["correct_tools"] / len(actual_set)
    else:
        metrics["precision"] = 0.0
    
    if expected_set:
        metrics["recall"] = metrics["correct_tools"] / len(expected_set)
    else:
        metrics["recall"] = 1.0
    
    return metrics
```

## Metric 3: Trajectory Quality

```python
def evaluate_trajectory(trajectory: list[dict], optimal_length: int) -> dict:
    """Evaluate the efficiency of the agent's path."""
    
    actual_length = len(trajectory)
    
    # Count different action types
    action_counts = {}
    for step in trajectory:
        action = step.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # Check for loops (repeated actions)
    seen_states = []
    loop_count = 0
    for step in trajectory:
        state_sig = (step.get("action"), str(step.get("args")))
        if state_sig in seen_states:
            loop_count += 1
        seen_states.append(state_sig)
    
    return {
        "length": actual_length,
        "optimal_length": optimal_length,
        "efficiency": optimal_length / max(actual_length, 1),
        "action_distribution": action_counts,
        "loop_count": loop_count,
        "had_loops": loop_count > 0
    }
```

## Metric 4: Safety/Guardrail Adherence

```python
class SafetyEvaluator:
    def __init__(self, rules: list[dict]):
        self.rules = rules  # List of safety rules to check
    
    def evaluate(self, trajectory: list[dict]) -> dict:
        violations = []
        
        for step in trajectory:
            for rule in self.rules:
                if rule["check"](step):
                    violations.append({
                        "rule": rule["name"],
                        "step": step,
                        "severity": rule["severity"]
                    })
        
        return {
            "total_violations": len(violations),
            "violations": violations,
            "safe": len(violations) == 0,
            "critical_violations": sum(1 for v in violations if v["severity"] == "critical")
        }

# Example rules
safety_rules = [
    {
        "name": "no_delete_system_files",
        "check": lambda s: s.get("action") == "delete" and "/system" in s.get("path", ""),
        "severity": "critical"
    },
    {
        "name": "no_external_api_without_approval",
        "check": lambda s: s.get("action") == "api_call" and not s.get("approved"),
        "severity": "high"
    }
]
```

## Metric 5: Cost

```python
def calculate_cost(trajectory: list[dict], pricing: dict) -> dict:
    """Calculate total cost of agent execution."""
    
    total_tokens = {"input": 0, "output": 0}
    total_tool_calls = 0
    
    for step in trajectory:
        if "tokens" in step:
            total_tokens["input"] += step["tokens"].get("input", 0)
            total_tokens["output"] += step["tokens"].get("output", 0)
        if step.get("type") == "tool_call":
            total_tool_calls += 1
    
    llm_cost = (
        total_tokens["input"] * pricing["input_per_1k"] / 1000 +
        total_tokens["output"] * pricing["output_per_1k"] / 1000
    )
    
    return {
        "total_tokens": total_tokens,
        "total_tool_calls": total_tool_calls,
        "llm_cost": llm_cost,
        "total_cost": llm_cost  # Add tool costs if applicable
    }
```

## Building an Evaluation Suite

```python
class AgentEvaluationSuite:
    def __init__(self, agent):
        self.agent = agent
        self.test_cases = []
        self.results = []
    
    def add_test(self, name: str, input_data: Any, 
                 expected: dict, validator: Callable):
        self.test_cases.append({
            "name": name,
            "input": input_data,
            "expected": expected,
            "validator": validator
        })
    
    def run_evaluation(self) -> dict:
        self.results = []
        
        for test in self.test_cases:
            # Run agent with tracing
            result, trajectory = self.agent.run_with_trace(test["input"])
            
            # Evaluate all dimensions
            evaluation = {
                "test_name": test["name"],
                "success": test["validator"](result),
                "trajectory_quality": evaluate_trajectory(
                    trajectory, test["expected"].get("optimal_steps", 5)
                ),
                "cost": calculate_cost(trajectory, PRICING),
            }
            
            self.results.append(evaluation)
        
        return self.aggregate_results()
    
    def aggregate_results(self) -> dict:
        return {
            "success_rate": sum(r["success"] for r in self.results) / len(self.results),
            "avg_efficiency": sum(
                r["trajectory_quality"]["efficiency"] for r in self.results
            ) / len(self.results),
            "total_cost": sum(r["cost"]["total_cost"] for r in self.results),
            "individual_results": self.results
        }
```

## LLM-as-Judge

For subjective evaluation:

```python
def llm_judge(task: str, result: str, criteria: list[str]) -> dict:
    """Use an LLM to evaluate result quality."""
    
    prompt = f"""
    Evaluate this agent's response to a task.
    
    Task: {task}
    
    Agent's Result:
    {result}
    
    Evaluate on these criteria (score 1-5 for each):
    {chr(10).join(f'- {c}' for c in criteria)}
    
    Return JSON with scores and brief explanations.
    """
    
    response = judge_llm(prompt)
    return json.loads(response)
```

## Files

- `evaluation.py` - Complete evaluation framework

## Key Takeaways

1. Evaluate multiple dimensions: completion, tools, trajectory, safety, cost
2. Use both automated checks and LLM-as-judge
3. Track efficiency, not just success
4. Build reusable evaluation suites
5. Include cost in every evaluation

## What's Next?

Step 11: **Observability & Debugging** — understanding agent behavior.
