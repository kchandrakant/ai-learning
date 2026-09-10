# Step 9: Workflow Orchestration

## Beyond Single Interactions

Real applications need reliable, observable, and recoverable workflows.

## Workflow Patterns

### 1. Sequential

Steps execute one after another:

```
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│Step 1│───▶│Step 2│───▶│Step 3│───▶│Step 4│
└──────┘    └──────┘    └──────┘    └──────┘
```

```python
class SequentialWorkflow:
    def __init__(self, steps: list[Callable]):
        self.steps = steps
    
    def run(self, initial_input: Any) -> Any:
        result = initial_input
        for step in self.steps:
            result = step(result)
        return result
```

### 2. Parallel

Independent steps run simultaneously:

```
              ┌──────┐
         ┌───▶│Step A│───┐
         │    └──────┘   │
┌──────┐ │    ┌──────┐   │    ┌──────┐
│ Fan  │─┼───▶│Step B│───┼───▶│ Join │
│ Out  │ │    └──────┘   │    └──────┘
└──────┘ │    ┌──────┐   │
         └───▶│Step C│───┘
              └──────┘
```

```python
import asyncio

class ParallelWorkflow:
    def __init__(self, parallel_steps: list[Callable]):
        self.steps = parallel_steps
    
    async def run(self, input_data: Any) -> list[Any]:
        tasks = [
            asyncio.create_task(step(input_data))
            for step in self.steps
        ]
        return await asyncio.gather(*tasks)
```

### 3. Conditional (Branching)

Different paths based on conditions:

```
              ┌──────┐
         ┌───▶│Path A│
         │    └──────┘
┌──────┐ │
│ Check│─┤
└──────┘ │    ┌──────┐
         └───▶│Path B│
              └──────┘
```

```python
class ConditionalWorkflow:
    def __init__(self):
        self.routes: dict[str, Callable] = {}
    
    def add_route(self, condition: str, handler: Callable):
        self.routes[condition] = handler
    
    def run(self, input_data: Any, router: Callable) -> Any:
        route_key = router(input_data)
        handler = self.routes.get(route_key)
        if handler:
            return handler(input_data)
        raise ValueError(f"No route for: {route_key}")
```

### 4. Loop (Iterative)

Repeat until condition met:

```
         ┌────────────────────┐
         │                    │
         ▼                    │
    ┌──────┐    ┌──────┐     │
───▶│ Step │───▶│Check │─NO──┘
    └──────┘    └──┬───┘
                   │YES
                   ▼
              ┌──────┐
              │ Done │
              └──────┘
```

```python
class IterativeWorkflow:
    def __init__(self, 
                 step: Callable, 
                 check: Callable,
                 max_iterations: int = 10):
        self.step = step
        self.check = check
        self.max_iterations = max_iterations
    
    def run(self, initial_input: Any) -> Any:
        result = initial_input
        for i in range(self.max_iterations):
            result = self.step(result)
            if self.check(result):
                return result
        raise MaxIterationsExceeded()
```

## Error Handling & Retries

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientWorkflow:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def execute_step(self, step: Callable, input_data: Any) -> Any:
        return step(input_data)
    
    def run_with_fallback(self, 
                          step: Callable, 
                          fallback: Callable,
                          input_data: Any) -> Any:
        try:
            return self.execute_step(step, input_data)
        except Exception as e:
            logger.warning(f"Step failed: {e}, using fallback")
            return fallback(input_data)
```

## Checkpointing & Recovery

```python
import json
from pathlib import Path

class CheckpointedWorkflow:
    def __init__(self, workflow_id: str, checkpoint_dir: str = "./checkpoints"):
        self.workflow_id = workflow_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, step_id: str, state: dict):
        path = self.checkpoint_dir / f"{self.workflow_id}_{step_id}.json"
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load_checkpoint(self, step_id: str) -> dict | None:
        path = self.checkpoint_dir / f"{self.workflow_id}_{step_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def run_with_checkpoints(self, steps: list[tuple[str, Callable]], initial: Any):
        result = initial
        
        for step_id, step_fn in steps:
            # Check for existing checkpoint
            checkpoint = self.load_checkpoint(step_id)
            if checkpoint:
                result = checkpoint["result"]
                continue
            
            # Execute step
            result = step_fn(result)
            
            # Save checkpoint
            self.save_checkpoint(step_id, {"result": result})
        
        return result
```

## Human-in-the-Loop Gates

```python
class ApprovalGate:
    def __init__(self, notification_service):
        self.notifications = notification_service
    
    async def request_approval(self, 
                                action: str, 
                                context: dict,
                                timeout_hours: int = 24) -> bool:
        # Notify human
        approval_id = self.notifications.send(
            f"Approval needed for: {action}",
            context
        )
        
        # Wait for response
        deadline = datetime.now() + timedelta(hours=timeout_hours)
        while datetime.now() < deadline:
            response = self.check_approval(approval_id)
            if response is not None:
                return response
            await asyncio.sleep(60)  # Check every minute
        
        raise ApprovalTimeout(f"No response for: {action}")

class WorkflowWithApprovals:
    def __init__(self, approval_gate: ApprovalGate):
        self.gate = approval_gate
    
    async def run_step_with_approval(self, 
                                      step: Callable,
                                      requires_approval: bool,
                                      context: dict) -> Any:
        if requires_approval:
            approved = await self.gate.request_approval(
                step.__name__, context
            )
            if not approved:
                raise ApprovalDenied()
        
        return step(context)
```

## Complete Workflow Engine

```python
class WorkflowEngine:
    def __init__(self):
        self.steps: dict[str, Callable] = {}
        self.transitions: dict[str, list[str]] = {}
        self.conditions: dict[str, Callable] = {}
    
    def add_step(self, name: str, handler: Callable):
        self.steps[name] = handler
        self.transitions[name] = []
    
    def add_transition(self, from_step: str, to_step: str, condition: Callable = None):
        self.transitions[from_step].append({
            "to": to_step,
            "condition": condition or (lambda x: True)
        })
    
    def run(self, start_step: str, initial_state: dict) -> dict:
        current_step = start_step
        state = initial_state
        
        while current_step:
            # Execute step
            handler = self.steps[current_step]
            state = handler(state)
            
            # Find next step
            next_step = None
            for transition in self.transitions[current_step]:
                if transition["condition"](state):
                    next_step = transition["to"]
                    break
            
            current_step = next_step
        
        return state

# Usage
engine = WorkflowEngine()
engine.add_step("research", research_agent.run)
engine.add_step("write", writer_agent.run)
engine.add_step("review", reviewer_agent.run)

engine.add_transition("research", "write")
engine.add_transition("write", "review")
engine.add_transition("review", "write", lambda s: not s["approved"])
engine.add_transition("review", None, lambda s: s["approved"])  # End

result = engine.run("research", {"topic": "AI agents"})
```

## Files

- `workflow_orchestration.py` - Workflow engine and patterns

## Key Takeaways

1. Sequential, parallel, conditional, iterative patterns
2. Always implement retry and error handling
3. Checkpoint long workflows for recovery
4. Add human approval gates for critical actions
5. Build reusable workflow engines

## What's Next?

Step 10: **Evaluation** — measuring agent performance.
