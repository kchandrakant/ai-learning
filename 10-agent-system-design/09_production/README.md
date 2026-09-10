# Module 5: Production Hardening

## Overview

The patterns from previous modules work in development. Production introduces new challenges: untrusted code execution, memory limits, observability gaps, and cost management. This module teaches you how to harden your harness for real-world deployment.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement sandbox strategies** at three security levels
2. **Build memory compaction** for long-running sessions
3. **Instrument your harness** for observability
4. **Design recovery mechanisms** for graceful failure handling
5. **Manage costs** through token economics

---

## Key Concepts

### 1. Sandboxing the Bash Tool

The advice to consolidate tools into a single bash tool is correct—**but only if you sandbox it**. An unsandboxed bash tool is dangerous: the model can accidentally (or a prompt injection can intentionally) run destructive commands.

#### Option 1: Command Allowlist (Simplest)

```python
ALLOWED_COMMANDS = {
    "ls", "cat", "grep", "find", "python", "pytest",
    "ruff", "git", "npm", "node", "pip", "echo", "mkdir", "touch"
}

def safe_bash(command: str) -> str:
    """Execute bash command with allowlist enforcement."""
    base_cmd = command.strip().split()[0]
    
    if base_cmd not in ALLOWED_COMMANDS:
        return f"Error: '{base_cmd}' is not in the allowed command list."
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
```

**Pros**: Simple, fast, predictable  
**Cons**: Very restrictive, requires manual curation

#### Option 2: Docker Container (Isolated)

```python
import docker

client = docker.from_env()

def sandboxed_bash(command: str, workspace_path: str) -> str:
    """Run command in an isolated Docker container."""
    try:
        result = client.containers.run(
            "python:3.11-slim",
            command=["bash", "-c", command],
            volumes={workspace_path: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            network_mode="none",  # No internet access
            mem_limit="512m",
            remove=True,
            timeout=30,
        )
        return result.decode("utf-8")
    except docker.errors.ContainerError as e:
        return f"Error: {e.stderr.decode('utf-8')}"
```

**Pros**: Full isolation, filesystem protection, network control  
**Cons**: Startup latency, requires Docker

#### Option 3: Cloud Sandbox (E2B, Daytona)

For production multi-tenant deployments:

```python
# Using E2B (e2b.dev)
from e2b import Sandbox

async def cloud_sandboxed_bash(command: str) -> str:
    sandbox = await Sandbox.create()
    try:
        result = await sandbox.process.start_and_wait(command)
        return result.stdout + result.stderr
    finally:
        await sandbox.close()
```

**Pros**: Managed infrastructure, scales easily, full isolation  
**Cons**: Network latency, cost per execution

#### Security Summary

| Level | Use Case | Isolation | Latency | Cost |
|-------|----------|-----------|---------|------|
| Allowlist | Local dev | Low | ~0ms | Free |
| Docker | Production repos | High | ~100ms | Infra |
| Cloud | Multi-tenant | Maximum | ~500ms | Per-exec |

### 2. Memory Compaction

The append-only pattern keeps KV-cache coherent—but every context window has a ceiling. At 50-100 turns, you'll hit it.

**The Solution**: Compact before you hit the limit.

```python
class CompactingContextManager:
    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 100000,
        compact_threshold: float = 0.8,  # Compact at 80% capacity
        keep_recent_turns: int = 20
    ):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent_turns = keep_recent_turns
        self.turns: list[dict] = []
        self.summaries: list[str] = []

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        
        if self._estimate_tokens() > self.max_tokens * self.compact_threshold:
            self._compact()

    def _estimate_tokens(self) -> int:
        # Rough estimate: 1 token ≈ 4 characters
        total = len(self.system_prompt)
        for turn in self.turns:
            total += len(turn["content"])
        for summary in self.summaries:
            total += len(summary)
        return total // 4

    def _compact(self) -> None:
        if len(self.turns) <= self.keep_recent_turns:
            return
        
        # Split turns
        to_summarize = self.turns[:-self.keep_recent_turns]
        to_keep = self.turns[-self.keep_recent_turns:]
        
        # Generate summary using a cheap, fast model
        summary = self._generate_summary(to_summarize)
        self.summaries.append(summary)
        
        # Replace old turns with summary reference
        self.turns = to_keep

    def _generate_summary(self, turns: list[dict]) -> str:
        # In production: call a cheap model (haiku/flash)
        # Key actions, decisions made, and current state
        assistant_content = [
            t["content"][:500] for t in turns 
            if t["role"] == "assistant"
        ]
        return f"Prior context summary: {' | '.join(assistant_content[-5:])}"

    def to_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add summaries as system context
        if self.summaries:
            messages.append({
                "role": "system",
                "content": "\n".join(self.summaries)
            })
        
        # Add recent turns
        messages.extend(self.turns)
        return messages
```

**When to compact**: 
- Set `compact_threshold` to 80% of max context
- If compaction fires before turn 50, your turns are too verbose

**What to preserve**:
- The most recent 15-20 turns verbatim
- Key decisions and state from older turns
- Error evidence from failed attempts

### 3. Observability Instrumentation

An unobserved harness is one you cannot improve.

```python
import time
import functools
from dataclasses import dataclass, field
from typing import Callable
import logging

@dataclass
class HarnessMetrics:
    # Counts
    tool_calls: int = 0
    tool_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    loop_detections: int = 0
    compactions: int = 0
    
    # Latencies
    total_latency_ms: float = 0.0
    tool_latencies: dict[str, list[float]] = field(default_factory=dict)
    model_latencies: list[float] = field(default_factory=list)
    
    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls > 0 else 0.0

    @property
    def avg_tool_latency_ms(self) -> dict[str, float]:
        return {
            k: sum(v) / len(v) if v else 0
            for k, v in self.tool_latencies.items()
        }

    def report(self) -> dict:
        return {
            "tool_calls": self.tool_calls,
            "error_rate": f"{self.error_rate:.1%}",
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            "loop_detections": self.loop_detections,
            "compactions": self.compactions,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "avg_tool_latency": self.avg_tool_latency_ms,
        }


def instrument_tool(metrics: HarnessMetrics):
    """Decorator to instrument tool calls."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                metrics.tool_calls += 1
                return result
            except Exception as e:
                metrics.tool_calls += 1
                metrics.tool_errors += 1
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                metrics.total_latency_ms += elapsed_ms
                name = fn.__name__
                metrics.tool_latencies.setdefault(name, []).append(elapsed_ms)
        return wrapper
    return decorator


class ObservableHarness:
    def __init__(self):
        self.metrics = HarnessMetrics()
        self.logger = logging.getLogger("harness")
    
    def log_model_call(self, response):
        """Log model call metrics from API response."""
        usage = response.get("usage", {})
        self.metrics.input_tokens += usage.get("input_tokens", 0)
        self.metrics.output_tokens += usage.get("output_tokens", 0)
        self.metrics.cached_tokens += usage.get("cache_read_input_tokens", 0)
        
        # Update cache hit tracking
        if usage.get("cache_read_input_tokens", 0) > 0:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def log_loop_detection(self):
        self.metrics.loop_detections += 1
        self.logger.warning("Loop detected in agent execution")
    
    def log_compaction(self, turns_compacted: int):
        self.metrics.compactions += 1
        self.logger.info(f"Compacted {turns_compacted} turns")
    
    def end_session(self):
        """Log final metrics at session end."""
        report = self.metrics.report()
        self.logger.info(f"Session complete: {report}")
        return report
```

### 4. Recovery Mechanisms

Things will fail. The harness should handle failures gracefully.

```python
class RecoveryManager:
    def __init__(self):
        self.checkpoints: list[dict] = []
        self.max_checkpoints = 5
    
    def checkpoint(self, state: dict) -> None:
        """Save a recoverable state."""
        self.checkpoints.append({
            "timestamp": datetime.now().isoformat(),
            "state": state.copy(),
        })
        
        # Keep only recent checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def recover(self, steps_back: int = 1) -> dict | None:
        """Recover to a previous checkpoint."""
        if steps_back > len(self.checkpoints):
            return None
        
        return self.checkpoints[-steps_back]["state"]
    
    def handle_failure(
        self,
        error: Exception,
        context: dict
    ) -> tuple[str, dict]:
        """Determine recovery strategy based on error type."""
        
        error_type = type(error).__name__
        
        strategies = {
            "TimeoutError": self._handle_timeout,
            "RateLimitError": self._handle_rate_limit,
            "ToolError": self._handle_tool_error,
            "ValidationError": self._handle_validation,
            "ContextOverflow": self._handle_overflow,
        }
        
        handler = strategies.get(error_type, self._handle_unknown)
        return handler(error, context)
    
    def _handle_timeout(self, error, context):
        """Retry with exponential backoff."""
        attempt = context.get("attempt", 0)
        if attempt < 2:
            wait = 5 * (attempt + 1)
            return "retry", {"wait_seconds": wait, "attempt": attempt + 1}
        return "escalate", {"reason": "Repeated timeouts"}
    
    def _handle_rate_limit(self, error, context):
        """Back off and retry."""
        return "retry", {"wait_seconds": 30, "attempt": context.get("attempt", 0) + 1}
    
    def _handle_tool_error(self, error, context):
        """Try a different approach."""
        return "rethink", {
            "error": str(error),
            "suggestion": "Try a different approach to achieve the same goal"
        }
    
    def _handle_validation(self, error, context):
        """Don't retry—fix the issue."""
        return "fix", {
            "error": str(error),
            "instruction": "Fix the validation error before proceeding"
        }
    
    def _handle_overflow(self, error, context):
        """Compact and retry."""
        return "compact", {
            "instruction": "Context too large. Compacting."
        }
    
    def _handle_unknown(self, error, context):
        """Escalate unknown errors."""
        return "escalate", {
            "error": str(error),
            "type": type(error).__name__
        }
```

### 5. Cost Management

Agent economics are dominated by a few key factors:

#### Token Economics

| Component | Impact | Optimization |
|-----------|--------|--------------|
| System prompt | Fixed cost per call | Keep under 2000 tokens |
| Context | Grows with session | Compact aggressively |
| Tool outputs | Often verbose | Truncate/summarize |
| Reasoning | 10x+ normal tokens | Use reasoning sandwich |

#### KV-Cache Economics (Reminder)

- Cached tokens: **$0.30/MTok**
- Uncached tokens: **$3.00/MTok**
- 10x difference!

```python
class CostTracker:
    # Example pricing (adjust for your provider)
    PRICING = {
        "input": 3.00 / 1_000_000,      # $3/MTok
        "output": 15.00 / 1_000_000,    # $15/MTok
        "cached": 0.30 / 1_000_000,     # $0.30/MTok
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.calls = []
    
    def track_call(self, usage: dict) -> float:
        input_cost = usage.get("input_tokens", 0) * self.PRICING["input"]
        output_cost = usage.get("output_tokens", 0) * self.PRICING["output"]
        
        # Subtract cached tokens from input cost
        cached = usage.get("cache_read_input_tokens", 0)
        cache_savings = cached * (self.PRICING["input"] - self.PRICING["cached"])
        
        call_cost = input_cost + output_cost - cache_savings
        self.total_cost += call_cost
        
        self.calls.append({
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cached": cached,
            "cost": call_cost,
        })
        
        return call_cost
    
    def report(self) -> dict:
        if not self.calls:
            return {"total_cost": 0, "calls": 0}
        
        total_input = sum(c["input"] for c in self.calls)
        total_cached = sum(c["cached"] for c in self.calls)
        
        return {
            "total_cost": f"${self.total_cost:.4f}",
            "calls": len(self.calls),
            "cache_hit_rate": f"{total_cached / total_input:.1%}" if total_input else "N/A",
            "avg_cost_per_call": f"${self.total_cost / len(self.calls):.4f}",
        }
```

#### Cost Optimization Strategies

1. **Maximize cache hits**: Follow the four rules from Module 1
2. **Compact early**: Don't wait until context is full
3. **Truncate tool outputs**: Most tools return more than needed
4. **Use reasoning sandwich**: xhigh only where it matters
5. **Set hard budgets**: Fail fast when cost exceeds threshold

```python
class BudgetEnforcer:
    def __init__(self, max_cost: float, max_turns: int):
        self.max_cost = max_cost
        self.max_turns = max_turns
        self.cost_tracker = CostTracker()
        self.turn_count = 0
    
    def check_budget(self) -> tuple[bool, str]:
        if self.cost_tracker.total_cost >= self.max_cost:
            return False, f"Cost budget exceeded: ${self.cost_tracker.total_cost:.2f}"
        
        if self.turn_count >= self.max_turns:
            return False, f"Turn budget exceeded: {self.turn_count} turns"
        
        return True, ""
    
    def on_turn_complete(self, usage: dict):
        self.cost_tracker.track_call(usage)
        self.turn_count += 1
        
        ok, reason = self.check_budget()
        if not ok:
            raise BudgetExceededError(reason)
```

---

## Production Hardening Checklist

- [ ] Bash tool sandboxed (allowlist minimum, Docker for production)
- [ ] Memory compaction configured with appropriate thresholds
- [ ] Observability instrumented (metrics, logging, tracing)
- [ ] Recovery mechanisms for common failure types
- [ ] Cost tracking and budget enforcement
- [ ] Checkpointing for long-running tasks
- [ ] Graceful degradation on budget exhaustion

---

## Exercises

### Exercise 5.1: Implement Sandboxing
Build a sandboxed bash tool with:
1. Command allowlist
2. Timeout handling
3. Output truncation

### Exercise 5.2: Memory Compaction
Implement a compacting context manager that:
1. Tracks token usage
2. Compacts at 80% capacity
3. Preserves recent context
4. Generates summaries

### Exercise 5.3: Observability Dashboard
Create metrics collection that tracks:
1. Cache hit rate
2. Tool latencies
3. Error rates
4. Cost per session

### Exercise 5.4: Recovery Strategy
Design recovery handlers for:
1. Network timeouts
2. Rate limits
3. Tool errors
4. Context overflow

---

## Key Takeaways

1. **Sandbox everything**: Never run untrusted code without isolation

2. **Compact before overflow**: Don't wait until context is full

3. **Observe to improve**: You can't optimize what you don't measure

4. **Plan for failure**: Recovery mechanisms turn crashes into continuations

5. **Track costs religiously**: Agent economics can surprise you

---

## Next Module

Continue to **[Module 6: Case Studies](../06_case_studies/README.md)** to learn from real-world implementations.

---

## References

- Manus. "Context Engineering for AI Agents" - Memory and cost management
- OpenAI. "Building Reliable Agents with Memory and Compaction"
- E2B Documentation - Cloud sandbox patterns
- Docker SDK for Python - Container isolation
