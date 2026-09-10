# Step 11: Observability & Debugging

## The Debugging Challenge

Agent failures are hard to diagnose:
- Non-deterministic behavior
- Multi-step chains
- Tool interactions
- Context dependencies

**Solution:** Comprehensive observability.

## Observability Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Logging    → What happened (events, decisions)            │
│   Tracing    → How it flowed (step-by-step path)            │
│   Metrics    → How it performed (latency, cost, success)    │
│   Replay     → Reproduce issues (deterministic re-run)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Structured Logging

```python
import structlog
from datetime import datetime

logger = structlog.get_logger()

class AgentLogger:
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.log = logger.bind(
            agent_id=agent_id,
            session_id=session_id
        )
    
    def log_thought(self, thought: str, context: dict = None):
        self.log.info(
            "agent_thought",
            thought=thought,
            context=context,
            timestamp=datetime.now().isoformat()
        )
    
    def log_action(self, action: str, args: dict, result: Any):
        self.log.info(
            "agent_action",
            action=action,
            arguments=args,
            result=str(result)[:500],  # Truncate long results
            timestamp=datetime.now().isoformat()
        )
    
    def log_error(self, error: Exception, context: dict = None):
        self.log.error(
            "agent_error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            timestamp=datetime.now().isoformat()
        )
```

## Tracing

### Custom Trace Implementation

```python
from dataclasses import dataclass, field
from typing import Any
import uuid

@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = None
    start_time: datetime = None
    end_time: datetime = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    
    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

class Tracer:
    def __init__(self):
        self.spans: list[Span] = []
        self.current_span: Span = None
    
    def start_span(self, name: str) -> Span:
        span = Span(
            name=name,
            parent_id=self.current_span.span_id if self.current_span else None,
            start_time=datetime.now()
        )
        self.spans.append(span)
        self.current_span = span
        return span
    
    def end_span(self, status: str = "ok"):
        if self.current_span:
            self.current_span.end_time = datetime.now()
            self.current_span.status = status
            # Find parent
            if self.current_span.parent_id:
                for span in self.spans:
                    if span.span_id == self.current_span.parent_id:
                        self.current_span = span
                        return
            self.current_span = None
    
    def get_trace(self) -> list[Span]:
        return self.spans
```

### Using the Tracer

```python
class TracedAgent:
    def __init__(self, agent, tracer: Tracer):
        self.agent = agent
        self.tracer = tracer
    
    def run(self, task: str) -> str:
        root_span = self.tracer.start_span("agent_run")
        root_span.set_attribute("task", task)
        
        try:
            # Planning
            plan_span = self.tracer.start_span("planning")
            plan = self.agent.plan(task)
            plan_span.set_attribute("plan_steps", len(plan))
            self.tracer.end_span()
            
            # Execution
            for i, step in enumerate(plan):
                step_span = self.tracer.start_span(f"step_{i}")
                step_span.set_attribute("action", step["action"])
                
                result = self.agent.execute_step(step)
                step_span.add_event("step_complete", {"result": str(result)[:100]})
                
                self.tracer.end_span()
            
            self.tracer.end_span("ok")
            return result
            
        except Exception as e:
            root_span.add_event("error", {"message": str(e)})
            self.tracer.end_span("error")
            raise
```

## Integration with LangSmith

```python
from langsmith import Client
from langchain.callbacks import LangChainTracer

# Setup
client = Client()
tracer = LangChainTracer(project_name="my-agent-project")

# Use with LangChain agent
result = agent.invoke(
    {"input": "Research AI agents"},
    config={"callbacks": [tracer]}
)

# Access traces in LangSmith UI or API
runs = client.list_runs(project_name="my-agent-project")
```

## Metrics Collection

```python
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = {}
    
    def increment(self, name: str, value: int = 1):
        self.counters[name] += value
    
    def time(self, name: str):
        """Context manager for timing operations."""
        return Timer(self, name)
    
    def gauge(self, name: str, value: float):
        self.gauges[name] = value
    
    def get_summary(self) -> dict:
        return {
            "counters": dict(self.counters),
            "timers": {
                k: {
                    "count": len(v),
                    "avg_ms": sum(v) / len(v) * 1000 if v else 0,
                    "max_ms": max(v) * 1000 if v else 0
                }
                for k, v in self.timers.items()
            },
            "gauges": self.gauges
        }

class Timer:
    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.collector.timers[self.name].append(elapsed)

# Usage
metrics = MetricsCollector()

with metrics.time("llm_call"):
    response = llm(prompt)

metrics.increment("tool_calls")
metrics.gauge("context_tokens", 1500)
```

## Replay System

```python
import json
from pathlib import Path

class ReplaySystem:
    def __init__(self, storage_dir: str = "./replays"):
        self.storage = Path(storage_dir)
        self.storage.mkdir(exist_ok=True)
    
    def record(self, session_id: str, events: list[dict]):
        """Save session for replay."""
        path = self.storage / f"{session_id}.json"
        with open(path, 'w') as f:
            json.dump(events, f, indent=2)
    
    def replay(self, session_id: str, agent) -> list[dict]:
        """Replay a session with deterministic inputs."""
        path = self.storage / f"{session_id}.json"
        with open(path) as f:
            events = json.load(f)
        
        results = []
        for event in events:
            if event["type"] == "llm_call":
                # Mock LLM response with recorded response
                result = event["response"]
            elif event["type"] == "tool_call":
                # Actually execute tool (or mock)
                result = agent.execute_tool(event["tool"], event["args"])
            
            results.append({
                "event": event,
                "replay_result": result,
                "match": result == event.get("result")
            })
        
        return results
```

## Debug Dashboard

```python
class DebugDashboard:
    def __init__(self, tracer: Tracer, metrics: MetricsCollector):
        self.tracer = tracer
        self.metrics = metrics
    
    def generate_report(self) -> str:
        """Generate a debug report for the current session."""
        trace = self.tracer.get_trace()
        metrics_summary = self.metrics.get_summary()
        
        report = ["=" * 60, "AGENT DEBUG REPORT", "=" * 60, ""]
        
        # Metrics
        report.append("## Metrics")
        report.append(f"Total LLM calls: {metrics_summary['counters'].get('llm_calls', 0)}")
        report.append(f"Total tool calls: {metrics_summary['counters'].get('tool_calls', 0)}")
        
        if 'llm_call' in metrics_summary['timers']:
            llm_timing = metrics_summary['timers']['llm_call']
            report.append(f"Avg LLM latency: {llm_timing['avg_ms']:.0f}ms")
        
        # Trace
        report.append("\n## Execution Trace")
        for span in trace:
            indent = "  " if span.parent_id else ""
            status_icon = "✓" if span.status == "ok" else "✗"
            report.append(f"{indent}{status_icon} {span.name}")
            if span.attributes:
                for k, v in span.attributes.items():
                    report.append(f"{indent}    {k}: {v}")
        
        # Errors
        errors = [s for s in trace if s.status == "error"]
        if errors:
            report.append("\n## Errors")
            for span in errors:
                for event in span.events:
                    if event["name"] == "error":
                        report.append(f"  - {span.name}: {event['attributes'].get('message')}")
        
        return "\n".join(report)
```

## Files

- `observability.py` - Complete observability stack

## Key Takeaways

1. Log thoughts, actions, and errors with structure
2. Trace every step for debugging
3. Collect metrics (latency, cost, success)
4. Enable replay for debugging
5. Integrate with tools like LangSmith

## What's Next?

Step 12: **Safety & Guardrails** — keeping agents under control.
