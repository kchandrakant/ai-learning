# Module 4: Error Recovery & Resilience

Handling failures gracefully in agent systems.

## Overview

Agent systems interact with unreliable external services, execute non-deterministic model calls, and operate in dynamic environments. Robust error recovery is essential for production systems.

## Key Topics

### Error Classification
- **Recoverable errors:** Network timeouts, rate limits, malformed responses
- **Terminal errors:** Authentication failures, resource not found, budget exhausted
- **Ambiguous errors:** Model refusals, unclear tool outputs

### Retry Strategies
```python
# Exponential backoff with jitter
def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except RecoverableError as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
    raise MaxRetriesExceeded()
```

### Checkpoint & Resume
- Save agent state at key decision points
- Enable resumption from last checkpoint on failure
- Idempotent tool calls for safe retry

### Fallback Chains
```
Primary model (GPT-4) → Timeout → Fallback (GPT-3.5) → Failure → Cached response
```

### Error Evidence Preservation
- Log full context at failure point
- Preserve tool call history for debugging
- Enable replay of failed trajectories

## Exercises

1. Implement exponential backoff for API calls
2. Build a checkpoint system for multi-step agents
3. Design a fallback chain with model routing
4. Create an error classification system

## Key Insight

Plan for failure. Every tool call can fail; every model response can be malformed. The question isn't *if* errors occur, but *how* the system responds.
