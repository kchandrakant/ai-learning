# Demo: Harness Engineering in Action

## Overview

This directory contains demonstrations of harness engineering concepts, bringing together all the components we learned. Each demo is a hands-on example you can run and modify.

---

## Demos

### Demo 1: Basic Agent Harness
**See a minimal harness in action**

Build a basic harness with:
- System prompt assembly
- Tool definitions
- Simple loop execution
- Output verification

```bash
python demo/basic_harness_demo.py
```

---

### Demo 2: Context Engineering
**Visualize context management**

Demonstrates:
- Environment map injection
- todo.md pattern in action
- KV-cache friendly serialization
- Context compaction

```bash
python demo/context_demo.py
```

---

### Demo 3: Tool Routing
**See dynamic tool selection**

Demonstrates:
- Embedding-based tool routing
- Tool consolidation (many → few)
- Permission matrix enforcement

```bash
python demo/tool_routing_demo.py
```

---

### Demo 4: Constraint Enforcement
**Watch constraints improve reliability**

Demonstrates:
- Reasoning sandwich (xhigh → high → xhigh)
- Loop detection triggering intervention
- Pre-completion verification gate
- Budget warnings

```bash
python demo/constraints_demo.py
```

---

### Demo 5: Behavioral Evaluation
**Build and run behavioral tests**

Demonstrates:
- Writing behavioral assertions
- Testing tool call sequences
- Batch evaluation for non-determinism
- Metrics collection

```bash
python demo/eval_demo.py
```

---

### Demo 6: Self-Improving Harness
**See a harness improve itself**

Demonstrates:
- Weakness mining from traces
- Bounded proposal generation
- Regression-safe validation
- Accept/reject loop

```bash
python demo/self_improve_demo.py
```

---

### Demo 7: End-to-End Coding Agent
**Complete coding agent harness**

A full example combining all concepts:
- LocalContextMiddleware
- Tool rationalization
- Reasoning sandwich
- Loop detection
- Verification gate
- Observability

```bash
python demo/coding_agent_demo.py
```

---

## Expected Outputs

Each demo generates:
- Console output explaining what's happening step-by-step
- Metrics (cache hit rate, tool calls, latency)
- Visualizations saved as PNG files (where applicable)
- Traces in JSON format for analysis

---

## Files

```
demo/
├── basic_harness_demo.py       # Minimal working harness
├── context_demo.py             # Context engineering patterns
├── tool_routing_demo.py        # Dynamic tool selection
├── constraints_demo.py         # Constraint enforcement
├── eval_demo.py                # Behavioral evaluation
├── self_improve_demo.py        # Self-improving loop
├── coding_agent_demo.py        # Complete coding agent
│
├── utils/
│   ├── mock_llm.py             # Mock LLM for demos
│   ├── mock_tools.py           # Sample tool implementations
│   └── visualize.py            # Visualization helpers
│
└── outputs/                    # Generated outputs
    ├── traces/                 # Execution traces
    └── visualizations/         # Charts and diagrams
```

---

## What You'll Learn

1. **How harness components work together** in a real system
2. **How to debug harness issues** using traces and metrics
3. **The impact of each technique** through before/after comparisons
4. **How to extend** these patterns for your own use cases

---

## Prerequisites

Complete Modules 0-6 first to understand all the components!

For the self-improvement demo (Demo 6), also complete Module 7.

---

## Running the Demos

### Setup

```bash
# Ensure you're in the harness-engineering directory
cd harness-engineering

# Activate your virtual environment
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Run any demo
python demo/<demo_name>.py

# Run with verbose output
python demo/<demo_name>.py --verbose

# Run with mock LLM (no API key needed)
python demo/<demo_name>.py --mock
```

---

## Demo Configurations

Each demo can be configured via command-line arguments or config files:

```bash
# Use a specific model
python demo/coding_agent_demo.py --model claude-3-5-sonnet

# Set max turns
python demo/coding_agent_demo.py --max-turns 20

# Enable detailed tracing
python demo/coding_agent_demo.py --trace

# Save outputs to specific directory
python demo/coding_agent_demo.py --output-dir ./my_outputs
```

---

## Extending the Demos

### Adding Your Own Demo

1. Create a new file in `demo/`
2. Import the harness components you need
3. Set up your task and configuration
4. Run and collect metrics

```python
# Template for a new demo
from harness import Harness, LocalContextMiddleware, LoopDetector

def main():
    # Setup
    harness = Harness(
        model="claude-3-5-sonnet",
        tools=[...],
        middleware=[LocalContextMiddleware(), LoopDetector()],
    )
    
    # Run task
    result = harness.run("Your task here")
    
    # Report metrics
    print(harness.metrics.report())

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API key error | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` env var, or use `--mock` |
| Import error | Run `pip install -r requirements.txt` |
| Slow execution | Use `--mock` for testing without API calls |
| No output | Add `--verbose` flag |

---

## Next Steps

After running the demos:
1. Modify parameters and see how behavior changes
2. Add your own tools and test them
3. Try different constraint configurations
4. Build a harness for your own use case
