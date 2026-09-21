# Module 1: System 1 vs System 2 Thinking

## Two Modes of Cognition

Daniel Kahneman's "Thinking, Fast and Slow" describes two cognitive systems:

**System 1**: Fast, intuitive, automatic
- Pattern recognition
- Immediate responses
- Low effort

**System 2**: Slow, deliberate, effortful
- Complex reasoning
- Step-by-step analysis
- High cognitive load

## Applied to AI

### System 1 AI (Traditional LLM)
```
Input → Model → Output (one forward pass)
```
- Fast, cheap
- Good for pattern matching
- Struggles with complex reasoning

### System 2 AI (Reasoning Models)
```
Input → Think → Verify → Revise → Think more → Output
```
- Slow, expensive
- Better for hard problems
- Can verify and correct mistakes

## The Key Insight

**System 2 trades inference compute for reasoning quality.**

Some problems benefit more from "thinking longer" than from larger models:
- Math word problems
- Code debugging
- Multi-step reasoning
- Planning tasks

## Examples in Practice

| Model | Approach | When to Use |
|-------|----------|-------------|
| GPT-4 | System 1 (fast) | Quick responses, simple tasks |
| o1 | System 2 (reasoning) | Math, coding, complex analysis |
| Claude | Adjustable | Can request extended thinking |
| DeepSeek-R1 | System 2 | Reasoning-heavy tasks |

## The Compute Tradeoff

```
Standard inference:  100ms, $0.001
Extended thinking:   10s,   $0.05

50× more time, 50× more cost
But: 10-30% accuracy improvement on hard tasks
```

## When to Use Which?

**System 1 (fast)**:
- Simple queries
- Latency-sensitive applications
- High-volume, low-complexity
- Creative generation

**System 2 (reasoning)**:
- Math and logic problems
- Code debugging
- Multi-step planning
- Tasks requiring verification

## Exercises

1. Compare GPT-4 vs o1 on math word problems
2. Measure latency vs accuracy tradeoffs
3. Identify tasks in your domain that would benefit from System 2

## Resources

- "Thinking, Fast and Slow" — Daniel Kahneman
- OpenAI o1 system card
- DeepSeek-R1 technical report

## What's Next?

Module 2 explores **Test-Time Compute** — the technical mechanisms behind System 2 reasoning.
