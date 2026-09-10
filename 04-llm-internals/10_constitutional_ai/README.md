# Module 10: Constitutional AI

Self-supervision for alignment.

## Overview

Constitutional AI reduces reliance on human labels by having models critique and revise their own outputs.

## Key Topics

### The Core Idea
```
Instead of human labeling every response:
1. Model generates response
2. Model critiques response (using principles)
3. Model revises based on critique
4. Use revised responses for training
```

### Constitutional Principles
```
Examples:
- "Please choose the response that is most helpful"
- "Choose the response that is least harmful"
- "Choose the response that is most honest"

Principles encode human values without per-example labels
```

### The CAI Pipeline
```
1. Red Teaming: Generate harmful prompts
2. Initial Response: Model responds (may be harmful)
3. Critique: "Identify problems with this response"
4. Revision: "Write an improved response"
5. Repeat: Multiple rounds of revision
6. RLAIF: Train on AI preferences, not human preferences
```

### RLAIF
```
RL from AI Feedback:
- AI model ranks responses using principles
- Train reward model on AI preferences
- RL as usual

Trade-off: Scalable but may inherit AI biases
```

### Comparison to RLHF
```
RLHF:  Human labels → expensive, limited
CAI:   AI labels → scalable, principle-based
Hybrid: Both, AI for scale, human for validation
```

## Exercises

1. Implement critique-revision loop
2. Write constitutional principles for a task
3. Compare RLAIF vs RLHF on a small scale

## Key Insight

Constitutional AI makes alignment more scalable by encoding values in principles, not examples.
