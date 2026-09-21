# Module 4: Looped & Adaptive Transformers

## Beyond Fixed Depth

Standard transformers have fixed depth:
```
x → Layer 1 → Layer 2 → ... → Layer N → output
```

**Problem**: All inputs get the same compute, regardless of difficulty.

**Looped transformers**: Iterate a block until "done thinking"

```
x → Block → Block → ... → Block → output
    (repeat until converged or max iterations)
```

## The Vision

Simple problem: 2 iterations
```
"What is 2+2?" → Block → Block → "4"
```

Hard problem: 10 iterations
```
"Prove that √2 is irrational" → Block ×10 → [Proof]
```

**Adaptive computation** based on problem difficulty.

## Historical Context

### Universal Transformer (2018)
First looped transformer:
- Share weights across "layers"
- Dynamic halting mechanism
- Showed promise but training challenges

### Pondering (ACT)
Adaptive Computation Time:
```python
def adaptive_forward(x, max_steps=20):
    halting_prob = 0
    output = 0
    
    for step in range(max_steps):
        h = block(x)
        p = halt_probability(h)  # Learned: "should we stop?"
        
        halting_prob += p
        output += p * h
        
        if halting_prob > 1 - epsilon:
            break
    
    return output
```

### Modern Looped Transformers (2025-2026)
Recent breakthroughs in training stability:
- Better initialization
- Careful normalization
- Curriculum learning

## Why Looping Matters for Reasoning

**Theoretical result**: Looped transformers can solve problems that require more steps than layers, by iterating.

```
Fixed depth:  Can solve problems up to depth N
Looped:       Can solve problems up to depth max_iterations
              (with same # of parameters!)
```

**Example**: Sorting
- Fixed 12-layer transformer: struggles with long lists
- 2-layer looped transformer (10 iterations): generalizes better

## Training Challenges

### Gradient Issues
Long loops → vanishing/exploding gradients

**Solutions**:
- Careful initialization
- Skip connections across iterations
- Gradient clipping

### Halting
When should the model stop?
- Learned halting probability
- Fixed maximum + early stopping
- Confidence-based termination

### Curriculum Learning
```
Start: Easy problems (few iterations needed)
Gradually: Harder problems (more iterations)
```

## Current State

**Research status**: Active area, not yet mainstream
**Production status**: o1/DeepSeek-R1 use related ideas (internal reasoning loops)

The gap: Explicit looped architectures vs implicit reasoning traces

## Connection to System 2

Looped transformers are one **architectural approach** to System 2 thinking:
- Standard transformer = System 1 (fixed compute)
- Looped transformer = System 2 (adaptive compute)

But current production systems (o1) achieve System 2 through **prompting** (generate reasoning tokens) rather than **architecture** (loop the forward pass).

## Exercises

1. Implement a simple universal transformer
2. Compare fixed-depth vs looped on an algorithmic task
3. Experiment with different halting mechanisms

## Resources

- "Universal Transformers" (Dehghani et al., 2018)
- "Looped Transformers for Length Generalization" (2025)
- "Reasoning with Latent Thoughts" (2025)

## What's Next?

Module 5 explores **State Space Models** — a fundamentally different architecture challenging transformers.
