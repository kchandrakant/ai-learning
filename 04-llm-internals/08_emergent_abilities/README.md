# Module 8: Emergent Abilities

What appears at scale.

## Overview

Some capabilities appear suddenly as models scale. Understanding emergence helps predict and leverage model behavior.

## Key Topics

### What is Emergence?
```
Emergence: Abilities absent in small models, present in large ones
- Not gradual improvement
- Sharp transition at some scale
- Often surprising
```

### Classic Examples
```
In-context learning: Few-shot prompting works
Chain-of-thought:    Step-by-step reasoning helps
Arithmetic:          3-digit addition suddenly works
Code execution:      Can trace through code mentally
```

### The Debate
```
Are emergent abilities real?
- Some argue: metric artifacts (discontinuous metrics)
- Others: real phase transitions in learning
- Truth: probably both, depends on the task
```

### Grokking
```
Model memorizes, then suddenly generalizes
- Loss flat, then drops
- Happens after "overfitting"
- Related to emergence? Unclear
```

### Capability Overhang
```
Models may have latent capabilities
- Unlocked by prompting techniques
- Or fine-tuning on examples
- Implications for safety
```

## Exercises

1. Test scaling on arithmetic tasks
2. Find emergence thresholds for specific tasks
3. Experiment with prompting to unlock capabilities

## Key Insight

Emergence means larger models may have qualitatively new capabilities. Test, don't assume.
