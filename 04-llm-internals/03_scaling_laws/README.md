# Module 3: Scaling Laws

The science of scaling language models.

## Overview

Scaling laws tell us how to allocate compute between model size and data. They transformed LLM development from art to science.

## Key Topics

### The Chinchilla Finding
```
For compute-optimal training:
- N (parameters) and D (data tokens) should scale together
- Most pre-Chinchilla models were undertrained

Optimal ratio: ~20 tokens per parameter
70B model → ~1.4T tokens
```

### The Power Law
```
L(N, D) ≈ (N_c/N)^α + (D_c/D)^β + L_∞

Loss decreases predictably with more compute.
This enables planning training runs in advance.
```

### Over-training for Inference
```
Modern approach: Train smaller models on MORE data
- LLaMA-7B trained on 1T tokens (not "optimal")
- Inference cost matters more than training cost
- Smaller models cheaper to serve
```

### When Scaling Breaks
- Scaling doesn't fix all problems (reasoning, factuality)
- Returns diminish at some point
- Data quality eventually matters more than quantity

## Exercises

1. Visualize loss vs compute curves
2. Calculate optimal allocation for a budget
3. Compare Chinchilla-optimal vs over-trained models

## Key Insight

Scaling laws made LLM development predictable. You can forecast performance before training.
