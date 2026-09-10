# Module 7: Architecture Decisions

Design choices in modern LLMs.

## Overview

Modern LLMs differ from the original transformer. Understanding these changes reveals hard-won lessons.

## Key Topics

### Attention Variants
```
MHA:  8 heads = 8Q, 8K, 8V (original)
GQA:  8 heads = 8Q, 2K, 2V (grouped query)
MQA:  8 heads = 8Q, 1K, 1V (multi-query)

GQA/MQA reduce KV cache size → faster inference
```

### Activation Functions
```
ReLU (original)     → zeros for negative inputs
GELU (GPT-2)        → smooth, probabilistic
SwiGLU (LLaMA)      → gated, best empirical results

SwiGLU: Swish(xW₁) ⊙ (xW₂)
```

### Normalization
```
LayerNorm (original): normalize, then scale/shift
RMSNorm (LLaMA):      only normalize by RMS

RMSNorm: x / √(mean(x²) + ε) × γ
Faster, empirically equivalent
```

### Position Encodings
```
Sinusoidal (original): Fixed, absolute positions
Learned (GPT):         Trained embeddings
RoPE (LLaMA):          Rotary, encodes relative position
ALiBi:                 Bias attention by distance

RoPE enables length extrapolation
```

### Mixture of Experts (MoE)
```
Multiple FFN "experts", router selects top-k
- More parameters, same compute per token
- Mixtral: 8 experts, 2 active
- Challenge: load balancing across experts
```

## Exercises

1. Implement GQA and compare KV cache size
2. Visualize RoPE rotation
3. Build a simple MoE layer

## Key Insight

Every choice is a trade-off. Modern architectures optimize for inference efficiency, not just quality.
