# Module 5: State Space Models (Mamba)

## The Problem with Attention

Attention is O(n²) in sequence length:
```
10 tokens:    100 comparisons
100 tokens:   10,000 comparisons
10,000 tokens: 100,000,000 comparisons
```

For long sequences, this becomes prohibitive.

## State Space Models

SSMs process sequences as linear recurrences:
```
h_t = A·h_{t-1} + B·x_t    (update hidden state)
y_t = C·h_t                 (produce output)
```

This is O(n) — linear in sequence length!

## From S4 to Mamba

### S4 (2021)
Structured State Spaces: 
- Careful initialization of A, B, C matrices
- Enables very long sequence modeling
- But: parameters are fixed (input-independent)

### Mamba (2023)
Selective State Spaces:
- A, B, C depend on the input (selective)
- Input-dependent dynamics = attention-like behavior
- But still O(n) complexity

```python
# Conceptual Mamba block
def mamba_block(x):
    # Input-dependent parameters
    delta = linear(x)  # Controls how much state changes
    B = linear(x)      # Input projection
    C = linear(x)      # Output projection
    
    # Selective scan (efficient CUDA kernel)
    h = selective_scan(x, delta, A, B, C)
    
    return h
```

## Mamba vs Transformer

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Complexity | O(n²) | O(n) |
| Long sequences | Expensive | Efficient |
| In-context learning | Strong | Good |
| Precise recall | Strong | Weaker |
| Parallelization | High | Moderate |

### Where Mamba Wins
- Very long sequences (100K+ tokens)
- Streaming inference
- Memory-constrained settings

### Where Transformers Win
- Tasks requiring precise recall ("what was word 7?")
- Strong in-context learning
- Tasks with training data

## Mamba-2

Improvements (2024):
- Reformulated for better hardware utilization
- Closer connection to attention (state-space duality)
- Easier to train

## Practical Usage

```python
from mamba_ssm import Mamba

# Create Mamba model
model = Mamba(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2,
)

# Forward pass
output = model(input_sequence)
```

**Note**: Requires CUDA for efficient kernels.

## Hybrid Architectures

Why choose? Combine both:
- Use Mamba for bulk of layers (efficiency)
- Sprinkle attention layers for precise recall
- Example: Jamba (AI21 Labs)

## The Verdict

Mamba is **not a transformer killer** — it's a **complement**.

Best approach depends on:
- Sequence length
- Task requirements
- Hardware constraints
- Latency requirements

## Exercises

1. Compare Mamba vs Transformer on a long-context task
2. Measure throughput on different sequence lengths
3. Experiment with hybrid architectures

## Resources

- "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- "Mamba-2" (2024)
- mamba-ssm GitHub repository

## What's Next?

Module 6 explores **Hybrid Architectures** — combining SSMs and transformers.
