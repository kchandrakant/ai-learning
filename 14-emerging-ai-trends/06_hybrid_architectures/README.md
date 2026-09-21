# Module 6: Hybrid Architectures

## Why Hybrid?

SSMs and transformers have complementary strengths:

| Capability | Transformer | Mamba |
|------------|-------------|-------|
| Long sequences | Expensive | Efficient |
| Precise recall | Strong | Weaker |
| In-context learning | Strong | Good |
| Training parallelism | High | Moderate |

**Hybrid insight**: Use each where it's strongest.

## Hybrid Patterns

### Interleaved Layers
Alternate between Mamba and attention:

```
Input → Mamba → Attention → Mamba → Attention → ... → Output
```

**Example**: Jamba (AI21 Labs)
- 1 attention layer for every 7 Mamba layers
- Gets efficiency of Mamba
- Precision of attention where needed

### Parallel Branches
Run both in parallel, merge results:

```
        ┌─ Mamba branch ──┐
Input ─┤                   ├─ Merge → Output
        └─ Attention branch┘
```

### Selective Routing
Route different tokens to different modules:

```python
def hybrid_forward(x):
    # Decide which tokens need precise recall
    needs_attention = router(x)
    
    # Process accordingly
    attn_output = attention(x[needs_attention])
    mamba_output = mamba(x[~needs_attention])
    
    return merge(attn_output, mamba_output)
```

## Jamba: A Case Study

AI21's Jamba (2024):
- 52B active parameters (256B total with MoE)
- Mamba-dominant architecture
- Sparse attention layers
- 256K context length

**Architecture**:
```
Jamba Block = Mamba layer × 6 + Attention layer × 1 + MoE FFN
Stack multiple Jamba blocks
```

**Results**:
- Matches transformer quality
- 3× throughput on long sequences
- 8× batch size on same hardware

## Implementation Considerations

### Memory Management
Different modules have different memory patterns:
- Attention: KV cache grows with context
- Mamba: Fixed state size

Hybrid can optimize total memory usage.

### Batching
Mixed architectures complicate batching:
- Attention batches nicely
- Mamba has sequential dependencies

### Training
Need to balance:
- Learning rate for different components
- Gradient scaling
- Initialization

## When to Consider Hybrids

**Good fit**:
- Very long contexts (>100K tokens)
- Need both efficiency and precision
- Memory-constrained with long inputs

**Probably overkill**:
- Standard context lengths (<32K)
- Simpler task requirements
- When pure transformer is fast enough

## The Future

Trends suggest:
1. **More sophisticated routing**: Learn when to use what
2. **Task-specific hybrids**: Different ratios for different domains
3. **Novel combinations**: Beyond just Mamba + Attention

## Exercises

1. Compare Jamba vs transformer-only on long-context tasks
2. Experiment with different interleaving ratios
3. Measure memory and throughput tradeoffs

## Resources

- "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21, 2024)
- "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- Hugging Face model implementations

## What's Next?

Module 7 covers **Tabular Transformers** — applying transformers to structured data.
