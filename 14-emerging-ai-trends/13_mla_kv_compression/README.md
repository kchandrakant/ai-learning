# Module 13: Multi-head Latent Attention (MLA)

## The KV Cache Problem

During autoregressive generation:
- Must store Keys and Values for all previous tokens
- Memory grows linearly with context length
- For long contexts, KV cache dominates memory

```
100K context, 32 layers, 32 heads, 128 dim, fp16:
KV cache ≈ 100K × 32 × 32 × 128 × 2 × 2 = 52GB

Just for KV cache! Plus model weights.
```

## Existing Solutions

### GQA (Grouped Query Attention)
Share K,V across groups of query heads:
- MHA: 32 heads = 32Q, 32K, 32V
- GQA: 32 heads = 32Q, 8K, 8V (4× reduction)

### MQA (Multi-Query Attention)
Extreme sharing:
- 32 heads = 32Q, 1K, 1V (32× reduction)
- Quality loss is noticeable

## MLA: A Different Approach

**DeepSeek-V2's innovation**: Compress KV into low-rank latent vectors.

Instead of storing full K,V:
```
Standard: Store K (d_head × n_heads) and V (d_head × n_heads)

MLA: Store compressed latent c (d_latent << d_head × n_heads)
     Decompress on-the-fly: K = W_K @ c, V = W_V @ c
```

### The Architecture

```
Input x
    ↓
c = Compress(x)    # Low-dimensional latent
    ↓
Store c in KV cache (small!)
    ↓
K = W_K @ c        # Decompress to keys
V = W_V @ c        # Decompress to values
    ↓
Standard attention with K, V
```

### Why It Works

Low-rank assumption: K and V have redundant information across heads.

```
Original KV: [K1, K2, ..., K32, V1, V2, ..., V32]
             = 64 × d_head dimensional

Compressed:  c = 1 × d_latent dimensional (d_latent << 64 × d_head)

The information is preserved, just represented compactly.
```

## DeepSeek-V2 Results

- **93% KV cache reduction** (11× smaller)
- **Quality preserved**: Matches larger models
- **Efficiency**: Longer contexts, larger batches

## Tradeoffs

**Advantages:**
- Dramatic memory savings
- Enables longer contexts
- Larger batch sizes

**Disadvantages:**
- Extra compute for decompression
- Training complexity
- Harder to implement

## When to Use

MLA makes sense when:
- KV cache is the bottleneck
- Very long contexts needed
- Memory-constrained deployment
- Batch size limited by KV cache

## Implementation Sketch

```python
class MLAAttention(nn.Module):
    def __init__(self, d_model, d_latent, n_heads):
        self.compress = nn.Linear(d_model, d_latent)
        self.W_K = nn.Linear(d_latent, d_model)
        self.W_V = nn.Linear(d_latent, d_model)
        # ... rest of attention
    
    def forward(self, x, kv_cache=None):
        # Compress to latent
        c = self.compress(x)
        
        # Store compressed representation
        if kv_cache is not None:
            kv_cache.append(c)
        
        # Decompress for attention
        K = self.W_K(c)
        V = self.W_V(c)
        
        # Standard attention from here
        ...
```

## Comparison

| Method | KV Cache Size | Quality | Complexity |
|--------|---------------|---------|------------|
| MHA | 1× (baseline) | Best | Simple |
| GQA | 0.25× | Good | Simple |
| MQA | 0.03× | Degraded | Simple |
| MLA | 0.09× | Good | Complex |

## Exercises

1. Calculate KV cache size for different architectures
2. Implement a simplified MLA layer
3. Benchmark memory usage vs quality tradeoff

## Resources

- "DeepSeek-V2: A Strong, Economical, Efficient MoE" (2024)
- "GQA: Training Generalized Multi-Query Transformer" (2023)
- KV cache analysis papers

## What's Next?

Module 14 covers **Frontier Directions** — speculative but important research areas.
