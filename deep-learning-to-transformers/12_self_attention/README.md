# Step 12: Self-Attention

## From Cross-Attention to Self-Attention

Cross-attention: decoder attends to encoder
Self-attention: sequence attends to itself

```
"The cat sat on the mat because it was tired"

What does "it" refer to?
Self-attention: "it" attends to "cat" (high weight)
```

## The Core Idea

Every position attends to every other position in the same sequence.

```
Input:  [x₁, x₂, x₃, x₄]
Output: [y₁, y₂, y₃, y₄]

y₁ = f(x₁, x₂, x₃, x₄)  ← considers all positions
y₂ = f(x₁, x₂, x₃, x₄)
...
```

## Query, Key, Value

Transform each position into three vectors:

```python
Q = X @ W_Q  # Query: "What am I looking for?"
K = X @ W_K  # Key: "What do I contain?"
V = X @ W_V  # Value: "What do I contribute?"
```

## Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.bmm(Q, K.transpose(1, 2))  # (batch, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)          # Scale
    
    # Softmax to get weights
    weights = F.softmax(scores, dim=-1)       # (batch, seq_len, seq_len)
    
    # Weighted sum of values
    output = torch.bmm(weights, V)            # (batch, seq_len, d_v)
    
    return output, weights
```

## Why Scale by √d_k?

```
Without scaling:
- Large d_k → large dot products
- Softmax becomes very peaked (near one-hot)
- Gradients vanish

With scaling:
- Variance stays reasonable
- Softmax has useful gradients
```

## Full Self-Attention Layer

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
    
    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

## Attention Matrix

```
          Position 1  Position 2  Position 3  Position 4
Position 1    0.1        0.5        0.2        0.2
Position 2    0.3        0.1        0.4        0.2
Position 3    0.2        0.3        0.1        0.4
Position 4    0.1        0.2        0.3        0.4

Row i: how much position i attends to each other position
```

## Self-Attention vs RNN

| Aspect | Self-Attention | RNN |
|--------|----------------|-----|
| Parallelization | Full parallel | Sequential |
| Long-range | Direct connection | Through many steps |
| Computation | O(n² · d) | O(n · d²) |
| Memory | O(n²) | O(n) |

Self-attention is faster for moderate sequences, better at long-range dependencies.

## Files

- `self_attention.py` - Self-attention implementation

## Key Takeaways

1. Self-attention: sequence attends to itself
2. Query-Key-Value: three projections
3. Scaling prevents gradient issues
4. Fully parallel unlike RNNs
5. Direct long-range connections

## What's Next?

Step 13: **Multi-Head Attention** — attending in parallel.
