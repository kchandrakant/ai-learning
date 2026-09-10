# Step 13: Multi-Head Attention

## Why Multiple Heads?

One attention head captures one type of relationship.
Multiple heads capture different aspects in parallel.

```
Head 1: syntactic relationships (subject-verb)
Head 2: semantic relationships (synonyms)
Head 3: positional relationships (adjacent words)
...
```

## The Architecture

```
Input X
   │
   ├──→ Head 1 (Q₁, K₁, V₁) → Z₁
   ├──→ Head 2 (Q₂, K₂, V₂) → Z₂
   ├──→ Head 3 (Q₃, K₃, V₃) → Z₃
   ...
   └──→ Head h (Qₕ, Kₕ, Vₕ) → Zₕ
   
   Concat [Z₁, Z₂, ..., Zₕ] → Linear → Output
```

## The Equations

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) @ W_O

where headᵢ = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

## Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, heads, seq_len, d_k)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        return self.W_O(output)
```

## Masking

### Padding Mask
Ignore padding tokens.

```python
# Create mask for padded positions
padding_mask = (tokens != PAD_IDX).unsqueeze(1).unsqueeze(2)
# (batch, 1, 1, seq_len) - broadcasts across heads and queries
```

### Causal Mask (for decoder)
Prevent attending to future positions.

```python
def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # True where attention allowed

# For sequence length 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

## What Different Heads Learn

Research shows heads specialize:
- Some attend to adjacent words
- Some attend to specific syntactic roles
- Some attend to rare words
- Some become "no-op" (attend to everything equally)

## Efficient Multi-Head Attention

```python
# PyTorch built-in
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    batch_first=True
)

output, weights = mha(query, key, value, attn_mask=mask)
```

## Files

- `multihead_attention.py` - Multi-head attention implementation

## Key Takeaways

1. Multiple heads capture different relationships
2. Split d_model into h heads of d_k each
3. Concatenate and project output
4. Masking: padding and causal
5. Same compute cost as single-head (roughly)

## What's Next?

Step 14: **Positional Encoding** — where am I?
