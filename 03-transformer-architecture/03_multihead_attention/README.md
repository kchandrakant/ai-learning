# Step 3: Multi-Head Attention

## The Problem with Single-Head Attention

A single attention head can only focus on one "type" of relationship at a time. But language has many simultaneous relationships:
- Syntactic (subject-verb agreement)
- Semantic (word meanings)
- Positional (nearby words)
- Coreference (pronouns to nouns)

## The Solution: Multiple Heads

Run **h parallel attention operations**, each with its own learned projections, then combine the results.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

## How It Works

### 1. Project to Multiple Subspaces
```python
# Instead of one d_model dimensional attention:
# Split into h heads, each with d_k = d_model / h dimensions

Q_heads = [Q @ W_Q[i] for i in range(num_heads)]  # Each: (seq, d_k)
K_heads = [K @ W_K[i] for i in range(num_heads)]
V_heads = [V @ W_V[i] for i in range(num_heads)]
```

### 2. Parallel Attention
```python
# Each head computes its own attention
head_outputs = [attention(Q_i, K_i, V_i) for Q_i, K_i, V_i in zip(...)]
```

### 3. Concatenate and Project
```python
# Combine all heads back to d_model dimensions
concat = torch.cat(head_outputs, dim=-1)  # (seq, h * d_k) = (seq, d_model)
output = concat @ W_O                      # (seq, d_model)
```

## Dimensions

For `d_model=512` and `num_heads=8`:
- Each head operates on `d_k = 512/8 = 64` dimensions
- Total parameters similar to single large attention
- But captures 8 different relationship types

## Efficient Implementation

Instead of separate matrices per head, use a single large matrix and reshape:

```python
# Single projection, then split into heads
Q = x @ W_Q  # (batch, seq, d_model)
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
# Now: (batch, num_heads, seq, d_k)
```

## What Each Head Learns

Research shows different heads specialize:
- Some focus on **adjacent tokens**
- Some track **syntactic dependencies**
- Some attend to **separator tokens**
- Some handle **rare words**

## Files

- `multihead_attention.py` - Full implementation with head visualization

## Key Takeaways

1. Multiple heads = multiple **relationship types** in parallel
2. Each head has **its own learned projections** (W_Q, W_K, W_V)
3. **Concatenate + project** combines head outputs
4. Total computation ≈ single head (dimensions split, not added)
5. Heads **specialize** during training

## What's Next?

Step 4: **Feed-Forward Network** — adding non-linear transformations at each position.
