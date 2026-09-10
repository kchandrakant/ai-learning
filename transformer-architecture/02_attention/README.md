# Step 2: Scaled Dot-Product Attention

## The Core Innovation

Attention is the mechanism that allows each token to "look at" every other token and decide what's relevant. It answers: **"Given where I am, what should I pay attention to?"**

## The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- `Q` (Query): "What am I looking for?"
- `K` (Key): "What do I contain?"
- `V` (Value): "What information do I provide?"
- `d_k`: Dimension of keys (for scaling)

## Step-by-Step Breakdown

### 1. Compute Attention Scores
```
scores = Q @ K^T
```
Matrix multiplication gives similarity between each query and all keys.
Shape: `(seq_len, seq_len)` — every position attends to every position.

### 2. Scale
```
scores = scores / √d_k
```
**Why scale?** Without scaling, large `d_k` → large dot products → softmax becomes extremely peaked (approaches one-hot). Scaling keeps gradients healthy.

### 3. Apply Softmax
```
attention_weights = softmax(scores, dim=-1)
```
Convert to probabilities. Each row sums to 1.

### 4. Weighted Sum of Values
```
output = attention_weights @ V
```
Each position's output is a weighted combination of all values.

## Visual Intuition

```
Query: "The cat sat on the ___"
        ↓
Keys:   [The] [cat] [sat] [on] [the] [mat]
        0.05  0.40  0.30  0.05 0.05  0.15  ← attention weights
        ↓
Values: Weighted combination → Context-aware representation
```

## Why This Works

- **Parallelizable**: All positions computed simultaneously (unlike RNN)
- **Direct connections**: Any position can attend to any other (no information bottleneck)
- **Learned relevance**: The model learns what relationships matter

## Masking

For certain tasks, we need to prevent attention to certain positions:

1. **Padding mask**: Ignore padding tokens
2. **Causal mask**: Prevent attending to future tokens (for autoregressive generation)

```python
# Causal mask example (lower triangular)
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
# Position 0 can only see position 0
# Position 3 can see positions 0, 1, 2, 3
```

## Files

- `scaled_dot_product_attention.py` - Implementation with visualization

## Key Takeaways

1. Attention computes **pairwise relationships** between all positions
2. **Scaling by √d_k** prevents softmax saturation
3. Output is a **weighted sum** of values based on query-key similarity
4. **Masking** controls what positions can attend to

## What's Next?

Step 3: **Multi-Head Attention** — running multiple attention operations in parallel for richer representations.
