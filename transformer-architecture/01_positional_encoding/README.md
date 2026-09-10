# Step 1: Positional Encoding

## The Problem

Transformers process all tokens **in parallel**, unlike RNNs which process sequentially. This means they have no inherent sense of word order:

```
"The cat sat on the mat"  →  Transformer sees: {cat, mat, on, sat, the, the}
```

Without position information, the model can't distinguish between "The cat sat on the mat" and "The mat sat on the cat".

## The Solution

Inject position information by **adding** a position-dependent vector to each token embedding.

### Sinusoidal Positional Encoding (Original Paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index
- `d_model` = embedding dimension

## Why Sin/Cos Waves?

1. **Unique patterns**: Each position gets a distinct encoding
2. **Bounded values**: Output always in [-1, 1]
3. **Relative positions**: `PE(pos+k)` is a linear function of `PE(pos)` (rotation property)
4. **Extrapolation**: Can handle sequences longer than training data

### Frequency Intuition

Think of it like a clock with multiple hands:
- **Low dimensions** = fast-moving hands (high frequency) → fine position detail
- **High dimensions** = slow-moving hands (low frequency) → coarse position info

## Key Implementation Details

```python
# Pre-compute all position encodings
pe = torch.zeros(max_seq_len, d_model)

# Position indices
position = torch.arange(0, max_seq_len).unsqueeze(1)

# Frequency divisor (numerically stable version)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))

# Apply sin to even indices, cos to odd indices
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

# Register as buffer (not a learnable parameter)
self.register_buffer('pe', pe)
```

## Usage

```python
# Create layer
pos_encoder = PositionalEncoding(d_model=512, max_seq_len=5000)

# Apply to embeddings
# Input:  (batch_size, seq_len, d_model)
# Output: (batch_size, seq_len, d_model)  ← same shape, position info added
output = pos_encoder(embeddings)
```

## Files

- `positional_encoding.py` - Implementation with detailed comments
- `positional_encoding_visualization.png` - Generated visualization

## Run It

```bash
python 01_positional_encoding/positional_encoding.py
```

## Key Takeaways

1. Position encoding is **added** to embeddings, not concatenated
2. It's **pre-computed once** and reused (efficient)
3. Stored as a **buffer**, not a parameter (not learned)
4. Nearby positions have **similar** encodings (smooth)
5. The encoding is **deterministic** — same position always gets same encoding

## What's Next?

In Step 2, we'll implement **Scaled Dot-Product Attention** — the mechanism that allows tokens to "look at" each other.
