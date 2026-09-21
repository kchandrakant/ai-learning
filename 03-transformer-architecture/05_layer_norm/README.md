# Step 5: Layer Normalization & Residual Connections

## The Problem

Deep networks are hard to train:
1. **Vanishing/exploding gradients**: Signal degrades through many layers
2. **Internal covariate shift**: Distribution of inputs to each layer keeps changing
3. **Optimization landscape**: Hard to find good minima

## Solution 1: Residual Connections (Skip Connections)

Instead of learning `H(x)`, learn the **residual** `F(x) = H(x) - x`:

```
output = x + Sublayer(x)
```

### Why This Works

1. **Gradient highway**: Gradients flow directly through the skip connection
2. **Easy to learn identity**: If optimal transform is identity, just learn F(x)=0
3. **Ensemble effect**: Each layer adds a small refinement

```
Layer 1: x → x + f₁(x)
Layer 2: x + f₁(x) → x + f₁(x) + f₂(x + f₁(x))
...
```

## Solution 2: Layer Normalization

Normalize activations **across features** (not across batch like BatchNorm):

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β

where:
  μ = mean across features (last dimension)
  σ² = variance across features
  γ, β = learned scale and shift
```

### Why Layer Norm (not Batch Norm)?

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch size | Yes | No |
| Works with seq2seq | Poorly | Well |
| Inference behavior | Different from training | Same |

## The "Add & Norm" Pattern

Original transformer (Post-LN):
```
output = LayerNorm(x + Sublayer(x))
```

This is applied after both attention and FFN:
```python
# Attention block
x = LayerNorm(x + MultiHeadAttention(x))

# FFN block  
x = LayerNorm(x + FeedForward(x))
```

## Visualizing the Flow

```
Input
  │
  ├──────────────────┐
  ↓                  │
[Multi-Head Attention]│
  ↓                  │
  + ←────────────────┘ (residual)
  ↓
[Layer Norm]
  │
  ├──────────────────┐
  ↓                  │
[Feed-Forward]       │
  ↓                  │
  + ←────────────────┘ (residual)
  ↓
[Layer Norm]
  ↓
Output
```

## Implementation

```python
class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_output):
        # Add residual, then normalize
        return self.norm(x + self.dropout(sublayer_output))
```

## Pre-LN vs Post-LN (Preview of Evolution 1)

**Post-LN (Original):**
```
output = LayerNorm(x + Sublayer(x))
```

**Pre-LN (GPT-2, Modern):**
```
output = x + Sublayer(LayerNorm(x))
```

Pre-LN is more stable for training deep networks — we'll explore this in the Beyond section.

## Files

- `layer_norm_residual.py` - Implementation with gradient flow visualization

## Key Takeaways

1. **Residual connections** let gradients flow directly (no degradation)
2. **Layer normalization** stabilizes training (normalizes across features)
3. Together they enable **very deep** transformers (100+ layers)
4. Pattern: `output = LayerNorm(x + Sublayer(x))`
5. **Dropout** is applied before the residual addition

## What's Next?

Step 6: **Encoder Block** — combining attention, FFN, and Add&Norm into a complete encoder layer.
