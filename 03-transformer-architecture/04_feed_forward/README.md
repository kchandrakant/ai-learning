# Step 4: Position-wise Feed-Forward Network

## The Problem

Attention only computes **weighted averages** — it's a linear operation on the values. The model needs non-linear transformations to learn complex functions.

## The Solution

Apply a small neural network **independently to each position**:

```
FFN(x) = ReLU(x × W_1 + b_1) × W_2 + b_2
```

Or more generally:
```
FFN(x) = Activation(x × W_1 + b_1) × W_2 + b_2
```

## Architecture

```
Input: (batch, seq_len, d_model=512)
    ↓
Linear: d_model → d_ff (512 → 2048)    [Expand]
    ↓
ReLU (or GELU, SwiGLU)
    ↓
Linear: d_ff → d_model (2048 → 512)    [Contract]
    ↓
Output: (batch, seq_len, d_model=512)
```

## Why This Design?

### 1. Position-wise
The same FFN is applied to each position independently. This is like having a separate network for each token, but with **shared weights**.

### 2. Expansion Factor (typically 4×)
```
d_ff = 4 × d_model
```
The "bottleneck" design:
- Expand to higher dimension → more expressive
- Contract back → manageable size

### 3. Non-linearity
- **ReLU**: Original paper. Simple, fast.
- **GELU**: Used in BERT, GPT-2. Smoother than ReLU.
- **SwiGLU**: Used in LLaMA, PaLM. Even better (Evolution 4).

## Role in the Transformer

```
Attention: "What should I look at?"
    → Gathers information from other positions

FFN: "How should I process what I gathered?"
    → Transforms the gathered information
```

Think of it as:
- Attention = **communication** between positions
- FFN = **computation** at each position

## Implementation

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)      # → (batch, seq_len, d_ff)
        x = F.relu(x)            # Non-linearity
        x = self.dropout(x)
        x = self.linear2(x)      # → (batch, seq_len, d_model)
        return x
```

## Parameter Count

For d_model=512, d_ff=2048:
- W_1: 512 × 2048 = 1,048,576 parameters
- W_2: 2048 × 512 = 1,048,576 parameters
- Total: ~2M parameters per FFN layer

This is often the **majority of parameters** in a transformer!

## Files

- `feed_forward.py` - Implementation with activation comparisons

## Key Takeaways

1. FFN adds **non-linear transformations** (attention is linear)
2. Applied **independently per position** (same weights, shared)
3. **Expand then contract**: d_model → 4×d_model → d_model
4. Contains **most of the parameters** in a transformer layer
5. Modern variants use **GELU or SwiGLU** instead of ReLU

## What's Next?

Step 5: **Layer Normalization & Residual Connections** — the glue that makes deep transformers trainable.
