# Step 6: The Encoder Block

## Overview

The encoder processes the input sequence and builds rich, contextual representations. Each encoder block combines everything we've built so far.

## Architecture

```
┌─────────────────────────────────────┐
│           ENCODER BLOCK             │
├─────────────────────────────────────┤
│                                     │
│  Input: (batch, seq_len, d_model)   │
│           │                         │
│           ↓                         │
│  ┌─────────────────────┐            │
│  │ Multi-Head Attention │◄── Self   │
│  │   (Q=K=V=input)     │    Attn    │
│  └─────────────────────┘            │
│           │                         │
│           + ←── Residual            │
│           │                         │
│     [Layer Norm]                    │
│           │                         │
│           ↓                         │
│  ┌─────────────────────┐            │
│  │   Feed-Forward      │            │
│  │   Network (FFN)     │            │
│  └─────────────────────┘            │
│           │                         │
│           + ←── Residual            │
│           │                         │
│     [Layer Norm]                    │
│           │                         │
│           ↓                         │
│  Output: (batch, seq_len, d_model)  │
│                                     │
└─────────────────────────────────────┘
```

## Self-Attention

In the encoder, **Q, K, and V all come from the same input**:

```python
# Self-attention: input attends to itself
attention_output = MultiHeadAttention(
    query=x,   # What am I looking for?
    key=x,     # What do I contain?
    value=x    # What info do I provide?
)
```

This allows each token to gather information from **all other tokens** in the sequence.

## Stacking Encoder Blocks

The original transformer uses **N=6 identical encoder layers**:

```python
class Encoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, ...):
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, ...) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

Each layer refines the representations:
- Layer 1: Basic syntactic features
- Layer 3: More complex patterns
- Layer 6: High-level semantic understanding

## Padding Mask

When batching sequences of different lengths, we pad shorter ones:

```
Sequence 1: [The, cat, sat, PAD, PAD]
Sequence 2: [Hello, world, PAD, PAD, PAD]
```

The **padding mask** prevents attention to PAD tokens:

```python
# mask: 1 for real tokens, 0 for padding
mask = [[1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0]]

# Apply in attention: set padded positions to -inf before softmax
scores = scores.masked_fill(mask == 0, float('-inf'))
```

## Implementation

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

## What the Encoder Produces

The encoder output is a sequence of **context-aware representations**:

```
Input:  "The cat sat on the mat"
        [tok1, tok2, tok3, tok4, tok5, tok6]
              ↓ Encoder (6 layers)
Output: [ctx1, ctx2, ctx3, ctx4, ctx5, ctx6]
```

Each `ctx_i` now contains information about its **relationship to all other tokens**.

## Files

- `encoder.py` - Complete encoder block and stack implementation

## Key Takeaways

1. Encoder block = **Self-Attention + FFN** with residuals and norms
2. **Self-attention**: Q=K=V (input attends to itself)
3. **Stack N layers** for increasingly abstract representations
4. **Padding mask** prevents attention to PAD tokens
5. Output: same shape as input, but **context-aware**

## What's Next?

Step 7: **Decoder Block** — adds masked self-attention and cross-attention for sequence generation.
