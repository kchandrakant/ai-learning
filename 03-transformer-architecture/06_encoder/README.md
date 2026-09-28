# Step 6: The Encoder Block

---

## Why This Matters

The encoder is where **understanding** happens. It's the component that reads and comprehends the input sequence, building rich representations that capture meaning, context, and relationships.

```
ENCODER-ONLY MODELS (Understanding):
    BERT, RoBERTa, ALBERT
    ↓
    Tasks: Classification, NER, Sentiment, QA (extractive)
    
ENCODER-DECODER MODELS (Understanding → Generation):
    T5, BART, Original Transformer
    ↓
    Tasks: Translation, Summarization, QA (generative)
```

The encoder is the "reader" — it processes input and creates representations that downstream components can use. Every encoder-based model shares this architecture.

---

## The Core Intuition

> **Intuition**: The encoder is like a **book club discussion**. Each word (token) starts with its own understanding, then through multiple rounds of discussion (layers), everyone gains a richer understanding by learning from each other.

```
LAYER 1: "What words are near me?"
         Each token notices its neighbors
         
LAYER 3: "What's the grammar structure?"
         Tokens understand subject-verb-object relationships
         
LAYER 6: "What does this MEAN?"
         Tokens have deep contextual understanding
         
"The bank by the river" → LAYER 6 knows "bank" = riverbank, not financial institution
```

---

## Architecture: The Encoder Block

```
                        ENCODER BLOCK (Pre-LN Style)
═══════════════════════════════════════════════════════════════════════

                            Input x
                         (batch, seq, d_model)
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           │                   ▼                   │
           │           ┌─────────────┐             │
           │           │  LayerNorm  │             │
           │           └──────┬──────┘             │
           │                  │                    │
           │                  ▼                    │
           │    ┌──────────────────────────┐       │
           │    │   Multi-Head Attention   │       │
           │    │      (Self-Attention)    │       │
           │    │       Q = K = V = x      │       │
           │    └────────────┬─────────────┘       │
           │                 │                     │
           │             [Dropout]                 │
           │                 │                     │
           └────────────────(+)────────────────────┘
                             │    ← Residual Connection
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           │                 ▼                 │
           │         ┌─────────────┐           │
           │         │  LayerNorm  │           │
           │         └──────┬──────┘           │
           │                │                  │
           │                ▼                  │
           │    ┌──────────────────────┐       │
           │    │   Feed-Forward Net   │       │
           │    │   (per position)     │       │
           │    └────────────┬─────────┘       │
           │                 │                 │
           │             [Dropout]             │
           │                 │                 │
           └────────────────(+)────────────────┘
                             │    ← Residual Connection
                             │
                             ▼
                          Output
                      (batch, seq, d_model)
```

**Key observation**: Input and output have the **same shape**! The encoder refines representations, it doesn't change dimensions.

---

## Self-Attention: The "Self" Part

In the encoder, attention is **self-attention** — the input attends to itself:

```
Regular Attention:
    Query comes from: Source A
    Key/Value from:   Source B
    → A looks at B

SELF-Attention (Encoder):
    Query comes from: Input X
    Key/Value from:   Input X (same!)
    → X looks at X
```

```python
# Self-attention: Q, K, V are ALL from the same input
attention_output = MultiHeadAttention(
    query=x,   # "What am I looking for?"    ← from x
    key=x,     # "What do I contain?"        ← from x
    value=x    # "What info do I provide?"   ← from x
)
```

**Why self-attention?** Every token needs to understand its relationship to every OTHER token in the same sequence.

```
"The cat sat on the mat"
      ↓
Self-attention lets "sat" see:
    - "cat" (who sat?)
    - "mat" (where?)
    - "on" (relationship)
    
All at once, in parallel!
```

---

## Stacking: Why Multiple Layers?

```
                    THE ENCODER STACK
═══════════════════════════════════════════════════════════════

    Input Embeddings + Positional Encoding
                    │
                    ▼
            ┌───────────────┐
            │ Encoder Block │  Layer 1: Local patterns
            │       1       │  "What's near me?"
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ Encoder Block │  Layer 2-3: Syntax
            │       2       │  "What's the grammar?"
            └───────┬───────┘
                    │
                    ⋮
                    │
                    ▼
            ┌───────────────┐
            │ Encoder Block │  Layer N-1, N: Semantics
            │       N       │  "What does it MEAN?"
            └───────┬───────┘
                    │
                    ▼
              [Final Norm]  ← Required for Pre-LN
                    │
                    ▼
           Encoder Output
```

**Research insight**: Different layers capture different types of information:
- **Early layers** (1-2): Surface features, local patterns
- **Middle layers** (3-4): Syntactic structure, grammar
- **Later layers** (5-6+): Semantic meaning, long-range dependencies

---

## ASCII Visualization: Data Flow Through 6 Layers

```
Input: "The cat sat"   (batch=1, seq=3, d_model=512)

Layer 0: [The]₀  [cat]₀  [sat]₀   ← Raw embeddings + position
            │       │       │
            ▼       ▼       ▼
Layer 1: [The]₁  [cat]₁  [sat]₁   ← Basic relationships learned
            │       │       │
            ▼       ▼       ▼
Layer 2: [The]₂  [cat]₂  [sat]₂   ← More context integrated
            │       │       │
            ▼       ▼       ▼
Layer 3: [The]₃  [cat]₃  [sat]₃   ← Syntactic understanding
            │       │       │
            ▼       ▼       ▼
Layer 4: [The]₄  [cat]₄  [sat]₄   ← Deeper patterns
            │       │       │
            ▼       ▼       ▼
Layer 5: [The]₅  [cat]₅  [sat]₅   ← Approaching semantic
            │       │       │
            ▼       ▼       ▼
Layer 6: [The]₆  [cat]₆  [sat]₆   ← Rich contextual representations
                                   
Each [token]ᵢ contains information from ALL tokens up to layer i!
```

---

## Encoder vs No Masking

A critical difference between encoder and decoder:

```
ENCODER: No Causal Mask (Bidirectional)
─────────────────────────────────────────
Every token can see EVERY other token:

        The   cat   sat    on   the   mat
The   [  ✓     ✓     ✓     ✓     ✓     ✓  ]
cat   [  ✓     ✓     ✓     ✓     ✓     ✓  ]
sat   [  ✓     ✓     ✓     ✓     ✓     ✓  ]   ← Full visibility
on    [  ✓     ✓     ✓     ✓     ✓     ✓  ]
the   [  ✓     ✓     ✓     ✓     ✓     ✓  ]
mat   [  ✓     ✓     ✓     ✓     ✓     ✓  ]

This is why BERT is "Bidirectional" - it sees past AND future!


DECODER: Causal Mask (Unidirectional)
─────────────────────────────────────────
Each token can only see past tokens:

        The   cat   sat    on   the   mat
The   [  ✓     ✗     ✗     ✗     ✗     ✗  ]
cat   [  ✓     ✓     ✗     ✗     ✗     ✗  ]
sat   [  ✓     ✓     ✓     ✗     ✗     ✗  ]   ← Limited visibility
on    [  ✓     ✓     ✓     ✓     ✗     ✗  ]
the   [  ✓     ✓     ✓     ✓     ✓     ✗  ]
mat   [  ✓     ✓     ✓     ✓     ✓     ✓  ]
```

---

## Padding Mask: Handling Variable Lengths

When batching sequences, shorter ones are padded:

```
Batch of sentences:
    "The cat sat"       → [The, cat, sat, PAD, PAD]
    "Hello world"       → [Hello, world, PAD, PAD, PAD]
    "Transformers rock" → [Transformers, rock, PAD, PAD, PAD]

Padding mask (1 = real, 0 = padding):
    [1, 1, 1, 0, 0]
    [1, 1, 0, 0, 0]
    [1, 1, 0, 0, 0]
```

**Why mask padding?** Without masking, tokens would attend to meaningless PAD tokens!

```python
# Apply padding mask in attention
if mask is not None:
    # Set padding positions to -inf → softmax gives 0 weight
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## Comparison: Encoder in Different Models

| Model | Encoder Layers | d_model | Heads | Use Case |
|-------|----------------|---------|-------|----------|
| **BERT Base** | 12 | 768 | 12 | Understanding |
| **BERT Large** | 24 | 1024 | 16 | Understanding |
| **Original Transformer** | 6 | 512 | 8 | Translation (enc-dec) |
| **T5 Base** | 12 | 768 | 12 | Enc-Dec generation |
| **RoBERTa** | 12/24 | 768/1024 | 12/16 | Better BERT |

**Note**: GPT models (decoder-only) don't have a separate encoder!

---

## Common Pitfalls & Debugging

### Pitfall 1: Forgetting Final LayerNorm (Pre-LN)

```python
# ❌ WRONG: Pre-LN architecture without final norm
class Encoder(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # Last layer's output isn't normalized!

# ✅ CORRECT: Add final normalization
class Encoder(nn.Module):
    def __init__(self, ...):
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)  # Normalize at the end
```

### Pitfall 2: Using Causal Mask in Encoder

```python
# ❌ WRONG: Encoder shouldn't use causal mask!
# This limits information flow unnecessarily
mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
encoder_output = encoder(x, mask=mask)

# ✅ CORRECT: No mask (or padding mask only)
encoder_output = encoder(x, mask=None)  # Or padding_mask
```

### Pitfall 3: Expecting Different Output Shape

```python
# ❌ WRONG: Expecting encoder to change dimensions
x = torch.randn(batch, seq_len, 512)
out = encoder(x)
# Don't expect out.shape to be different!

# ✅ CORRECT: Output shape equals input shape
assert out.shape == x.shape  # (batch, seq_len, 512)
```

### Pitfall 4: Confusing Self-Attention Arguments

```python
# ❌ WRONG: Different sources for Q, K, V in encoder self-attention
attn_out = attention(query=x, key=y, value=z)  # This is cross-attention!

# ✅ CORRECT: Same source for self-attention
attn_out = attention(query=x, key=x, value=x)  # Self-attention
```

---

## Implementation: From Scratch

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """Single Transformer Encoder Block (Pre-LN)."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer norms (Pre-LN: before each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention with residual
        x_norm = self.norm1(x)
        attn_out = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x


class Encoder(nn.Module):
    """Full Encoder: Stack of N EncoderBlocks."""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)  # Required for Pre-LN
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
```

---

## Implementation: Using PyTorch

```python
import torch.nn as nn

# PyTorch's built-in TransformerEncoderLayer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    batch_first=True,
    norm_first=True  # Pre-LN (set False for Post-LN)
)

# Full encoder stack
encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6,
    norm=nn.LayerNorm(512)  # Final norm
)

# Usage
x = torch.randn(batch_size, seq_len, 512)
encoded = encoder(x, src_key_padding_mask=padding_mask)
```

---

## Hands-On Experiment

Run the implementation to see the encoder in action:

```bash
python 06_encoder/encoder.py
```

**What you'll see:**
1. Single block input/output shapes
2. Full encoder stack processing
3. Parameter count breakdown
4. Attention weight shapes

---

## Files in This Module

| File | Description |
|------|-------------|
| `encoder.py` | EncoderBlock and full Encoder implementation |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Encoder = Understanding**: The encoder builds rich, contextual representations of the input — it's the "comprehension" component

2. **Self-Attention means Q=K=V**: Every token attends to every other token from the same input, enabling full context awareness

3. **Bidirectional by default**: Unlike the decoder, encoder has no causal mask — each token can see past AND future tokens

4. **Stack for depth**: Multiple layers (6-24+) progressively build from surface patterns to semantic understanding

5. **Same shape in/out**: Encoder refines representations without changing dimensions — (batch, seq, d_model) throughout

6. **Residuals + Norms enable depth**: Without these, deep encoders would fail to train due to gradient issues

7. **Pre-LN needs final norm**: When using Pre-LN architecture (modern default), add LayerNorm after the last block

---

## What's Next?

The encoder understands input, but can't generate output. In **Step 7: Decoder**, we'll add:
- **Masked self-attention**: Prevents looking at future tokens during generation
- **Cross-attention**: Allows decoder to "look at" encoder output
- **Autoregressive generation**: Producing one token at a time

