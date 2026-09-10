# Step 7: The Decoder Block

## Overview

The decoder generates the output sequence **one token at a time**, using:
1. What it has generated so far (masked self-attention)
2. The encoder's output (cross-attention)

## Architecture

```
┌─────────────────────────────────────────┐
│            DECODER BLOCK                │
├─────────────────────────────────────────┤
│                                         │
│  Input: (batch, tgt_len, d_model)       │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────┐            │
│  │ MASKED Self-Attention   │◄── Can't   │
│  │   (Q=K=V=decoder_input) │    peek!   │
│  └─────────────────────────┘            │
│           │                             │
│           + ←── Residual                │
│           │                             │
│     [Layer Norm]                        │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────┐            │
│  │   Cross-Attention       │◄── Encoder │
│  │   Q=decoder             │    output  │
│  │   K=V=encoder_output    │            │
│  └─────────────────────────┘            │
│           │                             │
│           + ←── Residual                │
│           │                             │
│     [Layer Norm]                        │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────┐            │
│  │   Feed-Forward          │            │
│  │   Network (FFN)         │            │
│  └─────────────────────────┘            │
│           │                             │
│           + ←── Residual                │
│           │                             │
│     [Layer Norm]                        │
│           │                             │
│           ↓                             │
│  Output: (batch, tgt_len, d_model)      │
│                                         │
└─────────────────────────────────────────┘
```

## The Three Attention Types

### 1. Masked Self-Attention (Causal)

**Problem**: During training, we have the full target sequence. But during inference, we generate one token at a time. The model shouldn't "cheat" by looking at future tokens.

**Solution**: Causal mask — each position can only attend to itself and previous positions.

```
Position 0: can see [0]
Position 1: can see [0, 1]
Position 2: can see [0, 1, 2]
Position 3: can see [0, 1, 2, 3]

Mask matrix:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

### 2. Cross-Attention (Encoder-Decoder)

**Query** comes from decoder, **Key/Value** come from encoder:

```python
cross_attn_output = MultiHeadAttention(
    query=decoder_hidden,      # "What am I looking for?"
    key=encoder_output,        # "What does the input contain?"
    value=encoder_output       # "What info from input do I need?"
)
```

This is how the decoder "looks at" the source sequence.

### 3. Comparison

| Type | Q Source | K,V Source | Use Case |
|------|----------|------------|----------|
| Self-Attention | input | input | Encoder, understanding context |
| Masked Self-Attention | input | input (masked) | Decoder, autoregressive |
| Cross-Attention | decoder | encoder | Decoder, reading source |

## Causal Mask Implementation

```python
def create_causal_mask(seq_len):
    """Create lower triangular mask for autoregressive attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # Upper triangle is -inf, lower is 0

# In attention:
scores = scores + causal_mask  # Add -inf to future positions
# After softmax, -inf → 0 (no attention to future)
```

## Training vs Inference

### Training (Teacher Forcing)
```
Input:  "Je suis étudiant"
Target: "<start> I am a student <end>"

# Full target available — use masking to simulate autoregressive
decoder_input = "<start> I am a student"  # Shifted right
decoder_output predicts: "I am a student <end>"
```

### Inference (Autoregressive)
```
Step 0: Input: "<start>"           → Predict: "I"
Step 1: Input: "<start> I"         → Predict: "am"
Step 2: Input: "<start> I am"      → Predict: "a"
Step 3: Input: "<start> I am a"    → Predict: "student"
...
```

## Implementation

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn_out = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
```

## Files

- `decoder.py` - Complete decoder block with mask visualization

## Key Takeaways

1. Decoder has **3 sub-layers**: masked self-attn, cross-attn, FFN
2. **Causal mask** prevents peeking at future tokens
3. **Cross-attention**: decoder queries, encoder provides keys/values
4. Training uses **teacher forcing** (full target, masked)
5. Inference is **autoregressive** (generate one token at a time)

## What's Next?

Step 8: **Complete Transformer** — putting encoder and decoder together with embeddings and output projection.
