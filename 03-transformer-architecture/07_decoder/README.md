# Step 7: The Decoder Block

---

## Why This Matters

The decoder is where **generation** happens. While the encoder understands input, the decoder produces output — one token at a time, using what it's generated so far and (optionally) what the encoder understood.

```
THREE TRANSFORMER ARCHITECTURES
═══════════════════════════════════════════════════════════════════════

ENCODER-ONLY (BERT, RoBERTa)          Used for: Understanding
    [Encoder] → [Output]              Tasks: Classification, NER, QA

ENCODER-DECODER (T5, BART)            Used for: Transformation
    [Encoder] → [Decoder] → [Output]  Tasks: Translation, Summarization

DECODER-ONLY (GPT, LLaMA, Claude)     Used for: Generation
    [Decoder] → [Output]              Tasks: Chat, Code, Creative writing

The decoder (with or without encoder) is what makes GENERATION possible.
```

---

## The Core Intuition

> **Intuition**: The decoder is like a **writer with a reference book**. It writes one word at a time (autoregressive), can look back at what it's already written (masked self-attention), and can consult the reference book (cross-attention to encoder).

```
Writing a translation:

Step 1: "___"           → Look at source → Write "The"
Step 2: "The ___"       → Look at source + "The" → Write "cat"  
Step 3: "The cat ___"   → Look at source + "The cat" → Write "sat"
...

Each step:
1. What have I written? (masked self-attention)
2. What does the source say? (cross-attention)
3. What should I write next? (output)
```

---

## Architecture: The Decoder Block

The decoder has **THREE sub-layers** (vs encoder's two):

```
                        DECODER BLOCK (Pre-LN Style)
═══════════════════════════════════════════════════════════════════════

                        Decoder Input
                     (batch, tgt_len, d_model)
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         │                    ▼                    │
         │            ┌─────────────┐              │
         │            │  LayerNorm  │              │
         │            └──────┬──────┘              │
         │                   │                     │
         │                   ▼                     │
         │   ┌───────────────────────────────┐     │
         │   │   MASKED Self-Attention       │     │
         │   │   (Causal - No Future Peek)   │     │
         │   │        Q = K = V              │     │
         │   └───────────────┬───────────────┘     │
         │                   │                     │
         │               [Dropout]                 │
         │                   │                     │
         └──────────────────(+)────────────────────┘
                             │    ← Residual
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         │                   ▼                   │
         │           ┌─────────────┐             │
         │           │  LayerNorm  │             │
         │           └──────┬──────┘             │
         │                  │                    │
         │                  ▼                    │
         │   ┌───────────────────────────────┐   │
         │   │      CROSS-Attention          │   │   ┌─────────────┐
         │   │   Q = decoder state           │◄──┼───│   Encoder   │
         │   │   K = V = encoder output      │   │   │   Output    │
         │   └───────────────┬───────────────┘   │   └─────────────┘
         │                   │                   │
         │               [Dropout]               │
         │                   │                   │
         └──────────────────(+)──────────────────┘
                             │    ← Residual
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
         │    │     Feed-Forward Net     │       │
         │    └────────────┬─────────────┘       │
         │                 │                     │
         │             [Dropout]                 │
         │                 │                     │
         └────────────────(+)────────────────────┘
                           │    ← Residual
                           │
                           ▼
                      Decoder Output
                  (batch, tgt_len, d_model)
```

---

## The Three Types of Attention

### Type 1: Masked Self-Attention (Causal)

**Problem**: During training, we feed the entire target sequence. But during generation, future tokens don't exist yet!

**Solution**: Mask future positions so the model can't cheat.

```
CAUSAL MASK: Each position sees only past + itself
═══════════════════════════════════════════════════════════════

Target: "<s>  I   am  a  student"
         0   1   2   3     4

Attention Matrix (1 = can see, 0 = blocked):

        <s>   I   am    a  student
  <s>  [ 1    0    0    0    0  ]   ← Position 0 sees only itself
   I   [ 1    1    0    0    0  ]   ← Position 1 sees [0,1]
  am   [ 1    1    1    0    0  ]   ← Position 2 sees [0,1,2]
   a   [ 1    1    1    1    0  ]   ← Position 3 sees [0,1,2,3]
student[ 1    1    1    1    1  ]   ← Position 4 sees all past

Lower triangular! Upper triangle → -∞ → softmax → 0 attention
```

### Type 2: Cross-Attention (Encoder-Decoder)

**Different sources for Q vs K,V:**

```
                    Encoder Output (Source)
                    "Le chat est assis"
                     ↓    ↓    ↓    ↓
                     K₀   K₁   K₂   K₃  (Keys)
                     V₀   V₁   V₂   V₃  (Values)
                     ↑    ↑    ↑    ↑
Cross-Attention:     └────┴────┴────┘
                           ↑
                    ┌──────┴──────┐
                    │   Decoder   │
                    │  "The cat"  │
                    │   Q₀   Q₁   │  (Queries)
                    └─────────────┘

Decoder asks: "Given what I've generated, what in the source is relevant?"
```

### Type 3: Comparison Table

| Attention Type | Q Source | K,V Source | Mask | Purpose |
|----------------|----------|------------|------|---------|
| **Self (Encoder)** | Input | Input | None (bidirectional) | Full context understanding |
| **Masked Self (Decoder)** | Decoder | Decoder | Causal | No future peeking |
| **Cross (Decoder)** | Decoder | Encoder | Optional padding | Read source sequence |

---

## Training vs Inference: The Key Difference

### Training: Teacher Forcing

During training, we have the complete target — use masking to simulate autoregression:

```
Source: "Je suis étudiant"
Target: "<s> I am a student </s>"

Training input:   ["<s>", "I",  "am", "a", "student"]     (shifted right)
Training target:  ["I",  "am", "a", "student", "</s>"]   (what to predict)

Forward pass processes ALL positions in PARALLEL
But causal mask ensures each only sees its past
Loss computed on all positions simultaneously
```

### Inference: Autoregressive Generation

During inference, we generate one token at a time:

```
Step 0: Input: ["<s>"]
        Predict: "I"
        
Step 1: Input: ["<s>", "I"]
        Predict: "am"
        
Step 2: Input: ["<s>", "I", "am"]
        Predict: "a"
        
Step 3: Input: ["<s>", "I", "am", "a"]
        Predict: "student"
        
Step 4: Input: ["<s>", "I", "am", "a", "student"]
        Predict: "</s>"  ← STOP
```

---

## ASCII Visualization: Full Encoder-Decoder Flow

```
                    ENCODER-DECODER TRANSFORMER
═══════════════════════════════════════════════════════════════════════

    SOURCE: "Je suis étudiant"          TARGET: "<s> I am a student"
              │                                      │
              ▼                                      ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │   Token Embedding   │              │   Token Embedding   │
    │   + Position Enc    │              │   + Position Enc    │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    │
    ┌──────────────────────┐                       │
    │                      │                       │
    │   ENCODER STACK      │                       │
    │   (N=6 layers)       │                       │
    │                      │                       │
    │  [Self-Attention]    │                       │
    │       + FFN          │                       │
    │                      │                       │
    └──────────┬───────────┘                       │
               │                                    │
               │ encoder_output                     ▼
               │                        ┌──────────────────────┐
               │                        │                      │
               └───────────────────────►│   DECODER STACK      │
                   (cross-attention)    │   (N=6 layers)       │
                                        │                      │
                                        │  [Masked Self-Attn]  │
                                        │  [Cross-Attention]   │◄─┘
                                        │       + FFN          │
                                        │                      │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │   Linear + Softmax  │
                                        │   (vocab_size)      │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                            "I am a student </s>"
```

---

## Decoder-Only: The GPT Architecture

Modern LLMs (GPT, LLaMA, Claude) use **decoder-only** — no encoder, no cross-attention:

```
DECODER-ONLY BLOCK (GPT-Style)
═══════════════════════════════════════════════════════════════

         Input x
            │
            ├──────────────────┐
            │                  │
            ▼                  │
      [LayerNorm]              │
            │                  │
            ▼                  │
   ┌─────────────────┐         │
   │ Masked Self-Attn│         │
   │ (causal only)   │         │
   └────────┬────────┘         │
            │                  │
        [Dropout]              │
            │                  │
            └──────(+)─────────┘
                    │ ← Residual
            ├───────┴──────────┐
            │                  │
            ▼                  │
      [LayerNorm]              │
            │                  │
            ▼                  │
   ┌─────────────────┐         │
   │  Feed-Forward   │         │
   └────────┬────────┘         │
            │                  │
        [Dropout]              │
            │                  │
            └──────(+)─────────┘
                    │ ← Residual
                    ▼
               Output

NO cross-attention! Input IS the context.
Everything is treated as "continuing a sequence."
```

**Why decoder-only dominates:**
- Simpler architecture (one type of block)
- Scales better (fewer components)
- Works for everything (prompt = "encoder", generation = "decoder")

---

## Comparison: Decoder Variants

| Aspect | Encoder-Decoder | Decoder-Only (GPT) |
|--------|-----------------|---------------------|
| Sub-layers | 3 (masked self, cross, FFN) | 2 (masked self, FFN) |
| Input handling | Separate encoder for source | Prompt is part of sequence |
| Cross-attention | Yes (to encoder) | No |
| Use cases | Translation, summarization | General generation, chat |
| Examples | T5, BART, mBART | GPT, LLaMA, Claude |
| Parameters | More (cross-attn params) | Fewer per layer |

---

## Common Pitfalls & Debugging

### Pitfall 1: Forgetting the Causal Mask

```python
# ❌ WRONG: No causal mask = decoder can see future!
output = decoder(tgt, encoder_output)  # Cheating during training!

# ✅ CORRECT: Always use causal mask
causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1) * -1e9
output = decoder(tgt, encoder_output, self_attn_mask=causal_mask)
```

### Pitfall 2: Mixing Up Q, K, V in Cross-Attention

```python
# ❌ WRONG: All from decoder (this is self-attention!)
cross_attn = attention(query=dec, key=dec, value=dec)

# ✅ CORRECT: Q from decoder, K/V from encoder
cross_attn = attention(query=dec, key=enc_out, value=enc_out)
```

### Pitfall 3: Wrong Mask Shape for Cross-Attention

```python
# ❌ WRONG: Using causal mask for cross-attention
# Causal mask is (tgt_len, tgt_len), but cross-attn is (tgt_len, src_len)!
cross_output = cross_attn(dec, enc_out, enc_out, mask=causal_mask)

# ✅ CORRECT: Different mask for cross-attention (usually padding mask)
# Shape: (batch, 1, src_len) or None
cross_output = cross_attn(dec, enc_out, enc_out, mask=src_padding_mask)
```

### Pitfall 4: Target Sequence Shifting

```python
# ❌ WRONG: Same sequence for input and target
target = ["I", "am", "a", "student", "</s>"]
decoder_input = target  # Model sees what it should predict!
prediction_target = target

# ✅ CORRECT: Shift right
decoder_input =     ["<s>", "I", "am", "a", "student"]    # Start token
prediction_target = ["I", "am", "a", "student", "</s>"]   # What to predict
```

---

## Implementation: From Scratch

```python
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    """Full Decoder Block with masked self-attention and cross-attention."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Three sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Three layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Masked self-attention
        x_norm = self.norm1(x)
        self_attn_out = self.self_attention(x_norm, x_norm, x_norm, self_attn_mask)
        x = x + self.dropout(self_attn_out)
        
        # Cross-attention to encoder
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attention(x_norm, encoder_output, encoder_output, cross_attn_mask)
        x = x + self.dropout(cross_attn_out)
        
        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x


class DecoderOnlyBlock(nn.Module):
    """GPT-style decoder block (no cross-attention)."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Only two sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Masked self-attention only
        x_norm = self.norm1(x)
        attn_out = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x
```

---

## Implementation: Using PyTorch

```python
import torch.nn as nn

# PyTorch's built-in TransformerDecoderLayer
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True,
    norm_first=True  # Pre-LN
)

# Full decoder stack
decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=6,
    norm=nn.LayerNorm(512)
)

# Usage
tgt = torch.randn(batch_size, tgt_len, 512)
memory = encoder_output  # From encoder

# Create causal mask
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

decoded = decoder(tgt, memory, tgt_mask=tgt_mask)
```

---

## Hands-On Experiment

Run the implementation to visualize decoder attention patterns:

```bash
python 07_decoder/decoder.py
```

**What you'll see:**
1. Causal mask structure
2. Self-attention pattern (lower triangular)
3. Cross-attention pattern (decoder attending to encoder)
4. Comparison of full decoder vs decoder-only

---

## Files in This Module

| File | Description |
|------|-------------|
| `decoder.py` | Full Decoder, DecoderOnly implementations |
| `decoder_attention.png` | Attention pattern visualizations |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Three sub-layers**: Decoder has masked self-attention → cross-attention → FFN (vs encoder's two sub-layers)

2. **Causal mask is critical**: Prevents future token visibility — lower triangular attention matrix ensures autoregressive property

3. **Cross-attention bridges encoder-decoder**: Queries from decoder, keys/values from encoder — this is how the decoder "reads" the source

4. **Training vs inference differ**: Training uses teacher forcing (parallel with mask), inference is truly autoregressive (one token at a time)

5. **Decoder-only is simpler**: GPT-style models skip cross-attention entirely — the prompt IS the context, generation IS the output

6. **Shift your targets**: Decoder input is shifted right from prediction target (input starts with `<s>`, target ends with `</s>`)

7. **Cross-attention needs different mask**: Self-attention uses causal mask (tgt_len × tgt_len), cross-attention may use padding mask (tgt_len × src_len)

---

## What's Next?

We now have all the pieces:
- Encoder (understanding)
- Decoder (generation)
- All the building blocks (attention, FFN, norms, residuals)

In **Step 8: Complete Transformer**, we'll put it all together into a working model — adding embeddings, output projection, and seeing the full architecture in action.

