# Step 8: The Complete Transformer

---

## Why This Matters

This is the culmination — the moment where all the pieces come together. The transformer architecture from "Attention Is All You Need" (2017) revolutionized machine learning and gave birth to:

```
THE TRANSFORMER FAMILY TREE
═══════════════════════════════════════════════════════════════════════

Original Transformer (2017)
         │
         ├──────────────────────────────────────────────────┐
         │                                                  │
         ▼                                                  ▼
    ENCODER-BASED                                    DECODER-BASED
         │                                                  │
         ├── BERT (2018)                                   ├── GPT (2018)
         ├── RoBERTa (2019)                                ├── GPT-2 (2019)
         ├── ALBERT (2019)                                 ├── GPT-3 (2020)
         ├── DeBERTa (2020)                                ├── LLaMA (2023)
         └── ...                                           ├── Claude (2023)
                                                           ├── GPT-4 (2023)
                                                           ├── Mistral (2023)
                                                           └── ...
         │
         ▼
    ENCODER-DECODER
         │
         ├── T5 (2019)
         ├── BART (2019)
         ├── mT5 (2020)
         └── ...

Every modern language model traces back to THIS architecture.
```

---

## The Core Intuition

> **Intuition**: The transformer is a **universal sequence processor**. It takes a sequence of symbols, builds rich contextual understanding of each symbol by letting them all communicate (attention), and either produces a transformed representation (encoder) or generates new symbols one at a time (decoder).

```
The key insight: ATTENTION REPLACES RECURRENCE

Before (RNN/LSTM):
    Token₁ → Token₂ → Token₃ → Token₄
         └─────┴─────┴─────┘
         Sequential, slow, forgets

After (Transformer):
    Token₁ ←→ Token₂ ←→ Token₃ ←→ Token₄
         ↑________|________|________|
         |________|________|
         |________|
    Parallel, fast, full context
```

---

## The Full Architecture

```
                    THE COMPLETE TRANSFORMER
═══════════════════════════════════════════════════════════════════════

        SOURCE SEQUENCE                      TARGET SEQUENCE
      "Je suis étudiant"                   "<s> I am a student"
              │                                     │
              ▼                                     ▼
    ┌─────────────────────┐            ┌─────────────────────┐
    │  Token Embedding    │            │  Token Embedding    │
    │  (vocab → d_model)  │            │  (vocab → d_model)  │
    └──────────┬──────────┘            └──────────┬──────────┘
               │                                   │
               │ × √d_model (scaling)              │ × √d_model
               │                                   │
               ▼                                   ▼
    ┌─────────────────────┐            ┌─────────────────────┐
    │ + Positional Encoding│            │ + Positional Encoding│
    │   (sinusoidal)      │            │   (sinusoidal)      │
    └──────────┬──────────┘            └──────────┬──────────┘
               │                                   │
               ▼                                   ▼
    ╔═══════════════════════╗          ╔═══════════════════════╗
    ║                       ║          ║                       ║
    ║   E N C O D E R       ║          ║   D E C O D E R       ║
    ║   S T A C K           ║          ║   S T A C K           ║
    ║                       ║          ║                       ║
    ║  ┌─────────────────┐  ║          ║  ┌─────────────────┐  ║
    ║  │ Self-Attention  │  ║          ║  │ Masked Self-Attn│  ║
    ║  │   + Add & Norm  │  ║          ║  │   + Add & Norm  │  ║
    ║  └────────┬────────┘  ║          ║  └────────┬────────┘  ║
    ║           │           ║          ║           │           ║
    ║  ┌────────▼────────┐  ║   ┌──────║──►┌───────▼───────┐   ║
    ║  │  Feed-Forward   │  ║   │      ║  │Cross-Attention│   ║
    ║  │   + Add & Norm  │  ║   │      ║  │  + Add & Norm │   ║
    ║  └────────┬────────┘  ║   │      ║  └───────┬───────┘   ║
    ║           │           ║   │      ║          │           ║
    ║       × N layers      ║   │      ║  ┌───────▼───────┐   ║
    ║                       ║   │      ║  │ Feed-Forward  │   ║
    ╚═══════════╤═══════════╝   │      ║  │  + Add & Norm │   ║
                │               │      ║  └───────┬───────┘   ║
                │  encoder      │      ║          │           ║
                │  output       │      ║      × N layers      ║
                └───────────────┘      ╚══════════╤═══════════╝
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │     Linear Projection   │
                                    │   (d_model → vocab_size)│
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │        Softmax          │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                        "I am a student </s>"
```

---

## Component Summary: What We Built

| Step | Component | Purpose | Key Formula/Concept |
|------|-----------|---------|---------------------|
| 1 | Positional Encoding | Position information | sin/cos at multiple frequencies |
| 2 | Scaled Dot-Product Attention | Token interaction | softmax(QK^T/√d_k)V |
| 3 | Multi-Head Attention | Multiple perspectives | Concat(head₁...headₕ)W_O |
| 4 | Feed-Forward Network | Non-linear processing | ReLU(xW₁+b₁)W₂+b₂ |
| 5 | Layer Norm + Residual | Training stability | x + Sublayer(Norm(x)) |
| 6 | Encoder | Understanding | Self-attention + FFN |
| 7 | Decoder | Generation | Masked self-attn + Cross-attn + FFN |
| 8 | Full Transformer | Complete model | Encoder + Decoder + Embeddings |

---

## The Data Flow: Step by Step

```
TRANSLATION: "Hello world" → "Bonjour monde"

STEP 1: TOKENIZATION
─────────────────────────────────────────────────────────────────────
    "Hello world" → [7592, 2088]              (source token IDs)
    "<s> Bonjour" → [1, 4521]                 (target token IDs, shifted)

STEP 2: EMBEDDING + POSITION
─────────────────────────────────────────────────────────────────────
    [7592, 2088] → Embedding → (2, 512)       (lookup vectors)
                 → + PE       → (2, 512)       (add position)
                 → × √512     → (2, 512)       (scale for stability)

STEP 3: ENCODER (×6 layers)
─────────────────────────────────────────────────────────────────────
    Layer 1: Self-attention lets "Hello" see "world" and vice versa
    Layer 2: Deeper patterns, more abstract relationships
    ...
    Layer 6: Rich contextual encoding of source
    
    Output: encoder_output (2, 512) — "Hello world" fully understood

STEP 4: DECODER (×6 layers, for each target position)
─────────────────────────────────────────────────────────────────────
    For target position 1 (predicting "Bonjour"):
        - Masked self-attn: "<s>" looks at itself only
        - Cross-attention: Query from decoder attends to encoder_output
        - FFN: Process gathered information
    
    For target position 2 (predicting "monde"):
        - Masked self-attn: "<s> Bonjour" (can see both)
        - Cross-attention: Query from decoder attends to encoder_output
        - FFN: Process gathered information

STEP 5: OUTPUT PROJECTION
─────────────────────────────────────────────────────────────────────
    Decoder output (2, 512) → Linear (512, vocab_size) → Logits
    Softmax → Probabilities over vocabulary
    
    Position 0 predicts: "Bonjour" (highest probability)
    Position 1 predicts: "monde"
```

---

## Hyperparameters: Original Paper

```
┌────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BASE                                │
├────────────────────────────────────────────────────────────────────┤
│  d_model (embedding dim)         512                               │
│  d_ff (FFN hidden dim)           2048        (4 × d_model)         │
│  num_heads                       8                                 │
│  d_k, d_v (per-head dim)         64          (512 / 8)             │
│  num_encoder_layers              6                                 │
│  num_decoder_layers              6                                 │
│  dropout                         0.1                               │
│  vocab_size                      37,000      (BPE tokens)          │
│  max_sequence_length             512                               │
├────────────────────────────────────────────────────────────────────┤
│  TOTAL PARAMETERS               ~65M                               │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BIG                                 │
├────────────────────────────────────────────────────────────────────┤
│  d_model                         1024                              │
│  d_ff                            4096                              │
│  num_heads                       16                                │
│  num_layers                      6                                 │
├────────────────────────────────────────────────────────────────────┤
│  TOTAL PARAMETERS               ~213M                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Weight Tying: A Clever Trick

The original paper shares weights between embeddings and output projection:

```
THREE PLACES, ONE MATRIX
═══════════════════════════════════════════════════════════════════

Same embedding matrix E (shape: vocab_size × d_model) used for:

1. Source embedding:  E[src_token_id]
2. Target embedding:  E[tgt_token_id]
3. Output projection: hidden @ E.T → logits

Why tie weights?
- Fewer parameters (significant with large vocab)
- Implicit regularization (embedding must be useful for both)
- Semantically similar: "input representation" ≈ "output prediction"
```

---

## Training Details

### Loss: Cross-Entropy with Label Smoothing

```
Standard target:     [0, 0, 1, 0, 0]     (one-hot)
Smoothed target:     [0.02, 0.02, 0.92, 0.02, 0.02]   (ε=0.1)

Why smooth?
- Prevents overconfidence
- Better generalization  
- Slightly worse perplexity, but better BLEU scores
```

### Learning Rate: The Warmup Schedule

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

    ↑
    │      ╱╲
 lr │     ╱  ╲_____
    │    ╱         ╲________
    │   ╱                    ╲_______
    │──────────────────────────────────→
         ↑                          step
      warmup
      (4000)
    
Phase 1: Linear warmup (0 → 4000 steps)
Phase 2: Inverse square root decay
```

---

## Comparison: Transformer Variants

| Variant | Architecture | Training Objective | Use Cases |
|---------|--------------|-------------------|-----------|
| **Original Transformer** | Encoder-Decoder | Seq2Seq (translation) | Translation, summarization |
| **BERT** | Encoder-only | Masked LM + NSP | Classification, NER, QA |
| **GPT** | Decoder-only | Autoregressive LM | Generation, chat |
| **T5** | Encoder-Decoder | Span corruption | All tasks as text-to-text |
| **LLaMA** | Decoder-only | Autoregressive LM | Generation (optimized) |

---

## Modern Improvements (Preview)

The original transformer works well, but modern models add improvements:

```
ORIGINAL (2017)                    MODERN (2023+)
═══════════════════════════════════════════════════════════════════

Sinusoidal Position Encoding  →    Rotary Position Embedding (RoPE)
                                   Better length generalization

Post-Layer Norm               →    Pre-Layer Norm
                                   More stable for deep models

ReLU activation               →    SwiGLU / GeGLU
                                   Better quality

LayerNorm                     →    RMSNorm
                                   Faster, same quality

Full KV storage               →    Grouped Query Attention (GQA)
                                   Less memory for long contexts

No KV cache                   →    KV Cache
                                   Faster inference

These are covered in the 'beyond/' folder!
```

---

## Common Pitfalls & Debugging

### Pitfall 1: Forgetting to Scale Embeddings

```python
# ❌ WRONG: Raw embeddings without scaling
x = self.embedding(tokens)

# ✅ CORRECT: Scale by √d_model
x = self.embedding(tokens) * math.sqrt(self.d_model)

# Why? Embeddings have small values (~0.02 std after init)
# Positional encoding has values in [-1, 1]
# Scaling makes them comparable magnitude
```

### Pitfall 2: Wrong Mask Application Order

```python
# ❌ WRONG: Add mask to scores (if mask is 0/1)
scores = scores + mask  # 0 doesn't block attention!

# ✅ CORRECT: Convert 0→-inf, then add
scores = scores.masked_fill(mask == 0, float('-inf'))
# Or use mask of 0/-inf directly
```

### Pitfall 3: Forgetting Causal Mask for Decoder

```python
# ❌ WRONG: Training decoder without causal mask
logits = transformer(src, tgt)  # Decoder can see future!

# ✅ CORRECT: Always use causal mask
causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1) * -1e9
logits = transformer(src, tgt, tgt_mask=causal_mask)
```

### Pitfall 4: Inference Without Autoregressive Loop

```python
# ❌ WRONG: One-shot inference (only works in training!)
output = transformer(src, tgt)

# ✅ CORRECT: Autoregressive generation
tgt = [BOS_TOKEN]
for _ in range(max_len):
    output = transformer(src, tgt)
    next_token = output[:, -1, :].argmax(-1)
    tgt = torch.cat([tgt, next_token], dim=1)
    if next_token == EOS_TOKEN:
        break
```

---

## Implementation: From Scratch (Encoder-Decoder)

```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    """Complete Encoder-Decoder Transformer."""
    
    def __init__(
        self,
        src_vocab_size, tgt_vocab_size,
        d_model=512, num_heads=8,
        num_encoder_layers=6, num_decoder_layers=6,
        d_ff=2048, max_seq_len=5000, dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder & Decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        src_emb = self.positional_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode target
        tgt_emb = self.positional_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
```

---

## Implementation: Using PyTorch

```python
import torch.nn as nn

# PyTorch's complete Transformer
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)

# Usage
src = torch.randn(batch_size, src_len, 512)  # After embedding
tgt = torch.randn(batch_size, tgt_len, 512)  # After embedding

# Generate masks
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)
src_padding_mask = (src_tokens == PAD_IDX)

output = transformer(
    src, tgt,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_padding_mask
)
```

---

## Hands-On Experiment

Run the complete implementation:

```bash
python 08_transformer/transformer.py
```

**What you'll see:**
1. Encoder-Decoder transformer for translation
2. Decoder-only GPT-style transformer
3. Modern LLaMA-style variant
4. Parameter count comparison
5. Simple generation example

---

## Files in This Module

| File | Description |
|------|-------------|
| `transformer.py` | Complete Transformer and GPT implementations |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **The transformer is modular**: Token embedding → Position encoding → Encoder (for understanding) → Decoder (for generation) → Output projection — each piece is separable and reusable

2. **Three architectures, one foundation**: Encoder-only (BERT), encoder-decoder (T5), and decoder-only (GPT) all derive from the same building blocks

3. **Scaling matters**: The architecture scales well — same design works from 65M to 175B+ parameters

4. **Weight tying reduces parameters**: Sharing embedding and output weights saves memory and provides regularization

5. **Training tricks are crucial**: Learning rate warmup, label smoothing, and careful initialization make training possible

6. **Inference is autoregressive**: Unlike training (which uses teacher forcing), inference generates one token at a time

7. **Modern improvements enhance but don't replace**: RoPE, SwiGLU, RMSNorm, GQA all improve the original design, but the core attention mechanism remains

---

## What You've Accomplished

```
CONGRATULATIONS! 🎉

You've built a transformer from scratch, understanding:

✓ Why position encoding is needed (no inherent order in parallel processing)
✓ How attention works (Q, K, V — query-key matching retrieves values)
✓ Why multi-head (multiple relationship types in parallel)
✓ Why FFN (non-linearity for complex transformations)
✓ Why LayerNorm + Residuals (training stability for deep networks)
✓ How encoder builds understanding (bidirectional self-attention)
✓ How decoder generates output (causal mask + cross-attention)
✓ How it all fits together (embeddings → encoder/decoder → output)

This is the foundation of ALL modern language models!
```

---

## What's Next?

### Immediate Next Steps
- **`demo/`**: Train a tiny transformer on a simple task
- **`beyond/`**: Modern improvements (RoPE, GQA, KV-cache, SwiGLU)

### The Learning Path Continues
- **Course 04**: LLM Internals (tokenization, training, inference optimization)
- **Course 05**: Fine-Tuning (adapting pre-trained models)
- **Course 06**: RAG (retrieval-augmented generation)
- **And beyond**: Agents, multimodal, deployment...

---

## References

1. **"Attention Is All You Need"** (Vaswani et al., 2017) — The original transformer paper
2. **"BERT"** (Devlin et al., 2018) — Encoder-only, bidirectional pre-training
3. **"GPT-2"** (Radford et al., 2019) — Decoder-only, large-scale language modeling
4. **"LLaMA"** (Touvron et al., 2023) — Modern improvements compilation
5. **The Annotated Transformer** — Excellent code walkthrough

