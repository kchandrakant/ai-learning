# Transformer Architecture: A Step-by-Step Learning Journey

This guide walks through implementing a transformer from scratch, starting with the original 2017 "Attention Is All You Need" architecture, then exploring the key advances that power modern LLMs like GPT-4, LLaMA, and Claude.

---

## 🎯 Prerequisites

- Basic Python and PyTorch knowledge
- Linear algebra fundamentals (matrix multiplication, softmax)
- Understanding of neural network basics (layers, activations, backprop)

---

## 📚 Part 1: The Original Transformer (2017)

The architecture from "Attention Is All You Need" by Vaswani et al.

### Step 1: Positional Encoding
**Why it matters:** Transformers process all tokens in parallel (unlike RNNs), so they have no inherent sense of order. Positional encodings inject sequence position information.

**What we'll build:**
- Sinusoidal positional encoding (original paper)
- Understand why sin/cos functions work for this

**Key equations:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### Step 2: Scaled Dot-Product Attention
**Why it matters:** This is THE core innovation. It allows every token to "look at" every other token and decide what's relevant.

**What we'll build:**
- Query, Key, Value projections
- Attention score computation
- Softmax normalization
- The crucial scaling factor (1/√d_k)

**Key equation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**We'll visualize:** Attention patterns to see what tokens attend to what.

---

### Step 3: Multi-Head Attention
**Why it matters:** A single attention head can only focus on one "type" of relationship. Multiple heads allow the model to jointly attend to information from different representation subspaces.

**What we'll build:**
- Parallel attention heads
- Learned projection matrices (W_Q, W_K, W_V, W_O)
- Concatenation and output projection

**Key insight:** With 8 heads and d_model=512, each head operates on d_k=64 dimensions.

---

### Step 4: Position-wise Feed-Forward Network
**Why it matters:** Attention only does weighted averaging — no actual "computation." The FFN adds non-linear transformations applied independently to each position.

**What we'll build:**
- Two linear layers with ReLU activation
- Typical expansion: d_model → 4×d_model → d_model

**Key equation:**
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

---

### Step 5: Layer Normalization & Residual Connections
**Why it matters:** Deep networks are hard to train. These techniques stabilize training and allow gradients to flow.

**What we'll build:**
- Layer normalization (normalize across features, not batch)
- Residual/skip connections
- The "Add & Norm" pattern

**Original pattern (Post-LN):**
```
output = LayerNorm(x + Sublayer(x))
```

---

### Step 6: The Encoder Block
**Why it matters:** The encoder processes the input sequence, building rich contextual representations.

**What we'll build:**
- Self-attention (input attends to itself)
- Feed-forward network
- Proper residual connections and normalization
- Stack N=6 identical layers

---

### Step 7: The Decoder Block
**Why it matters:** The decoder generates the output sequence, one token at a time, while attending to the encoder's output.

**What we'll build:**
- Masked self-attention (can't peek at future tokens!)
- Cross-attention (decoder queries, encoder keys/values)
- The causal mask

---

### Step 8: The Complete Transformer
**What we'll build:**
- Input embeddings + positional encoding
- Encoder stack
- Decoder stack  
- Final linear layer + softmax for predictions

---

## 📚 Part 2: Architectural Advances (2018-2024)

These improvements power modern models like GPT-4, LLaMA, Mistral, and Claude.

> **Note**: See [beyond/README.md](beyond/README.md) for all advances including frontier research (looped transformers, MoE, etc.)

### Evolution 1: Pre-Layer Normalization (GPT-2, 2019)
**The problem:** Post-LN transformers are hard to train at scale without careful learning rate warmup.

**The solution:** Move LayerNorm BEFORE the sublayer instead of after.

```
# Post-LN (original):  output = LayerNorm(x + Sublayer(x))
# Pre-LN (modern):     output = x + Sublayer(LayerNorm(x))
```

**Why it works:** Gradients flow more directly through residual connections.

---

### Evolution 2: Rotary Position Embeddings - RoPE (2021)
**The problem:** Sinusoidal encodings are added once and "forgotten." They don't handle relative positions well or extrapolate to longer sequences.

**The solution:** Encode position by ROTATING the query and key vectors. Relative position naturally emerges from the rotation difference.

**Used by:** LLaMA, Mistral, GPT-NeoX, PaLM

---

### Evolution 3: Grouped Query Attention - GQA (2023)
**The problem:** Multi-Head Attention needs separate K,V for each head. This is memory-expensive during inference (KV cache).

**The solution:** Share K,V across groups of query heads.
- MHA: 8 heads = 8Q, 8K, 8V
- GQA: 8 heads = 8Q, 2K, 2V (4 query heads share each K,V)
- MQA: 8 heads = 8Q, 1K, 1V (all queries share K,V)

**Used by:** LLaMA 2, Mistral, Gemma

---

### Evolution 4: SwiGLU Activation (2020)
**The problem:** ReLU in the FFN is a hard threshold — gradients are zero for negative inputs.

**The solution:** Gated Linear Unit with Swish activation. Smoother, better gradients.

```
SwiGLU(x) = Swish(xW_1) ⊙ (xW_2)
where Swish(x) = x × sigmoid(x)
```

**Used by:** LLaMA, PaLM, Mistral

---

### Evolution 5: RMSNorm (2019)
**The problem:** LayerNorm computes mean AND variance — expensive.

**The solution:** Only normalize by root mean square, skip the mean centering.

```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

**Why it works:** Empirically just as good, but faster.

**Used by:** LLaMA, Mistral, GPT-NeoX

---

### Evolution 6: KV Cache (Inference Optimization)
**The problem:** During autoregressive generation, we recompute K,V for ALL previous tokens at each step.

**The solution:** Cache K,V from previous steps, only compute for the new token.

---

## 🗂️ Project Structure

```
transformer-architecture/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Dependencies
│
├── 01_positional_encoding/
│   └── positional_encoding.py
│
├── 02_attention/
│   └── scaled_dot_product_attention.py
│
├── 03_multihead_attention/
│   └── multihead_attention.py
│
├── 04_feed_forward/
│   └── feed_forward.py
│
├── 05_layer_norm/
│   └── layer_norm_residual.py
│
├── 06_encoder/
│   └── encoder.py
│
├── 07_decoder/
│   └── decoder.py
│
├── 08_transformer/
│   └── transformer.py
│
├── beyond/
│   └── README.md           # All advances (hands-on + conceptual)
│
├── demo/
│   └── demo.py
│
└── REFERENCES.md           # Key papers & resources
```

---

## 🚀 Let's Begin!

When you're ready, we'll start with **Step 1: Positional Encoding**.

I'll explain the concept, then we'll implement it together — you can ask questions, suggest changes, and we'll make sure you understand each piece before moving on.

---

## 📖 References

See [REFERENCES.md](REFERENCES.md) for the complete collection of papers, organized by era:

**Foundation (2017-2019):**
- Attention Is All You Need, BERT, GPT-2, RMSNorm

**Scaling Era (2020-2023):**
- GPT-3, SwiGLU, RoPE, LLaMA, GQA, FlashAttention

**Frontier (2024-2026):**
- Looped Transformers, MLA, Test-Time Compute, ReLIT

Also see [EVOLUTION_TIMELINE.md](EVOLUTION_TIMELINE.md) for a visual timeline of all major milestones.
