# Transformer Architecture: Beyond the Basics (2018-2026)

Key architectural improvements since the original 2017 paper. We implement the foundational ones hands-on here; **advanced topics have moved to [Course 14: Emerging AI Trends](../../14-emerging-ai-trends/)**.

---

## 🔧 Hands-On Implementations (This Course)

### 1. Pre-Layer Normalization (2019)
**Paper**: GPT-2

**Problem**: Post-LN transformers are unstable at scale.

**Solution**: Move LayerNorm before the sublayer.
```python
# Original (Post-LN):  output = LayerNorm(x + Sublayer(x))
# Modern (Pre-LN):     output = x + Sublayer(LayerNorm(x))
```

**Why it works**: Gradients flow directly through residual path.

**Used by**: GPT-2/3/4, LLaMA, Mistral, most modern models

**File**: `pre_layer_norm.py`

---

### 2. RMSNorm (2019)
**Paper**: Root Mean Square Layer Normalization

**Problem**: LayerNorm computes mean AND variance — expensive.

**Solution**: Only normalize by root mean square.
```python
RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ
```

**Why it works**: Empirically equivalent quality, 10-15% faster.

**Used by**: LLaMA, Mistral, GPT-NeoX

**File**: `rmsnorm.py`

---

### 3. SwiGLU Activation (2020)
**Paper**: GLU Variants Improve Transformer

**Problem**: ReLU has zero gradient for negative inputs.

**Solution**: Gated Linear Unit with Swish activation.
```python
SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)
Swish(x) = x × sigmoid(x)
```

**Why it works**: Smooth gating, better optimization.

**Used by**: LLaMA, PaLM, Mistral

**File**: `swiglu.py`

---

### 4. Rotary Position Embeddings - RoPE (2021)
**Paper**: RoFormer

**Problem**: Sinusoidal encodings don't capture relative positions well.

**Solution**: Encode position by rotating Q and K vectors.
```python
q_rotated = rotate(q, position_m)
k_rotated = rotate(k, position_n)
# Dot product naturally encodes relative position (m - n)
```

**Why it works**: Relative position emerges from rotation difference; better extrapolation.

**Used by**: LLaMA, Mistral, GPT-NeoX, PaLM

**File**: `rope.py`

---

### 5. Grouped Query Attention - GQA (2023)
**Paper**: GQA: Training Generalized Multi-Query Transformer

**Problem**: MHA needs separate K,V per head — expensive KV cache.

**Solution**: Share K,V across groups of query heads.
```
MHA: 8 heads = 8Q, 8K, 8V  (full)
GQA: 8 heads = 8Q, 2K, 2V  (4 queries share each KV)
MQA: 8 heads = 8Q, 1K, 1V  (all share one KV)
```

**Why it works**: Reduces memory with minimal quality loss.

**Used by**: LLaMA 2, Mistral, Gemma

**File**: `grouped_query_attention.py`

---

### 6. FlashAttention (2022)
**Paper**: FlashAttention: Fast and Memory-Efficient Exact Attention

**Problem**: Standard attention is O(n²) memory.

**Solution**: Tiled, IO-aware algorithm — exact same math, much faster.

**Why it works**: Minimizes memory reads/writes via kernel fusion.

**Used by**: Everything modern (training and inference)

**Status**: 📖 Conceptual (implementation is CUDA-specific)

---

## 📖 Advanced Topics → Course 14

The following topics are covered in depth in **[Course 14: Emerging AI Trends](../../14-emerging-ai-trends/)**:

| Topic | Course 14 Module | Why There? |
|-------|------------------|------------|
| **Mixture of Experts (MoE)** | Module 10 | Scaling paradigm, broader context |
| **Multi-head Latent Attention (MLA)** | Module 13 | KV compression, cutting-edge |
| **Looped Transformers** | Module 4 | Reasoning paradigms, System 2 |
| **State Space Models (Mamba)** | Module 5 | Alternative architectures |
| **Hybrid Architectures** | Module 6 | SSM + Transformer combinations |

These topics require broader context beyond transformer architecture — understanding reasoning paradigms, efficiency tradeoffs, and the evolving AI landscape.

---

## Modern Architecture Recipe (LLaMA-style)

Combining the hands-on advances from this course:

```python
class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.norm1 = RMSNorm(d_model)          # Evolution 2
        self.norm2 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(     # Evolution 5
            d_model, num_q_heads=32, num_kv_heads=8
        )
        self.ffn = SwiGLUFeedForward(d_model, d_ff)  # Evolution 3
    
    def forward(self, x, freqs_cis):
        # Pre-LN (Evolution 1)
        h = x + self.attn(self.norm1(x), freqs_cis)  # RoPE via freqs_cis (Evolution 4)
        out = h + self.ffn(self.norm2(h))
        return out
```

---

## Summary Table

| Evolution | Year | Replaces | Benefit | Coverage |
|-----------|------|----------|---------|----------|
| Pre-LN | 2019 | Post-LN | Training stability | 🔧 This course |
| RMSNorm | 2019 | LayerNorm | 10-15% faster | 🔧 This course |
| SwiGLU | 2020 | ReLU FFN | Better gradients | 🔧 This course |
| RoPE | 2021 | Sinusoidal PE | Relative positions | 🔧 This course |
| GQA | 2023 | MHA | KV cache efficiency | 🔧 This course |
| FlashAttention | 2022 | Standard attn | Speed + memory | 📖 This course |
| MoE | 2024 | Dense FFN | Scale without compute | → Course 14 |
| MLA | 2024 | Standard KV | 90% KV reduction | → Course 14 |
| Looped | 2025+ | Fixed depth | Adaptive reasoning | → Course 14 |
| Mamba | 2023 | Attention | Linear complexity | → Course 14 |

---

## What's Next?

After completing this course:
- **Course 04** (LLM Internals): How transformers become LLMs
- **Course 14** (Emerging AI Trends): Advanced architectures and frontier research
