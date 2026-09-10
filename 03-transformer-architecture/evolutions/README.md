# Transformer Evolutions (2018-2026)

Key architectural improvements since the original 2017 paper. We'll implement the foundational ones hands-on; frontier topics are for conceptual understanding.

---

## 🔧 Hands-On Implementations

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

## 📖 Conceptual / Reading

### 7. Mixture of Experts - MoE (2024)
**Paper**: Mixtral

**Concept**: Multiple "expert" FFN layers, router selects 2 per token.
```
8 experts × 7B each = 56B total params
But only 2 active = 14B compute per token
```

**Why it matters**: Scale parameters without scaling compute.

**Used by**: Mixtral, GPT-4 (rumored), DeepSeek

---

### 8. Multi-head Latent Attention - MLA (2024)
**Paper**: DeepSeek-V2

**Concept**: Compress KV cache by 90%+ using learned latent projections.

**Why it matters**: Enables much longer contexts affordably.

---

### 9. Looped Transformers (2025-2026)
**Papers**: 
- "Reasoning with Latent Thoughts" (2025)
- "Fully Looped Transformer" (2026)
- "Loop, Think, & Generalize" (2026)

**Concept**: Instead of fixed N layers, loop a shallow block until "done thinking."
```python
# Fixed depth (traditional)
for layer in layers:
    x = layer(x)

# Looped (adaptive)
for _ in range(max_loops):
    x = loop_block(x)
    if converged(x):
        break
```

**Why it matters**: 
- Adaptive compute per input (simple → few loops, hard → more loops)
- Better reasoning extrapolation (train on 5 steps → solve 10 steps)
- Theoretical proof that loops ≥ depth for algorithmic reasoning

**Challenge**: Training stability (solved in 2026 papers).

---

### 10. Latent Reasoning / ReLIT (2026)
**Paper**: ReLIT Framework

**Concept**: Model refines internal "thinking vector" before generating output.

**Why it matters**: Decouples reasoning depth from output tokens — model can "ponder" internally.

---

## Modern Architecture Recipe (LLaMA-style)

Combining the hands-on evolutions:

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

| Evolution | Year | Replaces | Benefit | Hands-on? |
|-----------|------|----------|---------|-----------|
| Pre-LN | 2019 | Post-LN | Training stability | 🔧 Yes |
| RMSNorm | 2019 | LayerNorm | 10-15% faster | 🔧 Yes |
| SwiGLU | 2020 | ReLU FFN | Better gradients | 🔧 Yes |
| RoPE | 2021 | Sinusoidal PE | Relative positions | 🔧 Yes |
| GQA | 2023 | MHA | KV cache efficiency | 🔧 Yes |
| FlashAttention | 2022 | Standard attn | Speed + memory | 📖 Read |
| MoE | 2024 | Dense FFN | Scale without compute | 📖 Read |
| Looped | 2025+ | Fixed depth | Adaptive reasoning | 📖 Read |
