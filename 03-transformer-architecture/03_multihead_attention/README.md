# Step 3: Multi-Head Attention

---

## Why This Matters

Single-head attention is powerful but limited — it can only capture **one type of relationship** at a time. Language, however, is rich with simultaneous, overlapping relationships:

```
Sentence: "The cat that I saw yesterday was sleeping on the mat"
           ↑    ↑         ↑                  ↑              ↑
           │    │         │                  │              │
           └────│─────────│──── syntactic ───┴──── "was" agrees with "cat"
                │         │
                └─────────┴──── semantic ──── "cat" relates to "sleeping"
                          │
                          └──── temporal ──── "yesterday" modifies "saw"
```

Multi-head attention solves this by running **h parallel attention operations**, each learning to focus on different relationship types. It's like having 8 different "experts" analyzing the sentence simultaneously.

---

## The Core Intuition

> **Intuition**: Think of multi-head attention as a **team of analysts** looking at the same document. One analyst tracks grammar, another tracks meaning, another tracks position. Each brings a different perspective, and together they build a complete understanding.

```
              INPUT SENTENCE
                    │
     ┌──────────────┼──────────────┐
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Head 1  │   │ Head 2  │   │ Head 3  │  ... (h heads total)
│"syntax" │   │"meaning"│   │"nearby" │
│ expert  │   │ expert  │   │ expert  │
└────┬────┘   └────┬────┘   └────┬────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
                    ▼
           COMBINED OUTPUT
      (richer than any single head)
```

---

## The Formula

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) × W_O

where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

**Key dimensions:**
| Parameter | Typical Value | Meaning |
|-----------|---------------|---------|
| d_model | 512 | Total model dimension |
| h (num_heads) | 8 | Number of parallel attention heads |
| d_k = d_v | 64 (= 512/8) | Dimension per head |

**The key constraint**: `d_model = h × d_k`

We're **splitting** the representation space, not adding to it!

---

## Step-by-Step Breakdown

### Step 1: Project to Multiple Subspaces

Each head has its own projection matrices:

```
For head i:
    Q_i = Q × W_Q^i    (seq_len, d_model) × (d_model, d_k) → (seq_len, d_k)
    K_i = K × W_K^i    (seq_len, d_model) × (d_model, d_k) → (seq_len, d_k)  
    V_i = V × W_V^i    (seq_len, d_model) × (d_model, d_k) → (seq_len, d_k)
```

### Step 2: Parallel Attention

Each head computes scaled dot-product attention independently:

```
head_i = Attention(Q_i, K_i, V_i)
       = softmax(Q_i × K_i^T / √d_k) × V_i
       
Shape: (seq_len, d_k)
```

### Step 3: Concatenate Heads

```
Concat = [head_1 | head_2 | ... | head_h]
Shape: (seq_len, h × d_k) = (seq_len, d_model)
```

### Step 4: Output Projection

```
Output = Concat × W_O
Shape: (seq_len, d_model) × (d_model, d_model) → (seq_len, d_model)
```

---

## ASCII Visualization: The Full Flow

```
INPUT: x ∈ (batch, seq_len, d_model=512)
       │
       ├──────────────┬──────────────┬──────────────┐
       │              │              │              │
       ▼              ▼              ▼              ▼
    ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
    │  W_Q │      │ W_Q' │      │ W_Q''│      │ ...  │   (conceptually h
    │  W_K │      │ W_K' │      │ W_K''│      │      │    separate projections)
    │  W_V │      │ W_V' │      │ W_V''│      │      │
    └──┬───┘      └──┬───┘      └──┬───┘      └──┬───┘
       │              │              │              │
       ▼              ▼              ▼              ▼
    Q₁,K₁,V₁      Q₂,K₂,V₂      Q₃,K₃,V₃          Qₕ,Kₕ,Vₕ
    (seq, 64)     (seq, 64)     (seq, 64)         (seq, 64)
       │              │              │              │
       ▼              ▼              ▼              ▼
   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
   │Attention│   │Attention│   │Attention│   │Attention│
   │  Head 1 │   │  Head 2 │   │  Head 3 │   │  Head h │
   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
     head₁         head₂         head₃         headₕ
    (seq, 64)     (seq, 64)     (seq, 64)     (seq, 64)
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ CONCATENATE │
                    │ (seq, 512)  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     W_O     │
                    │ (512 → 512) │
                    └──────┬──────┘
                           │
                           ▼
                    OUTPUT (batch, seq, 512)
```

---

## The Efficient Implementation Trick

The conceptual model above shows h separate projections, but that's inefficient. The actual implementation uses **one big matrix** and **reshapes**:

```
CONCEPTUAL (slow):                    ACTUAL (fast):
┌──────────────────────────┐          ┌──────────────────────────┐
│ h separate W_Q matrices  │          │ One big W_Q matrix       │
│ each (d_model, d_k)      │    =     │ (d_model, d_model)       │
│                          │          │ Then reshape into heads  │
└──────────────────────────┘          └──────────────────────────┘

Both are mathematically IDENTICAL, but the big matrix is:
- Faster (one big matmul vs h small ones)
- More parallelizable (GPU-friendly)
```

**The reshape operation:**

```python
# After: Q = x @ W_Q  → (batch, seq, d_model)
# Reshape: (batch, seq, d_model) → (batch, seq, h, d_k) → (batch, h, seq, d_k)

Q = Q.view(batch, seq, num_heads, d_k)  # Split last dim
Q = Q.transpose(1, 2)                     # Swap seq and heads
# Now each head is a separate "batch" dimension!
```

---

## What Do Different Heads Learn?

Research on BERT and GPT models reveals heads specialize:

```
HEAD SPECIALIZATION PATTERNS (observed in trained models)
═══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│ POSITIONAL HEADS                                        │
│ "I look at nearby tokens"                               │
│                                                         │
│   [The] [cat] [sat] → Head attends: [cat] to [The,sat]  │
│    ←──────────→                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ SYNTACTIC HEADS                                         │
│ "I track grammatical relationships"                     │
│                                                         │
│   [The] [cat] ... [was] → Head: [was] attends to [cat]  │
│          └──── subject-verb ────┘                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ SEPARATOR HEADS                                         │
│ "I focus on punctuation and special tokens"             │
│                                                         │
│   [CLS] ... [.] → Head: all tokens attend to [CLS], [.] │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ RARE WORD HEADS                                         │
│ "I pay extra attention to uncommon words"               │
│                                                         │
│   [The] [archaeopteryx] [flew] → Heavy weight on rare   │
└─────────────────────────────────────────────────────────┘
```

---

## Projections vs Heads: Clarification

A common confusion:

| Concept | What It Is | Quantity |
|---------|------------|----------|
| **Projection** | A learned linear transformation (W_Q, W_K, W_V, W_O) | 4 matrices |
| **Head** | A parallel attention computation using a slice of Q, K, V | h heads |

```
Projections PRODUCE the Q, K, V for all heads:
    W_Q: (d_model, d_model) → splits into h heads of Q
    W_K: (d_model, d_model) → splits into h heads of K
    W_V: (d_model, d_model) → splits into h heads of V

Heads CONSUME slices of Q, K, V:
    Head 1: Q[:, :, 0:64],  K[:, :, 0:64],  V[:, :, 0:64]
    Head 2: Q[:, :, 64:128], K[:, :, 64:128], V[:, :, 64:128]
    ...

W_O combines after concatenation:
    W_O: (d_model, d_model) → applied AFTER heads are concatenated
```

---

## Comparison: Single-Head vs Multi-Head

| Aspect | Single-Head | Multi-Head |
|--------|-------------|------------|
| Attention ops | 1 | h parallel |
| Relationship types | 1 | h different |
| Dimension per op | d_model | d_k = d_model/h |
| Parameters | ~3×d_model² | ~4×d_model² (similar) |
| Expressiveness | Limited | Rich |
| Interpretability | One pattern | h patterns to analyze |

**Why similar parameter count?**
- Single-head: W_Q, W_K, W_V each (d_model, d_model)
- Multi-head: Same total, just conceptually split + W_O

---

## Common Pitfalls & Debugging

### Pitfall 1: Dimension Not Divisible

```python
# ❌ WRONG: 512 / 7 = 73.14... not an integer!
mha = MultiHeadAttention(d_model=512, num_heads=7)

# ✅ CORRECT: 512 / 8 = 64 ✓
mha = MultiHeadAttention(d_model=512, num_heads=8)
```

### Pitfall 2: Forgetting to Transpose

```python
# ❌ WRONG: Shapes won't broadcast correctly
Q = Q.view(batch, seq, num_heads, d_k)  # Missing transpose!

# ✅ CORRECT: Transpose to put heads in batch dimension
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
```

### Pitfall 3: Missing contiguous() Before View

```python
# ❌ WRONG: transpose() doesn't change memory layout
x = x.transpose(1, 2)
x = x.view(batch, seq, d_model)  # ERROR: not contiguous!

# ✅ CORRECT: Call contiguous() first
x = x.transpose(1, 2).contiguous()
x = x.view(batch, seq, d_model)  # Works ✓
```

### Pitfall 4: Mixing Up Self-Attention and Cross-Attention

```python
# Self-attention: Q = K = V (same input)
output = mha(x, x, x)

# Cross-attention: Q from decoder, K/V from encoder
output = mha(decoder_x, encoder_out, encoder_out)  # Different inputs!
```

---

## Implementation: From Scratch

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Single big projections (efficient)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        batch, heads, seq, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # Project
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))
        
        # Scaled dot-product attention (all heads in parallel)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        attn_output = torch.matmul(weights, V)
        
        # Combine heads and project
        output = self.W_O(self.combine_heads(attn_output))
        
        return output
```

---

## Implementation: Using PyTorch

```python
import torch.nn as nn

# PyTorch's built-in (highly optimized)
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True  # Important: use (batch, seq, dim) format
)

# Usage
output, attn_weights = mha(query, key, value, attn_mask=mask)

# Note: attn_weights are averaged across heads by default
# Use need_weights=True and average_attn_weights=False for per-head weights
```

---

## Hands-On Experiment

Run the implementation to visualize how different heads attend differently:

```bash
python 03_multihead_attention/multihead_attention.py
```

**What you'll see:**
1. Attention patterns for each head (they're different!)
2. How causal masking affects all heads equally
3. Shape transformations through the forward pass

---

## Files in This Module

| File | Description |
|------|-------------|
| `multihead_attention.py` | Complete implementation with visualization |
| `multihead_attention_visualization.png` | Per-head attention heatmaps |
| `multihead_causal_visualization.png` | Same with causal masking |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Multiple perspectives**: Each head learns to capture different relationship types (syntax, semantics, position) — together they're more powerful than a single head

2. **Split, don't add**: Multi-head splits d_model into h pieces of d_k each. Total parameters are similar to single-head, but expressiveness is much greater

3. **Efficient implementation**: Use one big projection matrix and reshape, rather than h separate small matrices — mathematically equivalent but GPU-friendly

4. **Four projection matrices**: W_Q, W_K, W_V project inputs, W_O combines head outputs — all are learned parameters

5. **Heads specialize**: During training, heads naturally learn to focus on different patterns — some attend to syntax, others to nearby words, others to rare tokens

6. **Self vs Cross attention**: For self-attention Q=K=V, for cross-attention (in decoder) Q comes from decoder, K/V from encoder

7. **The concat→project pattern**: Concatenate all head outputs, then apply W_O — this lets the model learn optimal head combinations

---

## What's Next?

Multi-head attention captures relationships between positions, but it's **linear** — it just computes weighted sums. In **Step 4: Feed-Forward Network**, we'll add the non-linear transformations that give the model its expressive power.

