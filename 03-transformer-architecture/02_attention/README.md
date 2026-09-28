# Step 2: Scaled Dot-Product Attention

---

## Why This Matters

Attention is **the** breakthrough that made transformers possible. Before attention, sequence models faced a fundamental bottleneck: information had to pass through a fixed-size hidden state.

```
RNN Information Flow:
    word1 → [h] → word2 → [h] → word3 → [h] → word4
                                               ↑
    Information about word1 must survive through this tiny bottleneck!
    
Transformer with Attention:
    word1 ←→ word2 ←→ word3 ←→ word4
      ↑_________|_________|_________|
      |_________|_________|
      |_________|
    
    Every word can DIRECTLY access every other word!
```

This direct access enables:
- **Long-range dependencies**: "The cat that ate the fish that the dog chased **was** tired" — "was" can directly attend to "cat"
- **Parallelization**: All attention computations happen simultaneously, not sequentially
- **Interpretability**: Attention weights show us what the model is "looking at"

---

## The Core Intuition

> **Intuition**: Attention is like a **searchable database lookup**. Each position asks "What information do I need?" (Query), each position advertises "Here's what I contain" (Key), and if there's a match, the requester gets the content (Value).

Think of it as a librarian (Query) looking through book titles (Keys) to find relevant books and extract their content (Values):

```
Query: "I need information about cats"

Library:
┌─────────────────────────────────────────────────────────┐
│ Key: "The"     │ Key: "cat"     │ Key: "sat"           │
│ Value: [info]  │ Value: [info]  │ Value: [info]        │
│ Match: 5%      │ Match: 70%     │ Match: 15%           │
└─────────────────────────────────────────────────────────┘

Result: 0.05×Value("The") + 0.70×Value("cat") + 0.15×Value("sat") + ...

The output is a WEIGHTED BLEND of all values!
```

---

## The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Let's break down each component:**

| Component | Shape | Meaning | Analogy |
|-----------|-------|---------|---------|
| Q (Query) | (seq_len, d_k) | "What am I looking for?" | Search terms |
| K (Key) | (seq_len, d_k) | "What do I contain?" | Document titles |
| V (Value) | (seq_len, d_v) | "What info do I provide?" | Document content |
| QK^T | (seq_len, seq_len) | Similarity scores | Relevance ranking |
| √d_k | scalar | Scaling factor | Keeps scores stable |
| softmax | (seq_len, seq_len) | Attention weights | Probability distribution |
| Output | (seq_len, d_v) | Weighted sum of values | Retrieved information |

---

## Step-by-Step Breakdown

### Step 1: Compute Raw Scores

```
scores = Q @ K^T    
Shape: (seq_len, d_k) @ (d_k, seq_len) → (seq_len, seq_len)
```

```
                          Keys (transposed)
                    ┌─────────────────────────┐
                    │ K₀   K₁   K₂   K₃  ...  │
                    └─────────────────────────┘
                          ↓     ↓    ↓    ↓
Queries            ┌─────────────────────────┐
  ┌──┐             │ q₀·k₀ q₀·k₁ q₀·k₂ ...   │  ← How much does pos 0
  │Q₀│        →    │ q₁·k₀ q₁·k₁ q₁·k₂ ...   │     attend to each position?
  │Q₁│             │ q₂·k₀ q₂·k₁ q₂·k₂ ...   │
  │Q₂│             │  ...                     │
  └──┘             └─────────────────────────┘
                         SCORES MATRIX
```

**Key insight**: The dot product `q·k` measures **similarity**. High dot product = high relevance.

---

### Step 2: Scale by √d_k

```
scaled_scores = scores / √d_k
```

> **Why scale?** This is subtle but crucial!

Without scaling, the variance of dot products grows with dimension:
```
d_k = 64:   Dot product variance ≈ 64
d_k = 512:  Dot product variance ≈ 512   ← Much larger!
```

Large values → softmax becomes extremely peaked:
```
softmax([1, 2, 3])     = [0.09, 0.24, 0.67]   ← Spread out ✓
softmax([10, 20, 30])  = [0.00, 0.00, 1.00]   ← Nearly one-hot ✗
```

Peaked softmax → vanishing gradients → training fails!

Scaling by √d_k normalizes the variance back to ~1.

---

### Step 3: Apply Mask (Optional)

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

Setting positions to -∞ makes their softmax weight effectively 0:
```
softmax([2.0, 3.0, -∞]) = [0.27, 0.73, 0.00]  ← Masked position ignored
```

---

### Step 4: Softmax → Attention Weights

```
weights = softmax(scores, dim=-1)
```

Each row becomes a probability distribution (sums to 1):
```
                    The    cat    sat     on    the    mat
             ┌──────────────────────────────────────────────┐
"The"   sees │ 0.10   0.30   0.25   0.10   0.10   0.15     │ = 1.0
"cat"   sees │ 0.15   0.20   0.35   0.10   0.05   0.15     │ = 1.0  
"sat"   sees │ 0.10   0.40   0.15   0.15   0.05   0.15     │ = 1.0
  ...        └──────────────────────────────────────────────┘
             
Each row is the "attention budget" for that position.
```

---

### Step 5: Weighted Sum of Values

```
output = weights @ V
Shape: (seq_len, seq_len) @ (seq_len, d_v) → (seq_len, d_v)
```

```
Position 0's output = 0.10×V₀ + 0.30×V₁ + 0.25×V₂ + 0.10×V₃ + 0.10×V₄ + 0.15×V₅

The output is a CONTEXT-AWARE blend of all token representations!
```

---

## ASCII Visualization: Full Attention Flow

```
INPUT: "The cat sat on the mat"
        ↓
   ┌─────────────────────────────────────────────────────┐
   │              TOKEN EMBEDDINGS                        │
   │   [The]  [cat]  [sat]  [on]  [the]  [mat]           │
   │    ↓       ↓      ↓     ↓      ↓      ↓             │
   │   E₀     E₁     E₂    E₃     E₄     E₅              │
   └─────────────────────────────────────────────────────┘
               ↓ Linear projections (learned)
   ┌─────────────────────────────────────────────────────┐
   │   Q₀     Q₁     Q₂    Q₃     Q₄     Q₅  (Queries)   │
   │   K₀     K₁     K₂    K₃     K₄     K₅  (Keys)      │
   │   V₀     V₁     V₂    V₃     V₄     V₅  (Values)    │
   └─────────────────────────────────────────────────────┘
               ↓ Attention computation
   ┌─────────────────────────────────────────────────────┐
   │           ATTENTION WEIGHTS MATRIX                   │
   │              K₀   K₁   K₂   K₃   K₄   K₅            │
   │         Q₀ [.10  .30  .25  .10  .10  .15]           │
   │         Q₁ [.15  .20  .35  .10  .05  .15]           │
   │         Q₂ [.10  .40  .15  .15  .05  .15]           │
   │         Q₃ [.20  .20  .30  .10  .10  .10]           │
   │         Q₄ [.15  .25  .20  .15  .10  .15]           │
   │         Q₅ [.10  .35  .25  .10  .05  .15]           │
   └─────────────────────────────────────────────────────┘
               ↓ Weighted sum with values
   ┌─────────────────────────────────────────────────────┐
   │           CONTEXT-AWARE OUTPUTS                      │
   │   O₀     O₁     O₂    O₃     O₄     O₅              │
   │    ↓       ↓      ↓     ↓      ↓      ↓             │
   │  Each output is a weighted blend of ALL values!     │
   └─────────────────────────────────────────────────────┘
```

---

## Masking: Controlling Attention

### Causal Mask (for Autoregressive/Decoder)

Prevents positions from attending to **future** tokens:

```
                 The   cat   sat    on   the   mat
            ┌────────────────────────────────────────┐
"The"  can  │  ✓     ✗     ✗     ✗     ✗     ✗      │  Only sees itself
"cat"  can  │  ✓     ✓     ✗     ✗     ✗     ✗      │  Sees The, cat
"sat"  can  │  ✓     ✓     ✓     ✗     ✗     ✗      │  Sees The, cat, sat
"on"   can  │  ✓     ✓     ✓     ✓     ✗     ✗      │  ...
"the"  can  │  ✓     ✓     ✓     ✓     ✓     ✗      │
"mat"  can  │  ✓     ✓     ✓     ✓     ✓     ✓      │  Sees everything

Lower triangular matrix of 1s (allowed) and 0s (blocked)
```

**Why needed?** During generation, future tokens don't exist yet!

### Padding Mask

Prevents attention to padding tokens:

```
Batch of sequences (padded to length 6):
   Seq 1: ["The", "cat", "sat", <PAD>, <PAD>, <PAD>]
   Seq 2: ["Hello", "world", <PAD>, <PAD>, <PAD>, <PAD>]
   
Padding mask:
   Seq 1: [1, 1, 1, 0, 0, 0]  ← Attend only to real tokens
   Seq 2: [1, 1, 0, 0, 0, 0]
```

---

## Comparison: Attention Mechanisms

| Type | Formula | Use Case | Complexity |
|------|---------|----------|------------|
| **Scaled Dot-Product** | softmax(QK^T/√d_k)V | Standard transformers | O(n²d) |
| **Additive (Bahdanau)** | v^T·tanh(W_q·Q + W_k·K) | Original seq2seq attention | O(n²d) |
| **Multi-Head** | Concat(head₁...headₕ)W_O | All modern transformers | O(n²d) |
| **Linear/Efficient** | φ(Q)φ(K)^T·V | Long sequences (Performer) | O(nd²) |
| **Sparse** | Only attend to selected positions | Long sequences (Longformer) | O(n·k) |
| **Flash Attention** | Same math, optimized memory | Training efficiency | O(n²d), faster |

---

## Common Pitfalls & Debugging

### Pitfall 1: Forgetting to Scale

```python
# ❌ WRONG: No scaling → unstable training
scores = torch.matmul(Q, K.transpose(-2, -1))
weights = F.softmax(scores, dim=-1)

# ✅ CORRECT: Scale by √d_k
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
```

### Pitfall 2: Wrong Softmax Dimension

```python
# ❌ WRONG: Softmax over queries (columns sum to 1)
weights = F.softmax(scores, dim=-2)  # Each key receives total attention of 1

# ✅ CORRECT: Softmax over keys (rows sum to 1)
weights = F.softmax(scores, dim=-1)  # Each query distributes its attention
```

### Pitfall 3: Mask Shape Mismatch

```python
# ❌ WRONG: Mask doesn't broadcast properly
mask = torch.ones(seq_len)  # Shape: (seq_len,)

# ✅ CORRECT: Mask with proper dimensions
mask = torch.ones(1, 1, seq_len)  # Shape: (1, 1, seq_len) for broadcasting
```

### Pitfall 4: Mask Values

```python
# ❌ WRONG: Using 0 to mask (0 is a valid score!)
scores = scores * mask

# ✅ CORRECT: Use -inf so softmax gives 0 weight
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## Implementation: From Scratch

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,   # (batch, seq_len, d_k)
    key: torch.Tensor,     # (batch, seq_len, d_k)
    value: torch.Tensor,   # (batch, seq_len, d_v)
    mask: torch.Tensor = None,
    dropout: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Returns:
        output: (batch, seq_len, d_v) - context-aware representations
        weights: (batch, seq_len, seq_len) - attention pattern
    """
    d_k = query.size(-1)
    
    # Step 1: Raw scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax → weights
    weights = F.softmax(scores, dim=-1)
    
    # Optional dropout
    if dropout > 0:
        weights = F.dropout(weights, p=dropout)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(weights, value)
    
    return output, weights


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Lower triangular mask for autoregressive attention."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
```

---

## Implementation: Using PyTorch

PyTorch 2.0+ provides an optimized implementation:

```python
import torch.nn.functional as F

# PyTorch's built-in (uses Flash Attention when possible)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,        # Optional mask
    dropout_p=0.1,         # Dropout probability
    is_causal=True         # Automatically creates causal mask
)

# Note: Built-in doesn't return attention weights (for efficiency)
# Use custom implementation if you need to visualize attention
```

---

## Hands-On Experiment

Run the implementation to see attention in action:

```bash
python 02_attention/scaled_dot_product_attention.py
```

**What you'll see:**
1. Attention weights matrix for "The cat sat on the mat"
2. How causal masking zeros out future positions
3. Visualization comparing masked vs unmasked attention

---

## Files in This Module

| File | Description |
|------|-------------|
| `scaled_dot_product_attention.py` | Complete implementation with demo |
| `attention_visualization.png` | Generated attention heatmaps |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Attention enables direct access**: Every position can "look at" every other position, solving the RNN information bottleneck

2. **Q, K, V paradigm**: Queries ask questions, keys advertise content, values provide information — the output is a weighted blend of values

3. **Scaling is critical**: Dividing by √d_k keeps dot products stable, preventing softmax saturation and gradient vanishing

4. **Masking controls visibility**: Causal masks prevent future-peeking (for generation), padding masks ignore filler tokens

5. **Attention is interpretable**: The weights matrix shows what the model is "looking at" — a rare window into neural network reasoning

6. **Quadratic complexity**: O(n²) in sequence length — this is why long-context models need special attention variants

7. **The output transformation**: Each position's output is a context-aware blend of ALL value vectors, weighted by relevance

---

## What's Next?

Single attention is powerful, but limited — it can only capture one type of relationship at a time. In **Step 3: Multi-Head Attention**, we'll run multiple attention operations in parallel, letting the model capture different types of relationships (syntax, semantics, coreference) simultaneously.

