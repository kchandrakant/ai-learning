# Step 4: Position-wise Feed-Forward Network

---

## Why This Matters

Attention is powerful, but it has a fundamental limitation: **it's linear**. No matter how sophisticated the attention patterns, the output is always a weighted sum of values — a linear combination.

```
Attention output = Σ (weight_i × value_i)    ← Just weighted averaging!
```

Neural networks need non-linearity to approximate complex functions. Without it:
- Stacking 100 linear layers = 1 linear layer (matrix multiplication is associative)
- The model couldn't learn XOR, let alone language understanding

The Feed-Forward Network (FFN) provides this crucial **non-linear transformation**:

```
TRANSFORMER BLOCK = Attention (linear mixing) + FFN (non-linear processing)
                         ↓                           ↓
                    "Gather info"              "Process info"
```

---

## The Core Intuition

> **Intuition**: Think of attention and FFN as two complementary operations. Attention is a **conference call** where tokens share information. FFN is **individual thinking** where each token processes what it learned.

```
Step 1: Attention (Communication)
┌─────────────────────────────────────────────────┐
│  Token_1  ←──────→  Token_2  ←──────→  Token_3  │
│     │                  │                  │     │
│     └──────────────────┼──────────────────┘     │
│                        │                        │
│   "Let me hear what everyone has to say..."     │
└─────────────────────────────────────────────────┘
                         ↓
Step 2: FFN (Processing)
┌─────────────────────────────────────────────────┐
│  [Token_1]         [Token_2]         [Token_3]  │
│      │                 │                  │     │
│    ┌─┴─┐            ┌─┴─┐             ┌─┴─┐    │
│    │FFN│            │FFN│             │FFN│    │
│    └─┬─┘            └─┬─┘             └─┬─┘    │
│      │                 │                  │     │
│   "Now let me think about what I heard..."      │
└─────────────────────────────────────────────────┘

Same FFN applied to each token INDEPENDENTLY (position-wise)
```

---

## The Formula

**Original Transformer (ReLU):**
```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
```

**Modern Transformers (SwiGLU):**
```
FFN(x) = (Swish(x · W_gate) ⊙ (x · W_up)) · W_down
```

Where:
- `W₁`: (d_model, d_ff) — expand to higher dimension
- `W₂`: (d_ff, d_model) — contract back
- `d_ff = 4 × d_model` typically (2048 for d_model=512)
- `⊙` = element-wise multiplication (gating)

---

## The Expand-Contract Pattern

```
                    d_ff = 2048
                   ┌─────────────────────┐
                   │                     │
                   │   "Room to think"   │
                   │                     │
                   └─────────────────────┘
                  ╱                       ╲
                 ╱  EXPAND (Linear + Act)  ╲ CONTRACT (Linear)
                ╱                           ╲
     d_model = 512                    d_model = 512
    ┌───────────┐                    ┌───────────┐
    │   INPUT   │  ──────────────→   │  OUTPUT   │
    └───────────┘                    └───────────┘

Why expand?
- More dimensions = more capacity to learn complex functions
- Think of it as "unpacking" the information, processing it, then "repacking"
- Similar to bottleneck layers in ResNets, but inverted
```

---

## Position-Wise: What It Really Means

The same FFN is applied to **each position independently**:

```
If you have 10 tokens, the FFN runs 10 times conceptually:

Token 0: FFN(x[0]) → output[0]
Token 1: FFN(x[1]) → output[1]
Token 2: FFN(x[2]) → output[2]
...
Token 9: FFN(x[9]) → output[9]

SAME weights (W₁, W₂) for all positions!
But each gets DIFFERENT input, so DIFFERENT output.

In practice: done as one big matrix multiply (parallel)
```

**Question from our learning session**: "If we have 10 tokens, how many times does FFN run?"
- **Conceptually**: 10 times (once per token)
- **Actually**: 1 parallel operation (matrix multiply handles all positions at once)

---

## ASCII Visualization: Complete FFN Flow

```
INPUT: x ∈ (batch=2, seq_len=6, d_model=512)

   ┌──────────────────────────────────────────────────────────────┐
   │  Position:    0       1       2       3       4       5      │
   │              [The]   [cat]   [sat]   [on]   [the]   [mat]    │
   │               ↓       ↓       ↓       ↓       ↓       ↓      │
   │              512     512     512     512     512     512     │
   └──────────────────────────────────────────────────────────────┘
                           ↓ Linear1: W₁ (512 → 2048)
   ┌──────────────────────────────────────────────────────────────┐
   │  HIDDEN LAYER (expanded)                                     │
   │              2048    2048    2048    2048    2048    2048     │
   └──────────────────────────────────────────────────────────────┘
                           ↓ ReLU (or GELU, SwiGLU)
   ┌──────────────────────────────────────────────────────────────┐
   │  AFTER ACTIVATION (non-linearity applied)                    │
   │              2048    2048    2048    2048    2048    2048     │
   └──────────────────────────────────────────────────────────────┘
                           ↓ Linear2: W₂ (2048 → 512)
   ┌──────────────────────────────────────────────────────────────┐
   │  Position:    0       1       2       3       4       5      │
   │               ↓       ↓       ↓       ↓       ↓       ↓      │
   │              512     512     512     512     512     512     │
   └──────────────────────────────────────────────────────────────┘

OUTPUT: same shape as input (batch=2, seq_len=6, d_model=512)
```

---

## Activation Functions: Evolution

| Activation | Formula | Used In | Pros | Cons |
|------------|---------|---------|------|------|
| **ReLU** | max(0, x) | Original Transformer | Fast, simple | "Dead neurons" (gradient=0 for x<0) |
| **GELU** | x·Φ(x) | BERT, GPT-2 | Smooth, better gradients | Slightly slower |
| **SiLU/Swish** | x·σ(x) | EfficientNet | Non-monotonic, smooth | Similar to GELU |
| **SwiGLU** | Swish(x·W_g)⊙(x·W_u) | LLaMA, PaLM, Mistral | Best empirical results | 3 matrices vs 2 |

```
ACTIVATION COMPARISON (x from -3 to 3):

ReLU:        _______/
            |      /
            |     /
            |____/___________
           -3    0    3

GELU:             ____----
                 /
            ___/
           -3    0    3

Swish:           ____----
              __/
            _/
           -3    0    3
           (dips slightly below 0!)
```

**Why SwiGLU is winning**: The gating mechanism lets the network learn *what* to let through, not just *how much* to let through.

---

## SwiGLU: The Modern Choice

```
Standard FFN:
    x → [W₁] → ReLU → [W₂] → output
    
SwiGLU FFN:
    x → [W_gate] → Swish ─┐
                          ├─→ ⊙ (multiply) → [W_down] → output
    x → [W_up] ───────────┘
```

The gating creates a **multiplicative interaction**:
- W_gate path decides "what to keep"
- W_up path provides "what's available"
- Element-wise multiply: selective information flow

**Trade-off**: 3 matrices instead of 2, so d_ff is typically reduced to ~2.67× (8/3) to keep similar parameter count.

---

## Parameter Count: Where the Mass Lives

For a typical transformer layer (d_model=512, d_ff=2048, heads=8):

```
COMPONENT               PARAMETERS         % OF LAYER
─────────────────────────────────────────────────────
Multi-Head Attention:
  W_Q (512×512)         262,144
  W_K (512×512)         262,144
  W_V (512×512)         262,144
  W_O (512×512)         262,144
  Subtotal:           1,048,576            33%

Feed-Forward Network:
  W_1 (512×2048)      1,048,576
  b_1 (2048)              2,048
  W_2 (2048×512)      1,048,576
  b_2 (512)                 512
  Subtotal:           2,099,712            67%
─────────────────────────────────────────────────────
TOTAL:                3,148,288           100%

FFN is 2/3 of parameters! 
(But attention has quadratic compute due to seq_len²)
```

---

## Comparison: Attention vs FFN

| Aspect | Attention | FFN |
|--------|-----------|-----|
| **What it does** | Mixes information between positions | Transforms each position independently |
| **Mathematical nature** | Linear (weighted sum) | Non-linear (activation function) |
| **Interaction** | Token ↔ Token | Token → Transform → Token |
| **Parameters** | ~1/3 of layer | ~2/3 of layer |
| **Compute complexity** | O(n² × d) | O(n × d × d_ff) |
| **Role** | Communication | Computation |

---

## Common Pitfalls & Debugging

### Pitfall 1: Wrong Expansion Ratio

```python
# ❌ WRONG: No expansion defeats the purpose
ffn = FeedForward(d_model=512, d_ff=512)  # Same dim!

# ✅ CORRECT: 4× expansion is standard
ffn = FeedForward(d_model=512, d_ff=2048)
```

### Pitfall 2: Forgetting Activation

```python
# ❌ WRONG: No activation = still linear!
def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)  # This is just linear1 @ linear2!
    return x

# ✅ CORRECT: Activation between layers
def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)  # Critical!
    x = self.linear2(x)
    return x
```

### Pitfall 3: Activation After Final Layer

```python
# ❌ WRONG: Activation after output layer limits range
def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))  # Output always ≥ 0!
    return x

# ✅ CORRECT: No activation on output
def forward(self, x):
    x = F.relu(self.linear1(x))
    x = self.linear2(x)  # Full range of outputs
    return x
```

### Pitfall 4: SwiGLU Dimension Mismatch

```python
# ❌ WRONG: Same d_ff as ReLU variant = 50% more params!
swiglu = SwiGLUFeedForward(d_model=512, d_ff=2048)

# ✅ CORRECT: Reduce d_ff to ~2.67× for similar param count
swiglu = SwiGLUFeedForward(d_model=512, d_ff=1365)  # 512 * 8/3
```

---

## Implementation: From Scratch (Standard ReLU)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with ReLU activation."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Standard 4× expansion
        
        self.linear1 = nn.Linear(d_model, d_ff)    # Expand
        self.linear2 = nn.Linear(d_ff, d_model)    # Contract
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            Output: (batch, seq_len, d_model)
        """
        x = self.linear1(x)        # (batch, seq, d_ff)
        x = F.relu(x)              # Non-linearity
        x = self.dropout(x)
        x = self.linear2(x)        # (batch, seq, d_model)
        return self.dropout(x)
```

---

## Implementation: From Scratch (SwiGLU)

```python
class SwiGLUFeedForward(nn.Module):
    """Modern SwiGLU Feed-Forward Network (LLaMA, Mistral, etc.)."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)  # ~2.67× for similar params
            d_ff = ((d_ff + 63) // 64) * 64  # Round to 64 for efficiency
        
        # Three projections (no bias, following LLaMA)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: Swish activation
        gate = F.silu(self.w_gate(x))  # silu = swish = x * sigmoid(x)
        
        # Up path: linear projection
        up = self.w_up(x)
        
        # Gated combination
        hidden = gate * up
        
        # Project back
        return self.dropout(self.w_down(hidden))
```

---

## Implementation: Using PyTorch

PyTorch doesn't have a dedicated FFN class, but it's trivial to compose:

```python
import torch.nn as nn

# Simple FFN using Sequential
ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 512),
    nn.Dropout(0.1)
)

# With GELU (more modern)
ffn_gelu = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 512),
    nn.Dropout(0.1)
)
```

---

## Hands-On Experiment

Run the implementation to compare ReLU and SwiGLU variants:

```bash
python 04_feed_forward/feed_forward.py
```

**What you'll see:**
1. Shape transformation through the network
2. Parameter count comparison (ReLU vs SwiGLU)
3. Proof of position-wise independence

---

## Files in This Module

| File | Description |
|------|-------------|
| `feed_forward.py` | Standard and SwiGLU implementations |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Non-linearity is essential**: Attention is linear (weighted sums). FFN provides the non-linear transformations needed to learn complex functions.

2. **Expand-contract pattern**: d_model → 4×d_model → d_model. The expansion gives "room to think" before compressing.

3. **Position-wise = independent**: Same weights for all positions, but each position processed separately. Token interaction happens in attention, not here.

4. **Most parameters live here**: ~2/3 of a transformer layer's parameters are in the FFN (but attention has quadratic compute).

5. **Activation matters**: ReLU (original) → GELU (BERT) → SwiGLU (LLaMA, modern). Each generation improves ~1-2%.

6. **Gating is powerful**: SwiGLU's multiplicative interaction lets the network learn *what* to let through, not just *how much*.

7. **Together with attention**: Attention = communication between positions. FFN = computation at each position. Both are needed for a complete transformer block.

---

## What's Next?

We now have the two core computations: attention and FFN. But stacking them naively leads to training instability. In **Step 5: Layer Normalization**, we'll learn the stabilization techniques that make deep transformers trainable.

