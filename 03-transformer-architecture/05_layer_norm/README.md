# Step 5: Layer Normalization & Residual Connections

---

## Why This Matters

Without these two techniques, **deep transformers simply cannot train**. The original transformer has 6 layers, but modern models have 32, 80, or even 100+ layers. What makes this possible?

**The two problems:**

```
PROBLEM 1: VANISHING/EXPLODING GRADIENTS
─────────────────────────────────────────
Layer 1 → Layer 2 → Layer 3 → ... → Layer 100 → Output
   ↑         ↑         ↑                           │
   │         │         │         gradient          │
   └─────────┴─────────┴──────  shrinks/grows  ────┘
   
Each layer multiplies gradients. After 100 layers:
  - If multiplier < 1: gradient → 0 (vanishing)
  - If multiplier > 1: gradient → ∞ (exploding)


PROBLEM 2: INTERNAL COVARIATE SHIFT
─────────────────────────────────────────
Layer N receives inputs with a certain distribution.
As earlier layers update, this distribution changes!
Layer N is constantly chasing a moving target.
```

**Residual connections** solve the gradient problem.
**Layer normalization** solves the stability problem.

Together, they're the "infrastructure" that makes deep learning deep.

---

## The Core Intuition

> **Intuition: Residual Connections** — Instead of asking "what should the output be?", ask "what should we ADD to the input?" If nothing needs to change, the layer can just learn to output zero.

> **Intuition: Layer Normalization** — Keep all the activations in a "calm" range. No numbers exploding to millions or shrinking to near-zero. Like automatic volume control for neural networks.

```
WITHOUT Residual + LayerNorm:        WITH Residual + LayerNorm:
                                     
Input ──→ Layer ──→ Layer ──→ Out    Input ──┬──→ Layer ──┬──→ Norm ──→
                                             │            │
  Gradients must survive              └─────────→ + ←─────┘
  ALL layers. Very hard!                    (direct path!)
                                     
                                     Gradients have a "highway" through!
```

---

## Residual Connections: Gradient Highways

### The Formula

```
output = x + Sublayer(x)
```

Instead of learning the transformation `H(x)` directly, learn the **residual** `F(x) = H(x) - x`:

```
Traditional:   H(x) = result you want
               ↓
Residual:      H(x) = x + F(x)
               where F(x) is what you need to ADD
```

### Why This Works

```
GRADIENT FLOW COMPARISON
════════════════════════════════════════════════════════════════════

WITHOUT Residual:
    Layer 1        Layer 2        Layer 3        Loss
      │              │              │              │
      ▼              ▼              ▼              │
    [f₁(x)] ────→ [f₂(·)] ────→ [f₃(·)] ────→   [L]
              ×W₂          ×W₃           ×dL
                                                   
    Gradient = dL × W₃ × W₂ × ... 
    
    If |W| < 1: shrinks exponentially
    If |W| > 1: explodes exponentially


WITH Residual:
    Layer 1        Layer 2        Layer 3        Loss
      │              │              │              │
      ▼              ▼              ▼              │
      x ─────────────────────────────────────────→ dL (direct path!)
      │              │              │              ↑
      +─→[f₁]─→+─→[f₂]─→+─→[f₃]─→+              |
         ↑        ↑        ↑                      |
         └────────┴────────┴──────────────────────┘
                   (also gets gradient through functions)
    
    Gradient = dL × (1 + stuff)
    
    The "1" ensures gradient ALWAYS flows!
```

### Learning the Identity

A powerful insight: If the optimal transformation is **do nothing**, the residual network just needs to learn F(x) = 0:

```
If optimal output = input:
    Traditional: Must learn H(x) = x (hard, requires specific weights)
    Residual: Just learn F(x) = 0 (easy, weights → 0)
```

This makes it safe to add more layers — worst case, they learn to do nothing.

---

## Layer Normalization: Stabilizing Activations

### The Formula

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```

Where:
- μ = mean of x across the feature dimension
- σ² = variance of x across the feature dimension
- γ (gamma) = learnable scale parameter (initialized to 1)
- β (beta) = learnable shift parameter (initialized to 0)
- ε = small constant for numerical stability (1e-5)

### Step-by-Step Example

```
Input x = [1.0, 2.0, 3.0, 4.0]  (one token, 4 features)

Step 1: Compute mean
   μ = (1 + 2 + 3 + 4) / 4 = 2.5

Step 2: Compute variance
   σ² = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
      = (2.25 + 0.25 + 0.25 + 2.25) / 4
      = 1.25

Step 3: Normalize
   x_norm = (x - μ) / √(σ² + ε)
          = ([1,2,3,4] - 2.5) / √1.25
          = [-1.5, -0.5, 0.5, 1.5] / 1.118
          = [-1.342, -0.447, 0.447, 1.342]

Step 4: Scale and shift
   output = γ × x_norm + β
   
   (With γ=1, β=0: output = x_norm)
```

Now the output has mean ≈ 0 and std ≈ 1!

---

## Why Layer Norm, Not Batch Norm?

```
BATCH NORMALIZATION                    LAYER NORMALIZATION
═══════════════════════════════════════════════════════════════════

Normalizes ACROSS batch:               Normalizes ACROSS features:

    Batch                                  Batch
     │                                      │
     ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│ [tok1] [tok2]   │                   │ [tok1] [tok2]   │
│   ↓      ↓      │  Compute          │   →→→    →→→   │  Each token
│   ↓      ↓      │  mean/var         │   →→→    →→→   │  normalized
│   ↓      ↓      │  vertically       │                │  horizontally
│ [tok1] [tok2]   │                   │ [tok1] [tok2]   │
└─────────────────┘                   └─────────────────┘
   sample 2                              sample 2
```

**Why BatchNorm fails for transformers:**

| Issue | BatchNorm | LayerNorm |
|-------|-----------|-----------|
| Variable seq length | ❌ Can't handle | ✓ Works fine |
| Batch size = 1 | ❌ Unreliable stats | ✓ Works fine |
| Inference vs training | ❌ Different behavior | ✓ Same behavior |
| Parallelization | ❌ Needs batch sync | ✓ Fully independent |

---

## ASCII Visualization: The Full Add & Norm Pattern

```
                    TRANSFORMER LAYER WITH ADD & NORM
═════════════════════════════════════════════════════════════════════

                              Input x
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         │                      ▼                      │
         │          ┌───────────────────────┐          │
         │          │  Multi-Head Attention │          │
         │          └───────────┬───────────┘          │
         │                      │                      │
         │                      ▼                      │
         │                  [Dropout]                  │
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                              (+) ← RESIDUAL CONNECTION
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Layer Norm        │
                     │   (mean=0, std=1)   │
                     └──────────┬──────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         │                      ▼                      │
         │          ┌───────────────────────┐          │
         │          │    Feed-Forward Net   │          │
         │          └───────────┬───────────┘          │
         │                      │                      │
         │                      ▼                      │
         │                  [Dropout]                  │
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                              (+) ← RESIDUAL CONNECTION
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Layer Norm        │
                     └──────────┬──────────┘
                                │
                                ▼
                            Output
```

---

## Pre-LN vs Post-LN: Where to Normalize?

### Post-LN (Original Transformer, BERT)

```
output = LayerNorm(x + Sublayer(x))
         ─────────────────────────
         Normalize AFTER the residual
```

### Pre-LN (GPT-2, LLaMA, Modern)

```
output = x + Sublayer(LayerNorm(x))
         ─────────────────────────
         Normalize BEFORE the sublayer
```

**Visual comparison:**

```
POST-LN (Original):                  PRE-LN (Modern):
                                     
   x ────────┐                          x ────────────┐
             │                                        │
             ▼                          ▼             │
         Sublayer                    LayerNorm       │
             │                           │            │
             ▼                           ▼            │
          (+) ← x                    Sublayer        │
             │                           │            │
             ▼                           ▼            │
         LayerNorm                    (+) ←──────────┘
             │                           │
             ▼                           ▼
          output                      output
```

**Why Pre-LN is better for deep models:**

| Aspect | Post-LN | Pre-LN |
|--------|---------|--------|
| Gradient scale | Varies by depth | More consistent |
| Training stability | Needs warmup | More stable |
| Final layer output | Normalized | Unnormalized (often add final LN) |
| Used in | BERT, original transformer | GPT-2, GPT-3, LLaMA |

---

## RMSNorm: The Modern Alternative

LLaMA, Mistral, and other modern models use RMSNorm instead of LayerNorm:

```
LayerNorm:  γ × (x - mean) / std + β    (subtract mean, divide by std)
RMSNorm:    γ × x / rms                  (just divide by RMS)

Where RMS = √(mean(x²))
```

**Why RMSNorm?**
- 10-15% faster (no mean computation)
- No β parameter (fewer weights)
- Empirically same quality

```
LayerNorm:  mean → subtract → variance → normalize → scale → shift
RMSNorm:    rms → normalize → scale

Fewer operations = faster!
```

---

## Comparison: Normalization Techniques

| Technique | Normalizes Over | Parameters | Speed | Used In |
|-----------|-----------------|------------|-------|---------|
| **BatchNorm** | Batch dimension | γ, β, running stats | Fast | CNNs |
| **LayerNorm** | Feature dimension | γ, β | Medium | BERT, GPT-2 |
| **RMSNorm** | Feature dimension | γ only | Faster | LLaMA, Mistral |
| **GroupNorm** | Groups of features | γ, β | Medium | Vision Transformers |

---

## Common Pitfalls & Debugging

### Pitfall 1: Applying Dropout to the Wrong Thing

```python
# ❌ WRONG: Dropout on the input before residual
x = dropout(x)
output = layer_norm(x + sublayer(x))

# ✅ CORRECT: Dropout on sublayer output before addition
output = layer_norm(x + dropout(sublayer(x)))
```

### Pitfall 2: Normalizing Over Wrong Dimension

```python
# ❌ WRONG: Normalizing over batch or sequence
# This would mix information between samples/positions!
mean = x.mean(dim=0)  # Over batch
mean = x.mean(dim=1)  # Over sequence

# ✅ CORRECT: Normalize over feature dimension
mean = x.mean(dim=-1, keepdim=True)  # Over d_model
```

### Pitfall 3: Forgetting the Epsilon

```python
# ❌ WRONG: Division by zero when variance is 0
x_norm = (x - mean) / torch.sqrt(var)

# ✅ CORRECT: Add small epsilon
x_norm = (x - mean) / torch.sqrt(var + 1e-5)
```

### Pitfall 4: Mismatched Pre-LN and Post-LN

```python
# ❌ WRONG: Mixing conventions
# If using Pre-LN, must add final LayerNorm at the end!
class TransformerPreLN(nn.Module):
    def __init__(self):
        self.layers = ...
        # Missing: self.final_norm = LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # Output isn't normalized!

# ✅ CORRECT: Add final norm for Pre-LN
class TransformerPreLN(nn.Module):
    def __init__(self):
        self.layers = ...
        self.final_norm = LayerNorm(d_model)  # Required!
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)  # Normalize at the end
```

---

## Implementation: From Scratch

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization — normalizes across feature dimension."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """RMS Normalization — faster alternative used in LLaMA."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale only
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.gamma * (x / rms)


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization (Post-LN)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_output))
```

---

## Implementation: Using PyTorch

```python
import torch.nn as nn

# PyTorch's LayerNorm
layer_norm = nn.LayerNorm(
    normalized_shape=512,  # d_model
    eps=1e-5,
    elementwise_affine=True  # Include gamma and beta
)

# Example usage in a transformer layer
class TransformerLayerPostLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention with Add & Norm
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

---

## Hands-On Experiment

Run the implementation to see normalization in action:

```bash
python 05_layer_norm/layer_norm.py
```

**What you'll see:**
1. Step-by-step manual LayerNorm computation
2. Effect of γ and β parameters
3. Verification that output has mean≈0, std≈1
4. Comparison with BatchNorm and RMSNorm

---

## Files in This Module

| File | Description |
|------|-------------|
| `layer_norm.py` | LayerNorm and RMSNorm implementations |
| `README.md` | This learning guide |

---

## Key Takeaways

1. **Residual connections create gradient highways**: The `x + f(x)` pattern ensures gradients always have a direct path backward, enabling training of 100+ layer networks

2. **LayerNorm stabilizes activations**: By normalizing to mean=0, std=1, it prevents values from exploding or vanishing through layers

3. **Normalize across features, not batch**: LayerNorm treats each token independently, making it perfect for variable-length sequences (unlike BatchNorm)

4. **Learnable scale and shift**: γ and β let the model learn the optimal output distribution after normalization

5. **Pre-LN vs Post-LN**: Modern models use Pre-LN (normalize before sublayer) for better training stability in deep networks

6. **RMSNorm is faster**: Skipping mean computation gives ~10-15% speedup with no quality loss — used in LLaMA, Mistral

7. **They work together**: Residual connections handle gradient flow, LayerNorm handles activation stability — both are essential for deep transformers

---

## What's Next?

We now have all the building blocks:
- Positional Encoding (position information)
- Multi-Head Attention (token interaction)  
- Feed-Forward Network (per-token processing)
- Layer Normalization & Residuals (training stability)

In **Step 6: Encoder**, we'll combine these into a complete encoder layer — the building block that processes input sequences in models like BERT.

