# Step 8: The Complete Transformer

## Overview

Now we assemble all components into the full transformer architecture from "Attention Is All You Need" (2017).

## Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SOURCE SEQUENCE                      TARGET SEQUENCE           │
│        │                                     │                   │
│        ↓                                     ↓                   │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │   Input      │                    │   Output     │           │
│  │  Embedding   │                    │  Embedding   │           │
│  └──────────────┘                    └──────────────┘           │
│        │                                     │                   │
│        + ←── Positional Encoding             + ←── Pos. Enc.    │
│        │                                     │                   │
│        ↓                                     ↓                   │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │   ENCODER    │                    │   DECODER    │           │
│  │   (N=6)      │───────────────────►│   (N=6)      │           │
│  │              │  encoder_output    │              │           │
│  └──────────────┘                    └──────────────┘           │
│                                              │                   │
│                                              ↓                   │
│                                      ┌──────────────┐           │
│                                      │   Linear     │           │
│                                      │  (to vocab)  │           │
│                                      └──────────────┘           │
│                                              │                   │
│                                              ↓                   │
│                                      ┌──────────────┐           │
│                                      │   Softmax    │           │
│                                      └──────────────┘           │
│                                              │                   │
│                                              ↓                   │
│                                      Output Probabilities        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components Summary

| Component | Purpose | Parameters |
|-----------|---------|------------|
| Input Embedding | Convert tokens to vectors | vocab_size × d_model |
| Output Embedding | Convert tokens to vectors | vocab_size × d_model |
| Positional Encoding | Inject position info | None (fixed) |
| Encoder × N | Process source sequence | ~7M per layer |
| Decoder × N | Generate target sequence | ~10M per layer |
| Output Linear | Project to vocabulary | d_model × vocab_size |

## Hyperparameters (Original Paper)

```python
d_model = 512      # Embedding/hidden dimension
d_ff = 2048        # FFN inner dimension (4 × d_model)
num_heads = 8      # Attention heads
num_layers = 6     # Encoder and decoder layers
dropout = 0.1      # Dropout rate
vocab_size = 37000 # BPE vocabulary (for translation)
```

Total: **65M parameters** (base model)

## Weight Sharing

The original paper shares weights between:
1. Input embedding
2. Output embedding  
3. Pre-softmax linear layer

```python
# Shared embedding matrix
self.embedding = nn.Embedding(vocab_size, d_model)

# Input: self.embedding(src_tokens)
# Output: self.embedding(tgt_tokens)
# Final: logits = hidden @ self.embedding.weight.T
```

This reduces parameters and provides regularization.

## Training

### Loss Function
Cross-entropy with label smoothing (ε = 0.1):
```python
# Instead of hard targets [0, 0, 1, 0, 0]
# Use soft targets [0.025, 0.025, 0.9, 0.025, 0.025]
```

### Optimizer
Adam with custom learning rate schedule:
```python
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```
- Warmup for 4000 steps
- Then decay proportional to inverse square root

### Regularization
- Dropout (0.1) on:
  - Embeddings after positional encoding
  - Each sub-layer output before residual
  - Attention weights
- Label smoothing (0.1)

## Inference

### Greedy Decoding
```python
def greedy_decode(model, src, max_len):
    encoder_output = model.encode(src)
    tgt = [BOS_TOKEN]
    
    for _ in range(max_len):
        logits = model.decode(tgt, encoder_output)
        next_token = logits[-1].argmax()
        tgt.append(next_token)
        if next_token == EOS_TOKEN:
            break
    
    return tgt
```

### Beam Search
Keep top-k candidates at each step for better results.

## Encoder-Only vs Decoder-Only

The original is encoder-decoder for **seq2seq tasks** (translation).

Modern variants:
- **Encoder-only** (BERT): Bidirectional, for understanding tasks
- **Decoder-only** (GPT): Autoregressive, for generation tasks

## Files

- `transformer.py` - Complete implementation
- Examples of training and inference

## Key Takeaways

1. **Encoder**: Processes source → contextual representations
2. **Decoder**: Generates target using encoder output + causal masking
3. **Embeddings**: Input/output often share weights
4. **Positional encoding**: Added to embeddings
5. **Output**: Linear projection + softmax → vocabulary probabilities

## What's Next?

**Beyond** — Modern improvements that power LLaMA, GPT-4, and other state-of-the-art models:
- Pre-Layer Norm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation
- RMSNorm
