# Step 14: Positional Encoding

## The Position Problem

Self-attention is permutation-invariant!

```
Attention("cat sat mat") = Attention("mat sat cat")

Order doesn't matter... but it should!
"Dog bites man" ≠ "Man bites dog"
```

## The Solution

Add position information to the input embeddings.

```
Input = Token Embedding + Positional Encoding
```

## Sinusoidal Positional Encoding

Original Transformer uses fixed sinusoidal functions.

```python
def sinusoidal_positional_encoding(max_len, d_model):
    PE = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
    
    PE[:, 0::2] = torch.sin(position * div_term)  # Even indices
    PE[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    
    return PE
```

## The Formula

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

## Why Sinusoids?

1. **Unique encoding:** Each position has distinct pattern
2. **Relative positions:** PE(pos+k) can be expressed as linear function of PE(pos)
3. **Extrapolation:** Works for longer sequences than training

```
sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

Model can learn to compute relative positions!
```

## Learned Positional Embeddings

Alternative: learn position embeddings like word embeddings.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

**Trade-off:**
- Learned: may fit data better, but limited to max training length
- Sinusoidal: generalizes to longer sequences

## Modern Approaches

### Rotary Position Embedding (RoPE)

Used in LLaMA, GPT-NeoX.

```python
# Rotate query and key vectors based on position
# Relative positions encoded in attention scores directly
```

### ALiBi (Attention with Linear Biases)

Used in BLOOM.

```python
# Add linear bias to attention scores based on distance
# Attention(i,j) += -m × |i - j|
# No position encoding added to embeddings!
```

## Using Positional Encoding

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = sinusoidal_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Scale token embeddings
        tok_emb = self.token_embedding(x) * self.scale
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].to(x.device)
        
        return self.dropout(tok_emb + pos_enc)
```

## Visualization

```
Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]
Position 1:  [sin(1), cos(1), sin(1/100), cos(1/100), ...]
Position 2:  [sin(2), cos(2), sin(2/100), cos(2/100), ...]

Low frequencies (slow changes) → capture distant positions
High frequencies (fast changes) → capture nearby positions
```

## Files

- `positional_encoding.py` - Encoding implementations and visualization

## Key Takeaways

1. Self-attention needs position information
2. Sinusoidal: fixed, generalizes to longer sequences
3. Learned: fits training data, limited length
4. RoPE/ALiBi: modern alternatives
5. Add to embeddings before transformer layers

## What's Next?

Step 15: **The Transformer Architecture** — putting it all together.
