# Step 15: The Transformer Architecture

## "Attention Is All You Need"

The 2017 paper that changed NLP (and eventually everything).

## Full Architecture

```
                 Outputs
                    ↑
            ┌──────────────┐
            │   Linear +   │
            │   Softmax    │
            └──────────────┘
                    ↑
         ┌─────────────────────┐
         │                     │
         │  Decoder Stack (Nx) │
         │                     │
         └─────────────────────┘
           ↑              ↑
    ┌──────┴──────┐       │
    │             │       │
    │   Encoder   │───────┘
    │   Stack     │  (cross-attention)
    │   (Nx)      │
    │             │
    └─────────────┘
           ↑
    Input Embeddings
    + Positional Enc
           ↑
        Inputs
```

## Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## Full Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = TransformerEmbedding(src_vocab, d_model, max_len)
        self.decoder_embedding = TransformerEmbedding(tgt_vocab, d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        enc_output = self.encoder_embedding(src)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decode
        dec_output = self.decoder_embedding(tgt)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        return self.fc_out(dec_output)
```

## Key Components Summary

| Component | Purpose |
|-----------|---------|
| Multi-Head Attention | Learn different relationship types |
| Positional Encoding | Inject position information |
| Layer Normalization | Stabilize training |
| Residual Connections | Enable gradient flow |
| Feed-Forward | Per-position transformation |

## The Original Hyperparameters

```
d_model = 512      # Embedding dimension
num_heads = 8      # Attention heads
num_layers = 6     # Encoder/decoder layers
d_ff = 2048        # Feed-forward hidden dim
dropout = 0.1      # Dropout rate
```

## Variants

### Encoder-only (BERT)
Classification, NLU tasks.

### Decoder-only (GPT)
Text generation, language modeling.

### Encoder-Decoder (T5, BART)
Translation, summarization.

## What's Next?

**Continue to the transformer-architecture course** for:
- Detailed positional encoding math
- Attention visualization
- Building transformers from scratch
- Modern variations (RoPE, Flash Attention, etc.)

## Files

- `transformer.py` - Complete transformer implementation

## Congratulations!

You've completed the journey from perceptron to transformer.

You now understand:
- How neural networks learn (backprop)
- How to process sequences (RNN → LSTM → Attention)
- Why transformers dominate (parallelism + attention)

**Next course:** transformer-architecture for deep implementation details.
