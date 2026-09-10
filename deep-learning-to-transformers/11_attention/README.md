# Step 11: Attention Mechanism

## The Bottleneck Problem

Seq2Seq compresses entire input into one vector.

```
"The quick brown fox jumps over the lazy dog"
                    ↓
            [single vector]  ← Information loss!
```

## The Attention Idea

Let the decoder look at ALL encoder states, focusing on relevant ones.

```
Decoder at step t:
"Which source words matter for generating this target word?"

Output "chat" → attend to "cat"
Output "brun" → attend to "brown"
```

## Attention as Soft Lookup

```
Query:  "What am I looking for?" (decoder state)
Keys:   "What do I have?" (encoder states)
Values: "What to return?" (encoder states)

Attention = softmax(Query · Keys) × Values
```

## Bahdanau Attention (Additive)

```python
def bahdanau_attention(query, keys, values):
    # query: (batch, hidden)
    # keys/values: (batch, seq_len, hidden)
    
    # Score function: v^T tanh(W_q @ query + W_k @ keys)
    query_expanded = query.unsqueeze(1)  # (batch, 1, hidden)
    scores = self.v(torch.tanh(self.W_q(query_expanded) + self.W_k(keys)))
    scores = scores.squeeze(-1)  # (batch, seq_len)
    
    weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
    context = torch.bmm(weights.unsqueeze(1), values)  # (batch, 1, hidden)
    
    return context.squeeze(1), weights
```

## Luong Attention (Multiplicative)

Simpler and faster.

```python
def luong_attention(query, keys, values):
    # Score function: query · keys^T
    scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
    scores = scores.squeeze(1)  # (batch, seq_len)
    
    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
    
    return context, weights
```

## Seq2Seq with Attention

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = LuongAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        
        # Compute attention
        context, weights = self.attention(hidden[0][-1], encoder_outputs, encoder_outputs)
        
        # Concat context with input
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Concat output with context for prediction
        combined = torch.cat([output.squeeze(1), context], dim=-1)
        prediction = self.fc(combined)
        
        return prediction, hidden, weights
```

## Visualizing Attention

```
Source: "The cat sat on the mat"
Target: "Le chat était assis sur le tapis"

            The   cat   sat   on   the   mat
Le         0.9   0.05  0.02  0.01  0.01  0.01
chat       0.1   0.8   0.05  0.02  0.02  0.01
était      0.02  0.1   0.7   0.1   0.05  0.03
...
```

Attention weights show alignment!

## Benefits of Attention

1. **No bottleneck:** Access all encoder states
2. **Interpretability:** See what model focuses on
3. **Better gradients:** Direct path from output to input
4. **Handles long sequences:** Focus on relevant parts

## Files

- `attention.py` - Attention implementations

## Key Takeaways

1. Attention computes weighted sum of encoder states
2. Query-Key-Value formulation
3. Softmax gives attention weights
4. Solves the bottleneck problem
5. Foundation for transformers

## What's Next?

Step 12: **Self-Attention** — attending to yourself.
