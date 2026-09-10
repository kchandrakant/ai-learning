# Step 10: Sequence-to-Sequence Models

## The Problem

Map one sequence to another:
- Translation: "Hello" → "Bonjour"
- Summarization: long article → short summary
- Question answering: question → answer

Input and output lengths can differ!

## Encoder-Decoder Architecture

```
Input: "How are you"
        ↓
    [Encoder]
        ↓
  Context Vector
        ↓
    [Decoder]
        ↓
Output: "Comment allez-vous"
```

## The Encoder

Process input sequence, compress into context vector.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(embedded)
        return h_n, c_n  # Context vector
```

## The Decoder

Generate output sequence from context.

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden
```

## Full Seq2Seq Model

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Encode
        hidden = self.encoder(src)
        
        # Decode step by step
        outputs = []
        input_token = trg[:, 0:1]  # <SOS> token
        
        for t in range(1, trg.size(1)):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            
            # Teacher forcing: use ground truth or prediction
            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t:t+1]
            else:
                input_token = output.argmax(2)
        
        return torch.cat(outputs, dim=1)
```

## Teacher Forcing

During training, feed ground truth instead of predictions.

```
With teacher forcing:    Without:
Input: <SOS>            Input: <SOS>
Target: "How"           Pred: "What" (wrong!)
Input: "How"            Input: "What"
Target: "are"           Pred: ??? (error compounds)
```

Start with high teacher forcing, reduce over training.

## The Bottleneck Problem

All information compressed into single context vector!

```
"The quick brown fox jumps over the lazy dog"
                    ↓
            [fixed-size vector]
                    ↓
            Must remember everything!
```

Long sequences → information loss.

**Solution:** Attention mechanism (next steps)

## Inference: Beam Search

Don't just take the best token at each step.

```python
def beam_search(model, src, beam_width=5, max_len=50):
    # Keep top-k candidates at each step
    # Score: sum of log probabilities
    
    # Start with <SOS>
    beams = [(0.0, [SOS_token], hidden)]
    
    for _ in range(max_len):
        candidates = []
        for score, seq, hidden in beams:
            if seq[-1] == EOS_token:
                candidates.append((score, seq, hidden))
                continue
            
            output, new_hidden = model.decoder(seq[-1], hidden)
            probs = F.log_softmax(output, dim=-1)
            
            top_k = probs.topk(beam_width)
            for prob, idx in zip(top_k.values, top_k.indices):
                candidates.append((score + prob, seq + [idx], new_hidden))
        
        # Keep top beam_width
        beams = sorted(candidates, reverse=True)[:beam_width]
    
    return beams[0][1]  # Best sequence
```

## Files

- `seq2seq.py` - Encoder-decoder implementation

## Key Takeaways

1. Encoder compresses input to context
2. Decoder generates output from context
3. Teacher forcing stabilizes training
4. Context bottleneck limits performance
5. Beam search improves inference

## What's Next?

Step 11: **Attention Mechanism** — solving the bottleneck.
