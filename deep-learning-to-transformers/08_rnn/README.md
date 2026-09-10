# Step 8: Recurrent Neural Networks (RNNs)

## The Challenge of Sequences

CNNs assume fixed-size input. Sequences vary in length.

```
"Hello" → 5 tokens
"How are you doing today?" → 6 tokens
```

We need a model that:
- Handles variable-length input
- Maintains memory of past inputs
- Processes sequentially

## The RNN Idea

Share weights across time steps. Maintain hidden state.

```
x₁ → [RNN] → h₁
      ↑↓
x₂ → [RNN] → h₂
      ↑↓
x₃ → [RNN] → h₃

Same weights at each step!
```

## The RNN Equation

```python
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)

# h_t: hidden state at time t
# x_t: input at time t
# W_xh: input-to-hidden weights
# W_hh: hidden-to-hidden weights
```

## Implementation from Scratch

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros(hidden_size)
    
    def forward(self, x_sequence):
        """x_sequence: (seq_len, input_size)"""
        h = np.zeros(self.W_hh.shape[0])  # Initial hidden state
        hidden_states = []
        
        for x_t in x_sequence:
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b)
            hidden_states.append(h)
        
        return np.array(hidden_states), h  # All states, final state
```

## PyTorch RNN

```python
import torch.nn as nn

rnn = nn.RNN(
    input_size=128,    # Embedding dimension
    hidden_size=256,   # Hidden state size
    num_layers=2,      # Stack RNNs
    batch_first=True   # (batch, seq, features)
)

# Input: (batch, seq_len, input_size)
x = torch.randn(32, 50, 128)
output, h_n = rnn(x)

# output: (32, 50, 256) - all hidden states
# h_n: (2, 32, 256) - final hidden states per layer
```

## Applications

```python
# Many-to-One (classification)
# Use final hidden state
class SentimentRNN(nn.Module):
    def forward(self, x):
        _, h_n = self.rnn(self.embed(x))
        return self.fc(h_n[-1])  # Last layer's final state

# Many-to-Many (sequence labeling)
# Use all hidden states
class POSRNN(nn.Module):
    def forward(self, x):
        output, _ = self.rnn(self.embed(x))
        return self.fc(output)  # Classify each position
```

## Backpropagation Through Time (BPTT)

Unroll the RNN and backprop through all time steps.

```
Loss = Σ L_t

Gradients flow backward through time:
L_T → h_T → h_{T-1} → ... → h_1
```

## The Vanishing Gradient Problem

```
∂L/∂h₁ = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h₂/∂h₁

Many multiplications → gradient vanishes (or explodes)!
```

When gradients vanish:
- RNN forgets long-term dependencies
- "The cat, which ate the fish, was ___" → forgets "cat"

## Files

- `rnn.py` - RNN implementation and examples

## Key Takeaways

1. RNNs share weights across time
2. Hidden state maintains memory
3. BPTT computes gradients through time
4. Vanishing gradients limit memory
5. Need better architectures → LSTM, GRU

## What's Next?

Step 9: **LSTM and GRU** — solving the vanishing gradient.
