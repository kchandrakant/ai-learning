# Step 9: LSTM and GRU

## The Long-Term Dependency Problem

RNNs struggle to connect information across many steps.

```
"The cat, which had been sitting on the mat for hours, finally ___"

RNN by step 12: "What cat? 🤷"
```

## LSTM: Long Short-Term Memory

Add a separate cell state and gates to control information flow.

```
        ┌─────────────────────────────────────┐
        │           Cell State (C)            │
        └──↑────────────↑──────────────↑──────┘
           │ forget     │ input       │ 
           │ gate       │ gate        │ output gate
        ┌──┴───┐    ┌───┴───┐     ┌───┴───┐
        │  ×   │    │   +   │     │   ×   │
        └──────┘    └───────┘     └───────┘
              ↑           ↑             ↑
              f_t        i_t × C̃_t      o_t × tanh(C_t)
```

## LSTM Equations

```python
# Forget gate: what to discard from cell state
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)

# Input gate: what new info to store
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)

# Update cell state
C_t = f_t * C_{t-1} + i_t * C̃_t

# Output gate: what to output
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

## Why LSTM Works

```
Cell state C flows with minimal transformation:
C_t = f_t * C_{t-1} + ...

If f_t ≈ 1: C_{t-1} flows through unchanged!
Gradients can flow through many steps.
```

## PyTorch LSTM

```python
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2,         # Between layers
    bidirectional=True   # Optional
)

x = torch.randn(32, 50, 128)
output, (h_n, c_n) = lstm(x)

# Bidirectional: hidden_size doubles
# output: (32, 50, 512) if bidirectional
```

## GRU: Gated Recurrent Unit

Simplified LSTM with fewer parameters.

```python
# Reset gate: how much past to forget
r_t = sigmoid(W_r @ [h_{t-1}, x_t])

# Update gate: balance old vs new
z_t = sigmoid(W_z @ [h_{t-1}, x_t])

# Candidate hidden state
h̃_t = tanh(W @ [r_t * h_{t-1}, x_t])

# New hidden state
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

**Differences from LSTM:**
- No separate cell state
- Combines forget and input gates into update gate
- Fewer parameters, often similar performance

## PyTorch GRU

```python
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True
)

output, h_n = gru(x)  # No cell state
```

## LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More | Fewer |
| Training speed | Slower | Faster |
| Long sequences | Better | Good |
| When to use | Default | Limited compute |

## Bidirectional RNNs

Process sequence in both directions.

```
Forward:  h₁ → h₂ → h₃
Backward: h₁ ← h₂ ← h₃

Concat: [h_forward, h_backward]
```

```python
bilstm = nn.LSTM(128, 256, bidirectional=True)
# Output hidden size: 256 × 2 = 512
```

## Files

- `lstm_gru.py` - LSTM and GRU implementations

## Key Takeaways

1. LSTM adds cell state and gates
2. Forget gate allows gradient flow
3. GRU is simpler, often works well
4. Bidirectional sees future context
5. Both solve vanishing gradients (mostly)

## What's Next?

Step 10: **Sequence-to-Sequence** — encoder-decoder models.
