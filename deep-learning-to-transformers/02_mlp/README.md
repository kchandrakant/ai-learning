# Step 2: Multi-Layer Perceptrons (MLPs)

## Beyond Single Neurons

Stack layers of neurons to learn complex patterns.

```
Input Layer → Hidden Layer(s) → Output Layer
```

## Architecture

```
    Input      Hidden       Output
    [x1] ──┬── [h1] ──┬── [y1]
           ├── [h2] ──┤
    [x2] ──┴── [h3] ──┴── [y2]
```

Every neuron in one layer connects to every neuron in the next (fully connected).

## Activation Functions

Without activation, stacked linear layers = one linear layer!

```python
# Sigmoid: squashes to (0, 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Tanh: squashes to (-1, 1)
def tanh(z):
    return np.tanh(z)

# ReLU: simple, effective, modern default
def relu(z):
    return np.maximum(0, z)
```

## Why ReLU Dominates

```
Sigmoid/Tanh:  Gradients vanish for large |z|
ReLU:          Gradient is 1 for z > 0, no vanishing

ReLU is also faster to compute!
```

## Forward Pass

```python
def forward(X, W1, b1, W2, b2):
    # Layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Layer 2
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)  # For binary classification
    
    return output
```

## Universal Approximation Theorem

A neural network with one hidden layer can approximate **any continuous function** (given enough neurons).

This is why neural networks are so powerful!

## MLP in PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Solving XOR

```python
# XOR with 2 hidden neurons
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Hidden layer learns:
# h1 ≈ x1 OR x2
# h2 ≈ x1 AND x2
# Output: h1 AND (NOT h2) = XOR
```

## Files

- `mlp.py` - MLP from scratch and with PyTorch

## Key Takeaways

1. MLPs: stack layers of neurons
2. Activation functions add non-linearity
3. ReLU is the modern default
4. One hidden layer can approximate any function
5. Deeper networks learn hierarchical features

## What's Next?

Step 3: **Backpropagation** — how to train MLPs.
