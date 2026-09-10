# Step 1: The Perceptron

## The Birth of Neural Networks

The perceptron (1958) was the first algorithmically described neural network.

## Biological Inspiration

```
Neuron receives inputs → Weighted sum → Fires if above threshold

   x1 ──w1──┐
             ├──→ Σ ──→ threshold ──→ output
   x2 ──w2──┘
```

## The Perceptron Model

```python
import numpy as np

def perceptron(x, w, b):
    """Single perceptron: binary output."""
    z = np.dot(w, x) + b
    return 1 if z > 0 else 0
```

## The Perceptron Learning Algorithm

```python
def train_perceptron(X, y, lr=0.1, epochs=100):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        for i in range(len(X)):
            prediction = 1 if np.dot(w, X[i]) + b > 0 else 0
            error = y[i] - prediction
            
            # Update rule
            w = w + lr * error * X[i]
            b = b + lr * error
    
    return w, b
```

**Update rule:** If wrong, nudge weights toward correct answer.

## What Can a Perceptron Learn?

Linear decision boundaries only:

```
AND gate: ✓ (linearly separable)
OR gate:  ✓ (linearly separable)
XOR gate: ✗ (NOT linearly separable)
```

## The XOR Problem

```
x1  x2  |  XOR
0   0   |   0
0   1   |   1
1   0   |   1
1   1   |   0
```

No single line can separate 0s from 1s!

This limitation nearly killed neural network research in the 1960s.

## The Solution: Multiple Layers

```
XOR = (x1 AND NOT x2) OR (NOT x1 AND x2)

Layer 1: Compute intermediate features
Layer 2: Combine them
```

Multi-layer networks can learn XOR — and much more.

## Files

- `perceptron.py` - Perceptron implementation and visualization

## Key Takeaways

1. Perceptron: weighted sum + threshold
2. Can only learn linearly separable functions
3. XOR proved we need multiple layers
4. Foundation for all neural networks

## What's Next?

Step 2: **Multi-Layer Perceptrons** — stacking layers.
