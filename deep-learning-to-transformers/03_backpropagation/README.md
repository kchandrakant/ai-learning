# Step 3: Backpropagation

## The Key to Training Neural Networks

How do we compute gradients through multiple layers?

**Answer:** The chain rule, applied systematically.

## The Chain Rule

```
If y = f(g(x)), then dy/dx = dy/dg × dg/dx

"Derivative of outside × derivative of inside"
```

## Forward Pass

Compute outputs layer by layer:

```python
# Input → Hidden → Output
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)
loss = binary_cross_entropy(y_true, y_pred)
```

## Backward Pass

Compute gradients in reverse order:

```python
# Start from loss, work backward
dL_dy = y_pred - y_true  # Gradient of loss w.r.t. output

# Layer 2
dL_dz2 = dL_dy * sigmoid_derivative(z2)
dL_dW2 = a1.T @ dL_dz2
dL_db2 = dL_dz2.sum(axis=0)
dL_da1 = dL_dz2 @ W2.T

# Layer 1
dL_dz1 = dL_da1 * relu_derivative(z1)
dL_dW1 = X.T @ dL_dz1
dL_db1 = dL_dz1.sum(axis=0)
```

## Computational Graph

```
X ──→ [×W1+b1] ──→ z1 ──→ [ReLU] ──→ a1 ──→ [×W2+b2] ──→ z2 ──→ [σ] ──→ ŷ ──→ [Loss] ──→ L
                                                                                          ↓
← dL/dW1 ← dL/dz1 ← dL/da1 ← dL/dz2 ← dL/dŷ ←────────────────────────────────────────────
```

## Full Implementation

```python
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_pred = 1 / (1 + np.exp(-self.z2))  # Sigmoid
        return self.y_pred
    
    def backward(self, X, y_true, lr=0.01):
        m = len(X)
        
        # Output layer
        dz2 = self.y_pred - y_true
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0)
        
        # Hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0)
        
        # Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
```

## PyTorch: Automatic Differentiation

PyTorch computes gradients automatically!

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()  # Compute gradient
print(x.grad)  # dy/dx = 2x + 3 = 7
```

## Training Loop

```python
for epoch in range(epochs):
    # Forward
    y_pred = model.forward(X)
    loss = compute_loss(y_pred, y_true)
    
    # Backward
    model.backward(X, y_true, lr)
    
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Files

- `backpropagation.py` - Manual backprop implementation

## Key Takeaways

1. Backprop = chain rule through the network
2. Forward: compute outputs
3. Backward: compute gradients in reverse
4. Each layer passes gradients to the previous
5. PyTorch automates this with autograd

## What's Next?

Step 4: **Training Deep Networks** — making it work in practice.
