# Step 2: Calculus for ML

## Why Calculus?

Training ML models = minimizing a loss function.
Calculus tells us which direction to move to reduce loss.

## Derivatives

The derivative measures rate of change.

```
f(x) = x²
f'(x) = 2x  ← Derivative

At x=3: slope = 2×3 = 6
```

### Interpretation
- f'(x) > 0: function increasing
- f'(x) < 0: function decreasing
- f'(x) = 0: local minimum or maximum

## Common Derivatives

```
f(x) = c       → f'(x) = 0
f(x) = x^n     → f'(x) = n × x^(n-1)
f(x) = e^x     → f'(x) = e^x
f(x) = ln(x)   → f'(x) = 1/x
f(x) = σ(x)    → f'(x) = σ(x)(1 - σ(x))  [sigmoid]
```

## The Chain Rule

For composed functions f(g(x)):

```
d/dx[f(g(x))] = f'(g(x)) × g'(x)

"Derivative of outside × derivative of inside"
```

**Example:**
```
f(x) = (3x + 1)²
     = u² where u = 3x + 1

f'(x) = 2u × 3 = 2(3x + 1) × 3 = 6(3x + 1)
```

**Why it matters:** Backpropagation IS the chain rule applied through the network.

## Partial Derivatives

When f depends on multiple variables, take derivative with respect to one at a time.

```
f(x, y) = x² + xy + y²

∂f/∂x = 2x + y     (treat y as constant)
∂f/∂y = x + 2y     (treat x as constant)
```

## Gradients

The gradient is a vector of all partial derivatives.

```
f(x, y) = x² + y²

∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]
```

**Key property:** The gradient points in the direction of steepest increase.

To minimize: move in the OPPOSITE direction of the gradient.

## Gradient Descent

```python
def gradient_descent(f, grad_f, x_init, learning_rate, n_steps):
    x = x_init
    for _ in range(n_steps):
        gradient = grad_f(x)
        x = x - learning_rate * gradient  # Move opposite to gradient
    return x

# Example: minimize f(x) = x²
# grad_f(x) = 2x
x = gradient_descent(
    f=lambda x: x**2,
    grad_f=lambda x: 2*x,
    x_init=5.0,
    learning_rate=0.1,
    n_steps=50
)
# x ≈ 0 (the minimum)
```

## In Neural Networks

```
Forward pass:  input → prediction
Loss:          L = (prediction - target)²
Backward pass: compute ∂L/∂w for all weights
Update:        w = w - α × ∂L/∂w
```

The chain rule connects the loss back through every layer.

## Files

- `calculus.py` - Gradient descent visualization

## Key Takeaways

1. Derivatives measure rate of change
2. Chain rule: derivative of composed functions
3. Gradient = vector of partial derivatives
4. Gradient points uphill; we go opposite (downhill)
5. Backpropagation = chain rule through the network

## What's Next?

Step 3: **Probability & Statistics** — quantifying uncertainty.
