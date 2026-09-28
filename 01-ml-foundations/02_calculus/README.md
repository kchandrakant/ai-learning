# Step 2: Calculus for ML

## Why Calculus?

Training ML models = minimizing a loss function.
Calculus tells us **which direction to move** to reduce loss.

Without calculus, we'd have to try random parameter changes and hope for the best. With calculus, we can efficiently find the optimal parameters.

---

## Derivatives: The Core Concept

The derivative measures the **rate of change** - how much the output changes when you slightly change the input.

```
f(x) = x²
f'(x) = 2x  <- Derivative

At x=3: slope = 2×3 = 6
"If I increase x by a tiny amount, f(x) increases 6 times as much"
```

### Visual Intuition

```
f(x)
  |        /
  |      /
  |    /  <- Steep slope = large derivative
  |  /
  |/________ x

f(x)
  |   ___
  |  /   \
  | /     \  <- Flat at top = derivative is 0 (minimum/maximum!)
  |/       \
  |__________ x
```

### Interpretation

- f'(x) > 0: function is increasing (going uphill)
- f'(x) < 0: function is decreasing (going downhill)
- f'(x) = 0: flat point - could be minimum, maximum, or saddle point

---

## Common Derivatives You'll See

```
f(x) = c         ->  f'(x) = 0              (constants don't change)
f(x) = x^n       ->  f'(x) = n × x^(n-1)    (power rule)
f(x) = e^x       ->  f'(x) = e^x            (exponential is its own derivative!)
f(x) = ln(x)     ->  f'(x) = 1/x            (log rule)
f(x) = sigmoid(x) -> f'(x) = sigmoid(x)(1 - sigmoid(x))  (important for neural nets)
f(x) = ReLU(x)   ->  f'(x) = 1 if x > 0, else 0  (piecewise)
```

---

## The Chain Rule: Backpropagation's Secret

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

### Why the Chain Rule is Everything in Deep Learning

A neural network is just composed functions:

```
output = f3(f2(f1(input)))

Layer 1: z1 = W1 @ x + b1,  a1 = relu(z1)
Layer 2: z2 = W2 @ a1 + b2, a2 = relu(z2)
Layer 3: z3 = W3 @ a2 + b3, output = softmax(z3)
```

To find how loss changes with W1, chain rule:
```
dL/dW1 = dL/doutput × doutput/da2 × da2/dz2 × dz2/da1 × da1/dz1 × dz1/dW1
```

**This IS backpropagation!** Chain rule applied layer by layer, backwards.

---

## Partial Derivatives

When f depends on multiple variables, take derivative with respect to one at a time, treating others as constants.

```
f(x, y) = x² + xy + y²

df/dx = 2x + y     (treat y as constant)
df/dy = x + 2y     (treat x as constant)
```

**In ML context:** Your loss depends on thousands of weights. Partial derivative tells you how loss changes if you tweak just ONE weight.

---

## The Gradient: All Partial Derivatives Together

The gradient is a vector of all partial derivatives:

```
f(x, y) = x² + y²

gradient f = [df/dx, df/dy] = [2x, 2y]
```

### The Key Property

**The gradient points in the direction of steepest increase.**

```
        Gradient
           ^
          /|\
         / | \
        /  |  \
       (points uphill)
```

To **minimize** a function: move in the **opposite** direction of the gradient!

---

## Gradient Descent: Putting It All Together

```python
def gradient_descent(f, grad_f, x_init, learning_rate, n_steps):
    x = x_init
    for _ in range(n_steps):
        gradient = grad_f(x)
        x = x - learning_rate * gradient  # Move OPPOSITE to gradient
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
# x converges to 0 (the minimum)
```

### Visual Intuition

```
Loss
  |\
  | \
  |  \  You are here
  |   \    |
  |    \   v
  |     \____  <- Want to get here
  |____________ parameters

Gradient tells you: "Go left to decrease loss"
```

---

## Learning Rate: How Big a Step?

```
Too small (lr = 0.0001):       Too large (lr = 10):
    |                              |
    |\                             |  /\  /\
    | \                            | /  \/  \
    |  \                           |/        -> Diverges!
    |   \____                      |
    Takes forever                  Overshoots and explodes

Just right (lr = 0.01):
    |
    |\
    | \
    |  \____  <- Converges nicely
    |
```

---

## Calculus in Neural Network Training

```
Forward pass:  input -> hidden layers -> prediction
Loss:          L = (prediction - target)²
Backward pass: compute dL/dw for ALL weights using chain rule
Update:        w = w - learning_rate × dL/dw
```

### Concrete Example

```python
# Simple network: input -> hidden -> output
# Forward pass
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
y_pred = z2
loss = (y_pred - y_true)²

# Backward pass (chain rule in reverse)
dL_dy_pred = 2(y_pred - y_true)
dL_dz2 = dL_dy_pred
dL_dW2 = dL_dz2 @ a1.T
dL_da1 = W2.T @ dL_dz2
dL_dz1 = dL_da1 * relu_derivative(z1)
dL_dW1 = dL_dz1 @ x.T

# Update
W1 = W1 - lr * dL_dW1
W2 = W2 - lr * dL_dW2
```

---

## Numerical vs Analytical Gradients

**Analytical:** Use calculus rules (fast, exact)
```python
# f(x) = x², f'(x) = 2x
gradient = 2 * x
```

**Numerical:** Approximate using finite differences (slow, approximate)
```python
# f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
h = 1e-5
gradient = (f(x + h) - f(x - h)) / (2 * h)
```

**Use numerical gradients to verify your analytical gradients are correct!** (Gradient checking)

---

## Key Calculus Concepts Summary

| Concept | What It Tells You | ML Application |
|---------|-------------------|----------------|
| Derivative | Rate of change | How loss changes with one parameter |
| Chain Rule | Derivative of composition | Backpropagation through layers |
| Partial Derivative | Change w.r.t. one variable | Gradient for one weight |
| Gradient | Direction of steepest ascent | Which way to NOT go (we minimize) |

---

## Files

- `calculus.py` - Gradient descent visualization and numerical verification

## Key Takeaways

1. **Derivatives** measure rate of change
2. **Chain rule** = derivative of composed functions = backpropagation
3. **Gradient** = vector of partial derivatives, points uphill
4. **Gradient descent**: move opposite to gradient to minimize loss
5. **Learning rate** controls step size - crucial hyperparameter
6. Every modern deep learning framework does this automatically (autograd)

## What's Next?

Step 3: **Probability & Statistics** - quantifying uncertainty in predictions.
