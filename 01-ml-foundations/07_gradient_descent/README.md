# Step 7: Gradient Descent Deep Dive

## The Core Idea

```
θ_new = θ_old - α × ∇L(θ)

α: learning rate (step size)
∇L: gradient of loss with respect to parameters
```

Move parameters in the direction that reduces loss.

## Batch Gradient Descent

Compute gradient using ALL training examples.

```python
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    
    for _ in range(epochs):
        # Gradient over ENTIRE dataset
        gradient = (1/len(X)) * X.T @ (X @ w - y)
        w = w - lr * gradient
    
    return w
```

**Pros:** Stable, converges smoothly
**Cons:** Slow for large datasets, expensive memory

## Stochastic Gradient Descent (SGD)

Compute gradient using ONE example at a time.

```python
def sgd(X, y, lr=0.01, epochs=10):
    w = np.zeros(X.shape[1])
    
    for _ in range(epochs):
        for i in np.random.permutation(len(X)):
            # Gradient from single example
            gradient = X[i] * (X[i] @ w - y[i])
            w = w - lr * gradient
    
    return w
```

**Pros:** Fast updates, can escape local minima
**Cons:** Noisy, doesn't converge smoothly

## Mini-Batch Gradient Descent

Best of both worlds: use batches of B examples.

```python
def mini_batch_gd(X, y, batch_size=32, lr=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    n = len(X)
    
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            gradient = (1/len(X_batch)) * X_batch.T @ (X_batch @ w - y_batch)
            w = w - lr * gradient
    
    return w
```

**Typical batch sizes:** 32, 64, 128, 256

## Learning Rate

```
Too high:  Overshoots, diverges
Too low:   Converges very slowly
Just right: Converges efficiently
```

**Start with:** 0.001 or 0.01, then adjust.

## Momentum

Accelerate in consistent directions, dampen oscillations.

```python
def sgd_momentum(X, y, lr=0.01, momentum=0.9, epochs=100):
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])  # Velocity
    
    for _ in range(epochs):
        gradient = compute_gradient(X, y, w)
        v = momentum * v - lr * gradient  # Update velocity
        w = w + v                          # Update weights
    
    return w
```

## Adam Optimizer

Adaptive learning rates + momentum. The default choice.

```python
def adam(X, y, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=100):
    w = np.zeros(X.shape[1])
    m = np.zeros(X.shape[1])  # First moment
    v = np.zeros(X.shape[1])  # Second moment
    t = 0
    
    for _ in range(epochs):
        t += 1
        gradient = compute_gradient(X, y, w)
        
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        
        m_hat = m / (1 - beta1**t)  # Bias correction
        v_hat = v / (1 - beta2**t)
        
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return w
```

## Optimizer Comparison

| Optimizer | When to Use |
|-----------|-------------|
| SGD | Simple baseline |
| SGD + Momentum | Faster convergence |
| Adam | Default choice for deep learning |
| AdamW | Adam + weight decay (modern default) |

## Files

- `gradient_descent.py` - All variants implemented

## Key Takeaways

1. Batch: stable but slow
2. SGD: fast but noisy
3. Mini-batch: best of both worlds
4. Momentum: accelerates convergence
5. Adam: adaptive, usually just works

## What's Next?

Step 8: **Regularization** — preventing overfitting.
