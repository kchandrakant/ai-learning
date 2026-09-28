# Step 7: Gradient Descent Deep Dive

## The Workhorse of Machine Learning

Gradient descent is the optimization algorithm that powers almost all modern machine learning. Understanding it deeply will help you debug models, tune hyperparameters, and understand why neural networks work.

---

## The Optimization Problem

**Goal:** Find parameters θ that minimize a loss function L(θ)

```
θ* = argmin L(θ)
        θ
```

**Why not solve analytically?**
- Many functions have no closed-form solution
- Matrix inversion is O(n³) - too slow for big data
- Neural networks have millions of parameters
- Non-linear functions require iterative methods

---

## The Core Algorithm

**Key insight:** The gradient points in the direction of steepest increase.
**Corollary:** Negative gradient points in the direction of steepest decrease.

```
Update rule:
θ_new = θ_old - α × ∇L(θ_old)

Where:
- α = learning rate (step size)
- ∇L = gradient (vector of partial derivatives)
```

**Intuition:** You're blindfolded on a hilly landscape, trying to find the lowest point. Feel the slope (gradient), step downhill (opposite direction).

```
    Loss
      |\
      | \  <- Gradient points uphill
      |  \    so we go opposite direction
      |   \
      |    \_____  <- Want to reach here (minimum)
      |__________ θ
```

---

## Types of Gradient Descent

### Batch Gradient Descent (BGD)

Uses **all** training examples to compute each gradient update:

```
gradient = (1/n) * sum of gradients for all n samples
```

```python
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    
    for _ in range(epochs):
        # Gradient over ENTIRE dataset
        gradient = (1/len(X)) * X.T @ (X @ w - y)
        w = w - lr * gradient
    
    return w
```

**Pros:**
- Stable convergence, smooth path to minimum
- Guaranteed to converge (for convex functions)

**Cons:**
- Very slow for large datasets
- Must load entire dataset into memory
- One update per pass through all data

```
Path to minimum:
    ____
   /    \
  /      \___  <- Smooth, direct path
```

---

### Stochastic Gradient Descent (SGD)

Uses **one** random example per update:

```
gradient ≈ gradient for single random sample
```

```python
def sgd(X, y, lr=0.01, epochs=10):
    w = np.zeros(X.shape[1])
    n = len(X)
    
    for _ in range(epochs):
        for i in np.random.permutation(n):
            # Gradient from SINGLE example
            gradient = X[i] * (X[i] @ w - y[i])
            w = w - lr * gradient
    
    return w
```

**Pros:**
- Very fast updates (n updates per epoch vs 1)
- Can escape local minima due to noise
- Works for online learning (streaming data)
- Lower memory requirements

**Cons:**
- Noisy updates, zigzags toward minimum
- May not converge to exact minimum
- Harder to parallelize

```
Path to minimum:
    ____
   / /\ \
  / /  \/\___  <- Noisy, zigzag path
```

---

### Mini-batch Gradient Descent

Uses **small batches** (typically 32-256 examples) - **the standard approach in practice!**

```
gradient ≈ (1/batch_size) * sum of gradients for batch samples
```

```python
def mini_batch_gd(X, y, batch_size=32, lr=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    n = len(X)
    
    for _ in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            gradient = (1/len(X_batch)) * X_batch.T @ (X_batch @ w - y_batch)
            w = w - lr * gradient
    
    return w
```

**Why mini-batch wins:**
- Batch size 32: 32x more updates than batch GD
- Noise provides regularization (helps generalization)
- Vectorization makes it efficient on GPUs
- Good balance of stability and speed

```
Path to minimum:
    ____
   /    \
  /   ~  \___  <- Slightly noisy but efficient
```

**Typical batch sizes:** 32, 64, 128, 256 (powers of 2 for GPU efficiency)

---

## The Learning Rate: Most Critical Hyperparameter

```
Too small (α = 0.0001):        Just right (α = 0.01):        Too large (α = 1.0):
    |                              |                             |
    |\                             |\                            |  /\  /\
    | \                            | \                           | /  \/  \
    |  \                           |  \____                      |/        -> Diverges!
    |   \                          |                             |
    |    \____                     Converges nicely              Overshoots
    Takes forever
```

### How to Choose Learning Rate

**Start with:** 0.001 or 0.01 (common defaults)

**Signs it's too high:**
- Loss explodes (goes to infinity or NaN)
- Loss oscillates wildly
- Training is unstable

**Signs it's too low:**
- Loss decreases very slowly
- Training takes forever
- Model doesn't improve for many epochs

**Practical approach:**
1. Start with 0.01
2. If loss explodes, divide by 10
3. If loss decreases too slowly, multiply by 3
4. Use learning rate finder (see below)

---

## Learning Rate Schedules

Start with larger learning rate (explore broadly), decrease over time (fine-tune).

### Step Decay

```python
# Reduce by factor every N epochs
lr = initial_lr * (0.1 ** (epoch // 30))
```

### Exponential Decay

```python
lr = initial_lr * exp(-decay_rate * epoch)
```

### Cosine Annealing

```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / total_epochs))
```

### Warm Restarts

Periodically reset learning rate to escape local minima.

```
Learning Rate
  |----
  |    \____
  |         \----
  |              \____
  |_____________________ epochs
       ^          ^
   Restart     Restart
```

---

## Momentum: Accelerating Convergence

**Problem:** SGD oscillates in narrow valleys (ravines):

```
Without momentum:
    |  /\/\/\/\/
    | /          <- Zigzags down slowly
    |/___________
```

**Solution:** Add "velocity" that accumulates past gradients:

```python
v = β * v - α * gradient     # Update velocity
θ = θ + v                    # Update parameters

# β ≈ 0.9 is common (momentum coefficient)
```

**Intuition:** Ball rolling downhill gains speed. Past motion influences current direction.

```python
def sgd_momentum(X, y, lr=0.01, momentum=0.9, epochs=100):
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])  # Velocity
    
    for _ in range(epochs):
        gradient = compute_gradient(X, y, w)
        v = momentum * v - lr * gradient  # Accumulate velocity
        w = w + v                          # Update with velocity
    
    return w
```

```
With momentum:
    |  ___
    | /   \___
    |/        \___  <- Smooth, fast descent
```

**Benefits:**
- Faster convergence (especially in ravines)
- Smooths out noisy gradients
- Can escape small local minima

---

## Advanced Optimizers

### AdaGrad

Adapts learning rate **per parameter** - smaller for frequent features, larger for rare ones:

```python
G = G + gradient²           # Accumulate squared gradients
θ = θ - α / sqrt(G + ε) * gradient
```

**Problem:** Learning rate eventually shrinks to zero (stops learning).

### RMSprop

Fixes AdaGrad with exponential moving average:

```python
G = β * G + (1 - β) * gradient²
θ = θ - α / sqrt(G + ε) * gradient

# β ≈ 0.9
```

### Adam (Adaptive Moment Estimation)

Combines momentum + RMSprop. **The default choice for most applications!**

```python
def adam(X, y, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=100):
    w = np.zeros(X.shape[1])
    m = np.zeros(X.shape[1])  # First moment (momentum)
    v = np.zeros(X.shape[1])  # Second moment (RMSprop)
    t = 0
    
    for _ in range(epochs):
        t += 1
        gradient = compute_gradient(X, y, w)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # Bias correction (important early in training)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return w
```

**Default hyperparameters:** β₁=0.9, β₂=0.999, ε=1e-8

### AdamW

Adam with proper weight decay (decoupled from adaptive learning rate).
**Modern default choice.**

```python
w = w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * w
```

---

## Convergence Criteria: When to Stop

### Fixed Epochs
```python
for epoch in range(100):  # Train for exactly 100 epochs
    train_one_epoch()
```
Simple but arbitrary.

### Early Stopping (Recommended!)
```python
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(1000):
    train_one_epoch()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_weights()  # Save best model
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        restore_best_weights()
        break  # Stop training
```
Prevents overfitting, saves compute.

### Gradient Norm
```python
if np.linalg.norm(gradient) < 1e-6:
    break  # Gradient near zero, at minimum
```

---

## Common Challenges and Solutions

### Local Minima and Saddle Points

```
Local minimum:          Saddle point:
    |\      /|              |    /
    | \    / |              |___/___  <- Gradient = 0 but not minimum
    |  \  /  |              |   \
    |   \/   |              |    \
```

**Solutions:**
- Momentum helps escape both
- Random initialization
- Learning rate warmup
- Modern research: saddle points more common than local minima in high dimensions

### Vanishing/Exploding Gradients

In deep networks, gradients can shrink to ~0 or explode to infinity.

```
Layer 100 <- Layer 99 <- ... <- Layer 1 <- Input

If each layer multiplies gradient by 0.9:
Final gradient = 0.9^100 ≈ 0 (vanished!)

If each layer multiplies gradient by 1.1:
Final gradient = 1.1^100 ≈ 13,781 (exploded!)
```

**Solutions:**
- Careful weight initialization (Xavier, He)
- Batch normalization
- Residual connections (skip connections)
- Gradient clipping

### Gradient Clipping

Prevent exploding gradients:

```python
max_norm = 1.0
grad_norm = np.linalg.norm(gradient)
if grad_norm > max_norm:
    gradient = gradient * max_norm / grad_norm
```

---

## Practical Tips

### 1. Learning Rate Finder

```python
# Increase LR exponentially, plot loss
lrs = np.logspace(-6, 0, 100)  # 1e-6 to 1
losses = []
for lr in lrs:
    loss = train_one_batch(lr)
    losses.append(loss)

# Pick LR just before loss explodes
plt.semilogx(lrs, losses)
```

### 2. Monitor Training Curves

```
Good training:          Overfitting:           LR too high:
  |\                      |\    /val             |/\/\/\
  | \___train             | \__/                 |      \__
  |     \___val           |    \train            |
```

### 3. Optimizer Selection Guide

| Optimizer | Best For | Default LR |
|-----------|----------|------------|
| SGD | Simple problems, fine-tuning | 0.01-0.1 |
| SGD+Momentum | General deep learning | 0.01-0.1 |
| Adam | Default choice, most tasks | 0.001 |
| AdamW | Transformers, modern networks | 0.001 |

### 4. Start with Adam, Switch to SGD for Fine-tuning

Adam converges fast but may generalize worse than SGD+momentum for final fine-tuning.

---

## Files

- `gradient_descent.py` - All variants implemented with visualization

## Key Takeaways

1. **Gradient descent** = iteratively step opposite to gradient
2. **Mini-batch** is the practical standard (not full batch or single sample)
3. **Learning rate** is the most important hyperparameter to tune
4. **Momentum** accelerates convergence and smooths oscillations
5. **Adam** is the safe default optimizer
6. **Early stopping** prevents overfitting and saves compute
7. **Monitor your loss curves** - they tell you everything about training

## What's Next?

Step 8: **Regularization** - preventing overfitting and building models that generalize.
