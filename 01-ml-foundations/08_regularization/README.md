# Step 8: Regularization & Generalization

## The Central Challenge of Machine Learning

The goal isn't to fit training data perfectly - it's to **generalize** to new, unseen data. Regularization is how we achieve this.

---

## The Overfitting Problem

**Overfitting:** Model memorizes training data instead of learning generalizable patterns.

```
Training Data:         Overfit Model:         Good Model:
    *   *                  /\  /\               ___
  *   *   *              */  \/  \*           */   \*
    *   *              */        \*          /     \
                      Memorizes every       Captures the
                      bump and noise        underlying trend
```

### Symptoms of Overfitting

```
Training accuracy: 99%
Validation accuracy: 65%  <- Big gap = overfitting!
```

```
Loss over epochs:

Loss
  |
  |\___train
  |    \________
  |  ____/val     <- Validation loss increases
  | /                while training loss decreases
  |________________ epochs
        ^
    Overfitting starts here
```

### Why Does Overfitting Happen?

**Model capacity > Data complexity**

| Too Few Parameters | Just Right | Too Many Parameters |
|-------------------|------------|---------------------|
| Underfits | Good generalization | Overfits |
| High bias | Balance | High variance |
| Can't capture patterns | Captures patterns | Captures noise too |

**Risk factors:**
- Too many features relative to samples
- Complex model (deep network, high-degree polynomial)
- Training too long
- Noisy data
- Small dataset

---

## L2 Regularization (Ridge)

**Idea:** Penalize large weights by adding sum of squared weights to loss.

```
L_regularized = L_original + λ × Σwᵢ²
              = L_original + λ × ||w||²
```

```python
from sklearn.linear_model import Ridge

# alpha = λ (regularization strength)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

### Effect on Weights

```
Without L2:              With L2:
w = [5.2, -3.8, 0.1]    w = [1.2, -0.9, 0.05]
Large, varied weights    Smaller, more uniform
```

### Why It Works

- Large weights → model relies heavily on specific features
- Small weights → model uses all features more evenly
- Prevents any single feature from dominating
- Smoother decision boundaries

### Geometric Intuition

```
         w₂
          |    
          |  * <- Original optimum (large weights)
          | /|
          |/ | <- L2 pulls toward origin
          +--*---- w₁
         /
      L2 constraint is a circle
      Optimization finds where loss contours
      touch the circle
```

### Gradient with L2

```
gradient_regularized = gradient_original + 2λw

This "pulls" weights toward zero at each update
```

---

## L1 Regularization (Lasso)

**Idea:** Penalize absolute values of weights.

```
L_regularized = L_original + λ × Σ|wᵢ|
              = L_original + λ × ||w||₁
```

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
```

### The Key Difference: Sparsity!

```
Without L1:              With L1:
w = [5.2, -3.8, 0.1]    w = [1.5, 0, 0]
All features used        Sparse! Only important features
```

L1 can push weights to **exactly zero** - automatic feature selection!

### Why L1 Creates Zeros

**Geometric intuition:**

```
         w₂
          |    
          |  * <- Original optimum
          |\ 
          | \  <- L1 pulls to corner (an axis)
          +--*---- w₁
         /|
      L1 constraint is a diamond
      Corners are on axes (one weight = 0)
```

The diamond has "corners" on the axes. When optimization finds the constraint, it often hits a corner where one or more weights are exactly zero.

---

## L1 vs L2: When to Use Which

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Weight effect** | Sparse (many exactly zero) | Small but non-zero |
| **Feature selection** | Yes (automatic) | No |
| **Correlated features** | Picks one arbitrarily | Keeps all, shrinks together |
| **Computation** | Slightly harder (not differentiable at 0) | Easier (smooth) |
| **Best when** | Many irrelevant features | All features somewhat useful |
| **Interpretability** | Higher (fewer features) | Lower (all features kept) |

### Elastic Net: Best of Both

```python
from sklearn.linear_model import ElasticNet

# l1_ratio: 0 = pure L2, 1 = pure L1, 0.5 = half and half
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

```
L_elastic = L_original + λ₁||w||₁ + λ₂||w||²
```

Good for when you have many features, some irrelevant, some correlated.

---

## Choosing λ (Regularization Strength)

```
λ too small:              λ just right:           λ too large:
Still overfits           Good generalization      Underfits
    /\  /\                    ___                    ___
   /  \/  \                  /   \                  -----
  Captures noise           Captures trend          Too simple
```

### How to Choose λ

**Use cross-validation:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
```

**Visualize the effect:**

```
Error
  |  ___train___
  | /           \val
  |/             \___/
  |__________________ log(λ)
          ^
      Sweet spot
```

---

## Early Stopping

**The simplest and most effective regularization!**

**Idea:** Stop training before the model starts overfitting.

```
Loss
  |
  |\___train
  |    \________
  |____/val
  |   ^
  |   STOP HERE! (before validation loss increases)
  |________________ epochs
```

### Implementation

```python
best_val_loss = float('inf')
patience = 10  # How many epochs to wait
patience_counter = 0
best_weights = None

for epoch in range(1000):
    train_one_epoch()
    val_loss = evaluate(X_val, y_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.get_weights()  # Save best model
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        model.set_weights(best_weights)  # Restore best
        break
```

### Why It Works

- Early in training: model learns signal (true patterns)
- Later in training: model starts memorizing noise
- Stop at the transition point

**Benefits:**
- Free! No hyperparameters to tune (besides patience)
- Saves compute time
- Works with any model

---

## Dropout (for Neural Networks)

**Idea:** Randomly "turn off" neurons during training.

```
Without Dropout:           With Dropout (p=0.5):
    O---O---O                 O---X---O
    |\ /|\ /|                 |   |   |
    | X | X |                 |   X   |  (X = dropped)
    |/ \|/ \|                 |   |   |
    O---O---O                 O---O---X
```

### Implementation

```python
# During training:
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, x.shape)
    return x * mask / (1-p)  # Scale to maintain expected value

# During inference:
# Use all neurons, no dropout
```

### Why Dropout Works

1. **Prevents co-adaptation:** Neurons can't rely on specific other neurons
2. **Forces redundancy:** Network learns multiple paths for same information
3. **Ensemble effect:** Like training many different networks, then averaging

**Typical dropout rates:**
- Input layer: 0.1-0.2 (light dropout)
- Hidden layers: 0.3-0.5 (heavier dropout)
- Never on output layer

---

## Data Augmentation

**Idea:** Create more training data through transformations.

### For Images

```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

```
Original    Flip      Rotate    Crop      Color
  [img]  →  [flip]  →  [rot]  → [crop]  → [color]
   cat     tac       cat       ca        cat
          (mirror)  (tilted)  (zoomed)  (darker)
```

### For Text

- Synonym replacement: "happy" → "joyful"
- Random insertion/deletion of words
- Back-translation: English → French → English

### For Tabular Data

- SMOTE (Synthetic Minority Oversampling)
- Adding Gaussian noise to features
- Mixup: blend examples and labels

### Why It Works

- More data = less overfitting
- Teaches invariances (cat is still cat when flipped)
- Cheap way to increase effective dataset size 10-100x

---

## Batch Normalization

**Idea:** Normalize activations within each mini-batch.

```python
# For each layer:
# 1. Compute mean and variance of batch
mu = batch.mean(axis=0)
var = batch.var(axis=0)

# 2. Normalize
x_norm = (x - mu) / sqrt(var + eps)

# 3. Scale and shift (learned parameters)
y = gamma * x_norm + beta
```

### Why It Helps

1. **Reduces internal covariate shift:** Each layer sees more stable inputs
2. **Regularization effect:** Noise from batch statistics acts like dropout
3. **Allows higher learning rates:** More stable optimization
4. **Smoother loss landscape:** Easier to optimize

**Important:** Different behavior during training vs inference (use running averages).

---

## Other Regularization Techniques

### Weight Decay

Equivalent to L2 but applied in optimizer (slightly different for Adam):

```python
# In optimizer
w = w - lr * gradient - lr * weight_decay * w
```

### Label Smoothing

Soften hard labels to prevent overconfidence:

```python
# Instead of [0, 1, 0] for class 1
# Use [0.05, 0.9, 0.05]
smooth_labels = (1 - smoothing) * hard_labels + smoothing / n_classes
```

### Mixup

Blend training examples:

```python
lambda_ = np.random.beta(alpha, alpha)
x_mixed = lambda_ * x1 + (1 - lambda_) * x2
y_mixed = lambda_ * y1 + (1 - lambda_) * y2
```

### Noise Injection

Add noise to inputs, weights, or gradients during training.

---

## Regularization Strategy

**Start simple, add regularization as needed:**

```
1. Train without regularization
   |
   v
2. If overfitting (train >> val performance):
   - Add L2 regularization (start small, increase)
   - Use dropout (for neural networks)
   - Use early stopping
   - Get more data / use data augmentation
   |
   v
3. If still overfitting:
   - Reduce model complexity
   - Increase regularization strength
   - Combine multiple techniques
   |
   v
4. If underfitting (poor train performance):
   - REDUCE regularization
   - INCREASE model complexity
   - Train longer
```

---

## Comparison Summary

| Technique | Type | Hyperparameters | When to Use |
|-----------|------|-----------------|-------------|
| L1 (Lasso) | Weight penalty | λ | Feature selection needed |
| L2 (Ridge) | Weight penalty | λ | Default choice |
| Elastic Net | Weight penalty | λ₁, λ₂, ratio | Many features, some correlated |
| Dropout | Architecture | drop rate p | Deep neural networks |
| Early stopping | Training | patience | Always! It's free |
| Data augmentation | Data | augmentation types | Limited training data |
| Batch normalization | Architecture | momentum, ε | Deep networks |
| Weight decay | Optimizer | decay rate | Modern optimizers (AdamW) |

---

## Files

- `regularization.py` - Implementation and comparison of techniques

## Key Takeaways

1. **Regularization = controlled underfitting** to improve generalization
2. **L2 (Ridge)** shrinks all weights; **L1 (Lasso)** zeros some out
3. **λ controls strength** - too much causes underfitting, too little allows overfitting
4. **Early stopping** is free regularization - always use it
5. **Dropout** prevents neuron co-adaptation in deep networks
6. **Data augmentation** is often the most effective regularization
7. **Combine techniques** for best results
8. **If underfitting, reduce regularization** - don't just add more!

## What's Next?

Step 9: **Decision Trees & Ensembles** - tree-based methods including Random Forests and Gradient Boosting.
