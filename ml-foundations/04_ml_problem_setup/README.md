# Step 4: The ML Problem Setup

## Types of Learning

### Supervised Learning
Given inputs AND outputs, learn the mapping.

```
Training: (X, y) pairs
Goal: Learn f such that f(X) ≈ y

Examples:
- Image → Label (classification)
- Features → Price (regression)
```

### Unsupervised Learning
Only inputs, find structure.

```
Training: X only
Goal: Find patterns, clusters, representations

Examples:
- Customer segmentation
- Dimensionality reduction
- Anomaly detection
```

### Reinforcement Learning
Learn through interaction and rewards.

```
Agent takes action → Environment responds → Reward
Goal: Maximize cumulative reward

Examples:
- Game playing
- Robotics
```

## Data Terminology

```python
# Features (X): inputs, predictors
# Labels (y): outputs, targets
# Example: one (X, y) pair
# Dataset: collection of examples

# Shape conventions:
X.shape = (n_samples, n_features)  # e.g., (1000, 10)
y.shape = (n_samples,)             # e.g., (1000,)
```

## Train / Validation / Test Split

```
All Data
├── Training Set (60-80%)    → Train model
├── Validation Set (10-20%)  → Tune hyperparameters
└── Test Set (10-20%)        → Final evaluation (touch once!)
```

```python
from sklearn.model_selection import train_test_split

# Split off test set first
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Then split train/validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# 0.25 of 0.8 = 0.2, so we get 60/20/20 split
```

## Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

Bias:      Error from wrong assumptions (underfitting)
Variance:  Error from sensitivity to training data (overfitting)
```

```
High Bias:     Model too simple, misses patterns
High Variance: Model too complex, memorizes noise

Goal: Find the sweet spot
```

## Underfitting vs Overfitting

```
Training Error vs Validation Error:

Underfitting:  High train error, high val error
Just right:    Low train error, low val error
Overfitting:   Low train error, HIGH val error ← Danger!
```

## Cross-Validation

More reliable than single split:

```python
from sklearn.model_selection import cross_val_score

# K-fold: split into K parts, train K times
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Files

- `ml_problem_setup.py` - Data splitting examples

## Key Takeaways

1. Supervised: learn from (input, output) pairs
2. Always split data: train/val/test
3. Never touch test set until final evaluation
4. Bias-variance tradeoff is fundamental
5. Use cross-validation for reliable estimates

## What's Next?

Step 5: **Linear Regression** — our first algorithm.
