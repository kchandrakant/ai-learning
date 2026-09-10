# Step 8: Regularization & Generalization

## The Overfitting Problem

Model memorizes training data, fails on new data.

```
Training accuracy: 99%
Test accuracy: 60%  ← Overfitting!
```

## Regularization: Add Penalty for Complexity

```
Loss_regularized = Loss_original + λ × Complexity_penalty

λ: regularization strength
```

## L2 Regularization (Ridge)

Penalize sum of squared weights.

```python
# Loss = MSE + λ × Σ(w²)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)
```

**Effect:** Shrinks all weights toward zero (but never exactly zero).

## L1 Regularization (Lasso)

Penalize sum of absolute weights.

```python
# Loss = MSE + λ × Σ|w|

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
```

**Effect:** Can shrink weights to exactly zero → feature selection!

## Elastic Net

Combination of L1 and L2.

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50% L1, 50% L2
```

## Early Stopping

Stop training when validation loss stops improving.

```python
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(1000):
    train_one_epoch()
    val_loss = evaluate(X_val, y_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        break  # Early stop
```

## Dropout (for Neural Networks)

Randomly "drop" neurons during training.

```
Training:  randomly set 50% of neurons to 0
Testing:   use all neurons (scaled)
```

Prevents neurons from co-adapting too much.

## Data Augmentation

Create more training data through transformations.

```python
# For images:
# - Rotate, flip, crop
# - Adjust brightness, contrast
# - Add noise

# For text:
# - Synonym replacement
# - Back-translation
```

## Choosing λ (Regularization Strength)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_}")
```

## Summary

| Technique | Effect |
|-----------|--------|
| L2 (Ridge) | Shrinks weights |
| L1 (Lasso) | Shrinks and zeros weights |
| Early stopping | Stops before overfitting |
| Dropout | Prevents co-adaptation |
| Data augmentation | More diverse training data |

## Files

- `regularization.py` - Regularization examples

## Key Takeaways

1. Overfitting: train ≫ test performance
2. Regularization penalizes complexity
3. L2 shrinks, L1 zeros out features
4. Early stopping is simple and effective
5. More data often beats fancier regularization

## What's Next?

Step 9: **Decision Trees & Ensembles** — tree-based methods.
