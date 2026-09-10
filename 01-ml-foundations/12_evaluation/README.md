# Step 12: Model Evaluation & Selection

## Why Evaluation Matters

A model is useless if you don't know how well it works.

## Train/Validation/Test

```
Training Set   → Train the model
Validation Set → Tune hyperparameters
Test Set       → Final evaluation (use ONCE)
```

**Golden rule:** Never tune based on test set!

## K-Fold Cross-Validation

More reliable than single split.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Classification Metrics

### Confusion Matrix

```
                Predicted
              Neg    Pos
Actual  Neg   TN     FP
        Pos   FN     TP
```

### Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: (TP + TN) / Total
accuracy = accuracy_score(y_true, y_pred)

# Precision: TP / (TP + FP)  "Of predicted positive, how many correct?"
precision = precision_score(y_true, y_pred)

# Recall: TP / (TP + FN)  "Of actual positive, how many found?"
recall = recall_score(y_true, y_pred)

# F1: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)
```

### When to Use Which

| Metric | Use When |
|--------|----------|
| Accuracy | Balanced classes |
| Precision | False positives costly (spam detection) |
| Recall | False negatives costly (disease detection) |
| F1 | Balance precision and recall |

## ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)
auc = roc_auc_score(y_test, y_prob)
```

## Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)  # 1.0 = perfect
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid search: try all combinations
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")

# Random search: sample from distributions (faster for large spaces)
from scipy.stats import uniform
param_dist = {'C': uniform(0.1, 10)}
random = RandomizedSearchCV(SVC(), param_dist, n_iter=20, cv=5)
```

## Files

- `evaluation.py` - Evaluation utilities

## Key Takeaways

1. Always use held-out test set for final evaluation
2. Cross-validation for reliable estimates
3. Choose metric based on problem (accuracy isn't always best)
4. ROC/AUC for ranking problems
5. Use grid/random search for hyperparameters

## What's Next?

Step 13: **Feature Engineering** — making features that matter.
