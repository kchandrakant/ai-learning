# Step 12: Model Evaluation

## Why Evaluation Matters

Evaluation is how you know if your model actually works. The right metrics and validation strategy are critical for building trustworthy models.

```
"My model has 99% accuracy!"

But wait...
- Is the data imbalanced? (99% one class = 99% by guessing)
- Did you test on training data? (overfitting)
- Is accuracy even right? (cancer detection needs recall)
```

---

## Train/Validation/Test Split

```
Full Data
    |
    +-- Train (60-70%)      -> Fit model parameters
    |
    +-- Validation (15-20%) -> Tune hyperparameters
    |
    +-- Test (15-20%)       -> Final evaluation (touch ONCE!)
```

**Golden Rule:** Never let test data influence any decision.

---

## Classification Metrics

### The Confusion Matrix

```
                    Predicted
                 |  Neg  |  Pos  |
           ------|-------|-------|
    Actual  Neg  |  TN   |  FP   |  <- Type I Error
           ------|-------|-------|
            Pos  |  FN   |  TP   |  <- Type II Error
                    ^
                    Type II Error

TN = True Negative   (correctly predicted negative)
TP = True Positive   (correctly predicted positive)
FP = False Positive  (predicted positive, actually negative)
FN = False Negative  (predicted negative, actually positive)
```

### Core Metrics

| Metric | Formula | Question Answered |
|--------|---------|-------------------|
| **Accuracy** | (TP+TN) / All | Overall, how often correct? |
| **Precision** | TP / (TP+FP) | When I predict positive, how often right? |
| **Recall** | TP / (TP+FN) | Of actual positives, how many caught? |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean of precision & recall |
| **Specificity** | TN / (TN+FP) | Of actual negatives, how many identified? |

### Computing from Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
```

---

## Why Accuracy Fails on Imbalanced Data

```
Dataset: 99% legitimate transactions, 1% fraud

Model A: Predicts "legitimate" always
- Accuracy: 99%  <- Looks great!
- Recall: 0%     <- Catches ZERO fraud!

Model B: Actually learned patterns
- Accuracy: 97%  <- Lower accuracy
- Recall: 85%    <- Catches most fraud!

Which model would you deploy?
```

**Lesson:** Use Precision, Recall, F1, or AUC on imbalanced data.

---

## Precision vs Recall Tradeoff

You can't maximize both - adjusting the prediction threshold trades one for the other.

```
High threshold (0.9):           Low threshold (0.1):
  Predict positive only           Predict positive often
  when very confident
  
  High Precision                  High Recall
  Low Recall                      Low Precision
  "Few but accurate"              "Catch all, many false alarms"
```

### Adjusting Threshold

```python
# Get probabilities
y_proba = model.predict_proba(X)[:, 1]

# Default threshold (0.5)
y_pred = (y_proba >= 0.5).astype(int)

# High precision (few false positives)
y_pred = (y_proba >= 0.8).astype(int)

# High recall (catch more positives)
y_pred = (y_proba >= 0.2).astype(int)
```

---

## When to Prioritize Which Metric

### Precision Matters (FP Costly)

- **Email spam filter:** Don't lose real emails
- **Recommending surgery:** Don't recommend unnecessarily
- **Criminal conviction:** Don't convict innocent people

### Recall Matters (FN Costly)

- **Cancer screening:** Don't miss any cancer
- **Fraud detection:** Catch all fraud
- **Safety systems:** Don't miss dangerous situations

### F1 Score

- Need balance between precision and recall
- Both types of errors matter similarly
- Imbalanced classes

---

## ROC Curve and AUC

**ROC Curve:** True Positive Rate (Recall) vs False Positive Rate at all thresholds

```
TPR (Recall)
  |      ___----  <- Good model (hugs top-left)
  |    /
  |   /
  |  /
  | / 
  |/_________ FPR (1 - Specificity)

Diagonal = random guessing (AUC = 0.5)
Top-left corner = perfect (AUC = 1.0)
```

### AUC Interpretation

- **AUC = 1.0:** Perfect classifier
- **AUC = 0.9:** Excellent
- **AUC = 0.8:** Good
- **AUC = 0.7:** Fair
- **AUC = 0.5:** Random guessing
- **AUC < 0.5:** Worse than random (flip predictions!)

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
```

---

## Precision-Recall Curve

Better than ROC for **imbalanced data**:

```
Precision
  |----\
  |     \___
  |         \___
  |             \
  |______________\ Recall

Baseline = proportion of positive class
```

### Average Precision (AP)

Area under PR curve - single number summary.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
```

---

## Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | mean((y - y_pred)^2) | Average squared error |
| **RMSE** | sqrt(MSE) | Error in original units |
| **MAE** | mean(\|y - y_pred\|) | Average absolute error |
| **R^2** | 1 - SS_res/SS_tot | Variance explained (0-1) |
| **MAPE** | mean(\|y - y_pred\|/y) | Percentage error |

### When to Use Which

| Situation | Metric | Why |
|-----------|--------|-----|
| Standard case | RMSE | Penalizes large errors |
| Outliers present | MAE | Robust to outliers |
| Relative error | MAPE | Percentage-based |
| Model comparison | R^2 | Normalized, interpretable |

### R^2 Interpretation

```
R^2 = 1.0  -> Perfect predictions
R^2 = 0.8  -> Model explains 80% of variance
R^2 = 0.0  -> No better than predicting mean
R^2 < 0.0  -> Worse than predicting mean (bad!)
```

---

## Cross-Validation

Single train/test split is noisy. Cross-validation gives reliable estimates.

### K-Fold Cross-Validation

```
Data: [1][2][3][4][5]

Fold 1: Train [2][3][4][5], Test [1] -> Score 1
Fold 2: Train [1][3][4][5], Test [2] -> Score 2
Fold 3: Train [1][2][4][5], Test [3] -> Score 3
Fold 4: Train [1][2][3][5], Test [4] -> Score 4
Fold 5: Train [1][2][3][4], Test [5] -> Score 5

Final = mean(Scores) +/- std(Scores)
```

### Benefits

- Every sample tested once
- More reliable estimate
- Get uncertainty (standard deviation)

### Implementation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Variants

- **Stratified K-Fold:** Preserves class ratios (use for classification)
- **Leave-One-Out:** K = n samples (thorough but expensive)
- **Time Series Split:** Respects temporal order

---

## Learning Curves

Diagnose overfitting vs underfitting:

```
Accuracy
  |  ___________train
  | /
  |/     ________val
  |     /
  |____/
  |_________________ Training Size

Large gap = OVERFITTING
Both low = UNDERFITTING
Converging = GOOD FIT
```

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
```

---

## Overfitting Detection

Compare train vs validation performance:

| Train | Validation | Diagnosis |
|-------|------------|-----------|
| 95% | 93% | Good fit |
| 99% | 70% | **OVERFITTING** |
| 60% | 58% | Underfitting |
| 70% | 75% | Data leakage? |

---

## Hyperparameter Tuning

### Grid Search

Try all combinations:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
```

### Random Search

Sample random combinations - often faster for large search spaces:

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=100, cv=5
)
```

---

## Common Evaluation Mistakes

1. **Testing on training data** -> Overly optimistic results

2. **Data leakage** -> Future info leaks into training
   - Normalizing before splitting
   - Feature selection using test data

3. **Wrong metric** -> Accuracy on imbalanced data

4. **Single random split** -> Unreliable (use cross-validation)

5. **Overfitting to validation** -> Tune too much, lose generalization

6. **Ignoring business context** -> Technical best != business best

---

## Metric Selection Guide

### Classification

| Scenario | Primary Metric | Secondary |
|----------|----------------|-----------|
| Balanced classes | Accuracy, F1 | - |
| Imbalanced | F1, Recall | PR-AUC |
| FP costly | Precision | F1 |
| FN costly | Recall | F1 |
| Ranking needed | ROC-AUC | - |
| Imbalanced + ranking | PR-AUC | - |

### Regression

| Scenario | Primary Metric |
|----------|----------------|
| Standard | RMSE, R^2 |
| Outliers | MAE |
| Relative error | MAPE |

---

## Files

- `evaluation.py` - All metrics, curves, cross-validation, learning curves examples

## Key Takeaways

1. **Accuracy is often misleading** - especially on imbalanced data
2. **Precision-Recall tradeoff** - can't maximize both
3. **F1 balances** precision and recall
4. **ROC/AUC** for model comparison; **PR curve** for imbalanced
5. **Cross-validation** gives reliable estimates with uncertainty
6. **Learning curves** diagnose overfitting vs underfitting
7. **Consider business context** - what errors cost most?

## What's Next?

Step 13: **Feature Engineering** - creating, selecting, and transforming features to improve model performance.
