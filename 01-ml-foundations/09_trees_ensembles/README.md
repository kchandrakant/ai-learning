# Step 9: Decision Trees & Ensembles

## Decision Trees

Split data recursively based on feature values.

```
          [Root: Age < 30?]
         /                 \
       Yes                  No
       /                     \
[Income > 50K?]         [Credit: Good?]
    /    \                  /     \
  Yes    No               Yes      No
   |      |                |        |
Approve Reject          Approve  Reject
```

## How Trees Decide Splits

### Information Gain (Entropy)

```python
def entropy(y):
    proportions = np.bincount(y) / len(y)
    return -np.sum(p * np.log2(p) for p in proportions if p > 0)

def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    n = len(y)
    child_entropy = (len(y_left)/n * entropy(y_left) + 
                     len(y_right)/n * entropy(y_right))
    return parent_entropy - child_entropy
```

### Gini Impurity

```python
def gini(y):
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(p**2 for p in proportions)
```

## Using Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

## Problem: Overfitting

Single trees easily overfit. Solution: **Ensembles**.

## Random Forest

Train many trees on random subsets of data and features.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
```

**Key ideas:**
- Bagging: random samples with replacement
- Feature randomness: random subset of features at each split
- Average predictions across trees

## Gradient Boosting

Train trees sequentially, each fixing errors of previous.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

## XGBoost / LightGBM

Optimized gradient boosting implementations.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

**For tabular data, gradient boosting often beats neural networks!**

## When to Use Trees

| Scenario | Recommendation |
|----------|---------------|
| Tabular data | Trees/Boosting |
| Need interpretability | Single tree |
| Maximum accuracy | Gradient Boosting |
| Fast training | Random Forest |
| Images/Text | Neural Networks |

## Files

- `trees_ensembles.py` - Tree and ensemble examples

## Key Takeaways

1. Trees split data recursively
2. Single trees overfit → use ensembles
3. Random Forest: parallel trees, bagging
4. Gradient Boosting: sequential, error correction
5. XGBoost/LightGBM for best tabular performance

## What's Next?

Step 10: **Support Vector Machines** — margins and kernels.
