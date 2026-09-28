# Step 9: Decision Trees & Ensemble Methods

## Why Tree-Based Methods Matter

Tree-based methods are among the most powerful algorithms for **tabular/structured data**. Random Forests and Gradient Boosting consistently win Kaggle competitions on non-image, non-text data. Understanding them is essential for practical ML.

---

## The Core Idea: Decision Trees

A decision tree makes predictions by asking a series of yes/no questions:

```
                    Is income > $50k?
                    /              \
                  Yes               No
                  /                   \
         Age > 35?                Credit score > 700?
         /      \                   /            \
       Yes      No                Yes             No
       /         \                /                \
   APPROVE    REVIEW          APPROVE           DENY
```

**Intuition:** Like playing "20 Questions" - each question optimally splits the data to separate classes.

---

## How Trees Decide Where to Split

At each node, the tree tries every feature and threshold, picking the one that best separates classes.

### Impurity Measures

**Gini Impurity:**
```
Gini = 1 - sum(p_i^2) for all classes

- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity for binary (50/50)
```

**Entropy:**
```
Entropy = -sum(p_i * log2(p_i))

- Entropy = 0: Pure node
- Entropy = 1: Maximum for binary
```

### Information Gain

```
Information Gain = Parent Impurity - Weighted Avg(Child Impurities)
```

**Visual example:**
```
Before split:                After split on "Age > 30":
[X X X O O O O O]           Left: [X X X O]    Right: [O O O O]
Gini = 0.469                Gini = 0.375       Gini = 0 (pure!)

Information Gain = 0.469 - (4/8 * 0.375 + 4/8 * 0) = 0.281
```

The algorithm picks the split with **highest information gain**.

---

## Building the Tree: The Algorithm

```
function BuildTree(data, depth):
    # Stopping conditions
    if depth >= max_depth OR node is pure OR too few samples:
        return Leaf(majority_class)
    
    # Find best split
    best_feature, best_threshold = find_best_split(data)
    
    # Split data
    left_data = data where feature <= threshold
    right_data = data where feature > threshold
    
    # Recursively build children
    left_child = BuildTree(left_data, depth + 1)
    right_child = BuildTree(right_data, depth + 1)
    
    return Node(best_feature, best_threshold, left_child, right_child)
```

---

## Decision Tree Strengths

**Interpretability:**
```
You can explain predictions!
"Customer denied because:
 - Income < $50k (went left)
 - Credit score < 700 (went left)
 - Previous defaults > 0 -> DENY"
```

**No feature scaling needed:**
- Trees compare values, don't care about magnitude

**Handles mixed types:**
- Numerical: "Age > 35"
- Categorical: "Color == Red"

**Captures non-linearity naturally:**
```
Linear model needs tricks:       Tree captures it directly:
    y                                y
    |   ***                          |  ---
    | **   **                        | |   |
    |*       *                       | |   |
    |_________x                      |_|___|_x
```

---

## Decision Tree Weaknesses

**Overfitting (High Variance):**
```
Full tree on training data:
    Train Accuracy: 100%   <- Memorized!
    Test Accuracy: 70%     <- Doesn't generalize

Decision boundary becomes jagged:
    /\  /\  /\
   /  \/  \/  \   <- Captures noise, not signal
```

**Instability:**
- Small data changes -> completely different tree
- High variance in predictions

**Can't extrapolate:**
```
Training: x in [0, 10]
Test: x = 15

Linear model: Extrapolates trend
Tree: Predicts same as nearest region (flat)
```

---

## Controlling Tree Complexity

```python
from sklearn.tree import DecisionTreeClassifier

# Regularized tree
tree = DecisionTreeClassifier(
    max_depth=5,           # Limit depth
    min_samples_split=10,  # Min samples to split a node
    min_samples_leaf=5,    # Min samples in leaf
    max_features='sqrt',   # Features considered per split
)
```

```
Deep tree (overfit):          Shallow tree (generalizes):
        /\                          /\
       /  \                        /  \
      /\  /\                     Leaf  Leaf
     /\/\/\/\
    Memorizes                  Captures pattern
```

---

## The Ensemble Idea

**Problem:** Single trees are unstable and overfit.

**Solution:** Combine many trees!

```
Single tree:              Ensemble:
     /\                   /\ + /\ + /\ + /\ = Average
    /  \                 Different trees make different errors
   High variance         Errors cancel out!
```

**Why it works:**
- Individual trees make uncorrelated errors
- Averaging reduces variance
- Wisdom of crowds!

---

## Bagging (Bootstrap Aggregating)

**Idea:** Train trees on different random subsets, average predictions.

```
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Bootstrap 1: [2, 3, 3, 5, 7, 7, 8, 9, 9, 10] -> Tree 1
Bootstrap 2: [1, 1, 2, 4, 5, 6, 6, 8, 9, 10] -> Tree 2
Bootstrap 3: [1, 3, 4, 4, 5, 6, 7, 7, 8, 10] -> Tree 3

Final = average(Tree1, Tree2, Tree3)
```

**Bootstrap sampling:** Sample with replacement
- ~63% of samples appear in each bootstrap
- ~37% are "out-of-bag" (OOB) - free validation!

---

## Random Forest

**Random Forest = Bagging + Random Feature Selection**

```
For each tree:
    1. Bootstrap sample of data
    2. At each split, only consider random subset of features
    3. Grow full tree
    
Prediction = majority vote (classification) or mean (regression)
```

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_features='sqrt',   # sqrt(n_features) per split
    max_depth=None,        # Grow full trees
    oob_score=True,        # Out-of-bag score
    n_jobs=-1,             # Use all cores
)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_}")  # Free validation!
```

### Why Random Features?

- **Decorrelates trees:** They make different errors
- **Prevents dominance:** One strong feature won't be in every tree
- **Increases diversity:** Better ensemble

### Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| n_estimators | More = better (diminishing returns) | 100-500 |
| max_features | Lower = more diversity, higher bias | 'sqrt', 'log2' |
| max_depth | Lower = less overfitting | None, 10-30 |
| min_samples_leaf | Higher = less overfitting | 1-10 |

---

## Feature Importance

Random Forests provide feature importance for free:

```python
importances = rf.feature_importances_

# Most important features
for i in np.argsort(importances)[::-1][:5]:
    print(f"Feature {i}: {importances[i]:.4f}")
```

**How it's calculated:**
- Total impurity decrease from splits on that feature
- Averaged across all trees
- Normalized to sum to 1

**Caveats:**
- Correlated features split importance
- High-cardinality features can be biased higher

---

## Boosting: Sequential Ensemble

**Different philosophy from bagging:**

| Bagging | Boosting |
|---------|----------|
| Trees trained independently | Trees trained sequentially |
| Parallel | Sequential |
| Reduces variance | Reduces bias |
| Averages predictions | Weighted sum |

**Boosting idea:**
```
Round 1: Train Tree1
         -> Some samples wrong
         
Round 2: Train Tree2, focus on mistakes
         -> Still some errors
         
Round 3: Train Tree3, focus on remaining errors
         ...

Final: Weighted combination of all trees
```

---

## Gradient Boosting

The most powerful boosting method. Each tree fits the **residuals** (errors) of the current ensemble.

```
F0(x) = mean(y)                    # Initial prediction

Round 1:
    residuals1 = y - F0(x)         # What's left to predict
    Train h1 to predict residuals1
    F1(x) = F0(x) + lr * h1(x)     # Update ensemble

Round 2:
    residuals2 = y - F1(x)         # Remaining errors
    Train h2 to predict residuals2
    F2(x) = F1(x) + lr * h2(x)
    
...continue for n_estimators rounds
```

**Why "Gradient"?**
```
Residuals = negative gradient of squared error loss!

Fitting residuals = gradient descent in function space
```

This generalizes to any differentiable loss function.

---

## Gradient Boosting Implementations

### Scikit-learn
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
)
```

### XGBoost (eXtreme Gradient Boosting)
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
)
```

### LightGBM (Fastest)
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
)
```

### CatBoost (Best for Categoricals)
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    cat_features=cat_cols,  # Handles categoricals natively
)
```

### Which to Choose?

| Library | Best For |
|---------|----------|
| XGBoost | General use, battle-tested |
| LightGBM | Large datasets, speed |
| CatBoost | Data with many categoricals |

---

## Gradient Boosting Hyperparameters

**Most Important:**

| Parameter | Effect |
|-----------|--------|
| n_estimators | More trees (use with early stopping) |
| learning_rate | Lower = need more trees, better generalization |
| max_depth | Lower = less overfitting (typically 3-8) |

**Key insight:** learning_rate and n_estimators are coupled.

```python
# Best practice: Set high n_estimators, use early stopping
model = xgb.XGBClassifier(
    n_estimators=10000,
    learning_rate=0.01,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          verbose=False)
```

---

## Random Forest vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training | Parallel (fast) | Sequential (slower) |
| Overfitting risk | Low | Higher (needs tuning) |
| Default performance | Good | Often better when tuned |
| Ease of use | Robust defaults | Needs careful tuning |
| Extrapolation | Neither can extrapolate | Neither can extrapolate |

**Rule of thumb:**
1. Start with Random Forest (hard to mess up)
2. Move to Gradient Boosting for competitions/production

---

## When to Use Tree-Based Methods

**Great for:**
- Tabular/structured data (the most common data type!)
- Mixed feature types
- When interpretability matters
- When you need feature importance
- Kaggle competitions on tabular data

**Not ideal for:**
- Images, text, audio -> use neural networks
- Very high-dimensional sparse data
- When you need smooth predictions
- Extrapolation beyond training range

---

## Common Pitfalls

1. **Not using early stopping with boosting**
   - Can severely overfit without it

2. **Too many trees without regularization**
   - More isn't always better for single trees

3. **Ignoring feature importance caveats**
   - Correlated features, cardinality bias

4. **Using trees for extrapolation**
   - They predict flat outside training range

---

## Files

- `trees_ensembles.py` - Decision tree from scratch, Random Forest, Gradient Boosting examples

## Key Takeaways

1. **Decision trees** split data to maximize information gain (reduce impurity)
2. **Single trees overfit** - unstable, high variance
3. **Random Forest** = bagging + random features -> reduces variance
4. **Gradient Boosting** = sequential trees on residuals -> reduces bias
5. **XGBoost/LightGBM/CatBoost** are industrial-strength implementations
6. **Use early stopping** with boosting methods
7. **Trees dominate** tabular data, but can't extrapolate

## What's Next?

Step 10: **Support Vector Machines** - finding optimal decision boundaries with margins.
