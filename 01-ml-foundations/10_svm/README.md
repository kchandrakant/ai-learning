# Step 10: Support Vector Machines (SVM)

## Why SVM Matters

SVMs are elegant algorithms that find the **optimal decision boundary** by maximizing the margin between classes. They're particularly effective for high-dimensional data and remain a strong baseline for many classification tasks.

---

## The Core Idea: Maximum Margin

Instead of finding just *any* separator, SVM finds the one with the **largest margin** - the maximum distance to the nearest points of each class.

```
Poor separator:              SVM's optimal separator:
    |                              |
  o | x                        o   |   x
  o |   x                      o   |   x
    | x                          o |     x
  o |   x                      <-margin->
    
Arbitrary line               Maximum gap on both sides
```

**Why maximum margin?**
- Better generalization to new data
- Most robust to small perturbations
- Theoretically justified (statistical learning theory)

---

## Support Vectors

The **support vectors** are data points closest to the decision boundary - they "support" or define where the boundary sits.

```
         o
        o
       [o]  <-- Support vector (on margin)
    ======== <-- Decision boundary
       [x]  <-- Support vector (on margin)
        x
         x
```

**Key insight:** Only support vectors matter for the decision boundary. Removing other points doesn't change it!

---

## The Math (Simplified)

**Decision boundary:** `w·x + b = 0`

**Margins:** `w·x + b = +1` (positive class margin) and `w·x + b = -1` (negative class margin)

**Margin width:** `2 / ||w||`

**Optimization:**
```
Minimize:    (1/2)||w||²
Subject to:  y_i(w·x_i + b) >= 1 for all points

Translation: Make ||w|| small (wide margin) while keeping
             all points on the correct side of the margin.
```

---

## Soft Margin: The C Parameter

Real data is rarely perfectly separable. **Soft margin SVM** allows some misclassifications, controlled by **C**:

```
C = small (0.01):         C = medium (1):          C = large (100):
  o    |    x              o   |   x                o|x  
  o  x |    x              o   |   x                o|   x
  o    |  x                o   | x                  o|   x
  
Wide margin              Balanced                 Narrow margin
Allows violations        Good tradeoff            Forces correct classification
May underfit             Usually best             May overfit
```

**Interpretation:**
- **Small C:** "I don't care much about misclassifications, give me a wide margin"
- **Large C:** "I really want every point classified correctly, margin size doesn't matter"

---

## The Kernel Trick: Non-Linear Boundaries

**Problem:** Many datasets aren't linearly separable.

```
Linearly separable:          NOT linearly separable:
    o o o                         o o o
    -----                        o x x o
    x x x                        o x x o
                                  o o o
Can draw a line              No line works!
```

**Solution:** Map data to a higher dimension where it IS separable!

### The Kernel Trick

Instead of explicitly computing the high-dimensional mapping (expensive), we use a **kernel function** that computes dot products in that space directly.

```
K(x, y) = phi(x) · phi(y)

We never compute phi(x), just the dot product!
```

---

## Common Kernels

### Linear Kernel
```
K(x, y) = x · y
```
- No transformation
- Fast, works when data is linearly separable
- Good for high-dimensional sparse data (text classification)

### Polynomial Kernel
```
K(x, y) = (gamma * x·y + r)^d
```
- Creates polynomial decision boundaries
- Degree `d` controls complexity

### RBF (Radial Basis Function) Kernel
```
K(x, y) = exp(-gamma * ||x - y||²)
```
- Most popular for non-linear problems
- Creates smooth, flexible boundaries
- `gamma` controls "influence radius" of each point

### When to Use Which

| Kernel | Best For |
|--------|----------|
| Linear | High-dimensional data, text, when linear separation works |
| RBF | General non-linear problems, default choice |
| Polynomial | When you know polynomial relationship exists |

---

## The Gamma Parameter

For RBF kernel, **gamma** controls how far the influence of a single training example reaches:

```
Small gamma (0.1):          Large gamma (10):
    ___________                 _   _   _
   /           \               / \ / \ / \
  Smooth boundary             Wiggly boundary
  Far-reaching influence      Local influence only
  May underfit                May overfit
```

**Intuition:**
- Small gamma: Each point influences a large area -> smooth boundary
- Large gamma: Each point only influences nearby area -> complex boundary

---

## Feature Scaling is CRITICAL

SVM uses distances, so features must be on the same scale:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Same transformation!
```

```
Without scaling:              With scaling:
Feature 1: 0-1000            Feature 1: -2 to +2
Feature 2: 0-1               Feature 2: -2 to +2

Feature 1 dominates!         Both contribute equally
Distance is meaningless      Proper distance calculation
```

**Always scale before SVM!**

---

## Hyperparameter Tuning

The key hyperparameters to tune:

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| C | Regularization (margin vs errors) | 0.001 to 1000 |
| gamma | RBF kernel reach | 'scale', 0.001 to 10 |
| kernel | Decision boundary shape | 'rbf', 'linear', 'poly' |

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
```

---

## Multi-Class Classification

SVM is inherently binary. For multiple classes:

**One-vs-One (OvO):** Default in sklearn
- Train K(K-1)/2 classifiers for every pair of classes
- Predict by voting

**One-vs-Rest (OvR):**
- Train K classifiers: "Class i vs all others"
- Predict class with highest confidence

```python
# sklearn uses OvO by default
svm = SVC(kernel='rbf')  # Works with any number of classes
svm.fit(X_train, y_train)  # y can have multiple classes
```

---

## SVM Variants

### SVC (Support Vector Classifier)
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1, gamma='scale')
```
- Supports all kernels
- Slower for large datasets

### LinearSVC
```python
from sklearn.svm import LinearSVC
svm = LinearSVC(C=1, max_iter=10000)
```
- Linear kernel only
- Much faster for large datasets
- Scales better (uses different algorithm)

### SVR (Support Vector Regression)
```python
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1, epsilon=0.1)
```
- For regression tasks
- Same kernel trick applies

---

## Pros and Cons

### Advantages
- Effective in high-dimensional spaces
- Memory efficient (stores only support vectors)
- Versatile through different kernels
- Works well with clear margin of separation
- Robust to outliers (only support vectors matter)

### Disadvantages
- Slow on large datasets O(n²) to O(n³)
- Sensitive to feature scaling
- No probability estimates by default (use `probability=True`)
- Choosing the right kernel can be tricky
- Difficult to interpret (non-linear kernels)

---

## When to Use SVM

**Good for:**
- Medium-sized datasets (hundreds to tens of thousands)
- High-dimensional data (text, genomics)
- Binary classification
- When you need a clear margin
- When data is not too noisy

**Not ideal for:**
- Very large datasets (> 100k samples) - too slow
- When you need probability outputs
- When interpretability is important
- Heavily overlapping classes

**Alternatives:**
- Large datasets: Use LinearSVC, Logistic Regression, or tree-based methods
- Probability needed: Logistic Regression or calibrated classifiers
- Interpretability: Decision Trees, Logistic Regression

---

## Practical Tips

1. **Always scale features** - use StandardScaler
2. **Start with RBF kernel** - it's the most flexible
3. **Use GridSearchCV** to tune C and gamma
4. **For large datasets**, use LinearSVC instead of SVC
5. **Check support vector count** - too many suggests overfitting or poor C
6. **Cross-validate** - don't trust single train/test split

---

## Files

- `svm.py` - Complete SVM examples with visualization

## Key Takeaways

1. **SVM maximizes margin** - finds the most robust separator
2. **Support vectors** define the boundary - other points don't matter
3. **C controls regularization** - tradeoff between margin width and errors
4. **Kernel trick** enables non-linear boundaries without explicit transformation
5. **RBF kernel** is the default choice for non-linear problems
6. **Gamma controls reach** - small = smooth, large = wiggly
7. **Always scale features** - SVM is distance-based
8. **Use LinearSVC for large datasets** - much faster

## What's Next?

Step 11: **Unsupervised Learning** - clustering, dimensionality reduction, and finding structure without labels.
