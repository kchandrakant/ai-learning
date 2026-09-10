# Step 10: Support Vector Machines

## The Idea: Maximum Margin

Find the hyperplane that maximizes the margin between classes.

```
       o     o
         o       margin
    ──────────────  ← hyperplane
         ×       margin
       ×     ×
```

## Support Vectors

The data points closest to the decision boundary.
Only these points determine the boundary!

## Linear SVM

```python
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Support vectors
print(svm.support_vectors_)
```

## Soft Margin (C parameter)

Real data isn't perfectly separable. Allow some misclassification.

```python
# C: penalty for misclassification
# High C: try hard to classify all correctly (may overfit)
# Low C:  allow more errors (more regularization)

svm = SVC(kernel='linear', C=1.0)
```

## The Kernel Trick

Handle non-linear boundaries by mapping to higher dimensions.

```python
# RBF (Gaussian) kernel - most common
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)
```

### Intuition
```
Original space:  x₁, x₂
Mapped space:    x₁, x₂, x₁², x₂², x₁×x₂, ...

Linear boundary in high-dim = non-linear in original space
Kernel computes this without explicitly transforming!
```

## Gamma Parameter (RBF)

```
High gamma: Tight decision boundary, can overfit
Low gamma:  Smooth boundary, more generalization
```

## SVM vs Other Methods

| Aspect | SVM | Logistic Regression | Neural Networks |
|--------|-----|---------------------|-----------------|
| Decision boundary | Maximum margin | Maximum likelihood | Learned |
| Non-linearity | Kernels | Feature engineering | Layers |
| Interpretability | Medium | High | Low |
| Scale | Medium datasets | Any | Large datasets |

## When to Use SVM

- Medium-sized datasets
- Need non-linear boundaries
- High-dimensional data (text classification)
- When interpretability of support vectors matters

## Files

- `svm.py` - SVM examples and visualization

## Key Takeaways

1. SVM finds maximum margin hyperplane
2. Support vectors are the critical points
3. Kernel trick enables non-linear boundaries
4. C controls regularization
5. Good for medium-sized, high-dimensional data

## What's Next?

Step 11: **Unsupervised Learning** — learning without labels.
