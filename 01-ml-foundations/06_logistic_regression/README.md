# Step 6: Logistic Regression & Classification

## From Regression to Classification

Linear regression outputs any number.
Classification needs probabilities (0 to 1).

**Solution:** Squash output through sigmoid function.

## The Sigmoid Function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Properties:
# - Always outputs between 0 and 1
# - σ(0) = 0.5
# - σ(large positive) ≈ 1
# - σ(large negative) ≈ 0
```

## Logistic Regression Model

```
z = wx + b
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
```

The output is a probability!

## Cross-Entropy Loss

MSE doesn't work well for classification. Use cross-entropy:

```python
def cross_entropy_loss(y_true, y_pred):
    # Avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )
```

**Intuition:**
- If y=1 and we predict 0.9: small loss ✓
- If y=1 and we predict 0.1: large loss ✗

## Training

```python
def train_logistic(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    
    for _ in range(epochs):
        # Forward
        z = X @ w + b
        y_pred = sigmoid(z)
        
        # Gradients
        error = y_pred - y
        dw = (1/len(X)) * (X.T @ error)
        db = np.mean(error)
        
        # Update
        w = w - lr * dw
        b = b - lr * db
    
    return w, b
```

## Decision Boundary

```python
# Classify based on probability threshold
def predict(X, w, b, threshold=0.5):
    prob = sigmoid(X @ w + b)
    return (prob >= threshold).astype(int)
```

The decision boundary is where P(y=1) = 0.5, i.e., wx + b = 0.

## Multi-Class: Softmax

For K classes, use softmax:

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)

# Output: probability distribution over K classes
```

## Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: correct / total
# Precision: TP / (TP + FP) — "of predicted positives, how many correct?"
# Recall: TP / (TP + FN) — "of actual positives, how many found?"
# F1: harmonic mean of precision and recall
```

## Files

- `logistic_regression.py` - Implementation and visualization

## Key Takeaways

1. Sigmoid squashes linear output to [0, 1]
2. Output is a probability
3. Cross-entropy loss for classification
4. Decision boundary: where probability = 0.5
5. Softmax extends to multi-class

## What's Next?

Step 7: **Gradient Descent Deep Dive** — optimization variants.
