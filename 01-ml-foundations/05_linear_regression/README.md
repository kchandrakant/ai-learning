# Step 5: Linear Regression

## The Simplest ML Algorithm

Fit a line (or hyperplane) to data.

```
y = wx + b

w: weight (slope)
b: bias (intercept)
```

## The Setup

```python
import numpy as np

# Data: house size → price
X = np.array([1000, 1500, 2000, 2500, 3000])  # sq ft
y = np.array([200, 300, 400, 500, 600])        # $1000s

# Goal: learn w and b such that y ≈ w*X + b
```

## Loss Function: Mean Squared Error

```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Measures how wrong our predictions are
```

## Training with Gradient Descent

```python
def train_linear_regression(X, y, lr=0.0001, epochs=1000):
    w = 0.0
    b = 0.0
    n = len(X)
    
    for _ in range(epochs):
        # Predictions
        y_pred = w * X + b
        
        # Gradients
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)
        
        # Update
        w = w - lr * dw
        b = b - lr * db
    
    return w, b

w, b = train_linear_regression(X, y)
print(f"Price = {w:.2f} × size + {b:.2f}")
```

## Closed-Form Solution

For linear regression, we can solve directly:

```python
# Normal equation: w = (X^T X)^(-1) X^T y
def closed_form(X, y):
    X_b = np.c_[np.ones(len(X)), X]  # Add bias column
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta[1], theta[0]  # w, b
```

## Multiple Features

```python
# Multiple inputs: y = w1*x1 + w2*x2 + ... + b
# In matrix form: y = Xw

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Regularization

Prevent overfitting by penalizing large weights.

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (L2): penalize w²
ridge = Ridge(alpha=1.0)

# Lasso (L1): penalize |w| (can zero out features)
lasso = Lasso(alpha=1.0)
```

## Files

- `linear_regression.py` - Implementation from scratch

## Key Takeaways

1. Linear regression: y = wx + b
2. Loss function: mean squared error
3. Train with gradient descent or closed-form
4. Regularization prevents overfitting
5. Foundation for neural networks

## What's Next?

Step 6: **Logistic Regression** — classification.
