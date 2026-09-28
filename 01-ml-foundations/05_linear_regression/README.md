# Step 5: Linear Regression

## Your First Complete ML Algorithm

Linear regression is simple yet powerful. Understanding it deeply builds intuition for everything that follows - from logistic regression to neural networks.

---

## The Core Idea

**Goal:** Find the best straight line (or hyperplane) through your data.

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Or in vector form:
y = w^T x + b
```

Where:
- **w** = weights (how much each feature matters)
- **b** = bias/intercept (where line crosses y-axis)
- **x** = input features
- **y** = predicted output

```
    y
    |       *
    |     *   <- data points
    |   *  /
    | *   /  <- best fit line: y = wx + b
    |    /
    |___/___________ x
```

---

## What Makes a Line "Best"?

We measure error using **Mean Squared Error (MSE)**:

```
MSE = (1/n) * sum((y_i - y_pred_i)²)
```

### Why Squared?

1. **Penalizes large errors more** - being off by 10 is worse than being off by 1
2. **Makes math nice** - differentiable, convex (one minimum)
3. **Cancellation prevention** - negative and positive errors don't cancel out

```
Visual: Minimize sum of squared vertical distances

    y
    |       *
    |    /  |  <- error (residual)
    |   / * |
    |  /  | |
    | / * | |
    |/  | | |
    |_____|_|_______ x
    
Goal: Make these vertical distances as small as possible
```

---

## Two Ways to Find Optimal Weights

### Method A: Normal Equation (Closed-form)

Direct solution using linear algebra:

```
w = (X^T X)^(-1) X^T y
```

```python
def normal_equation(X, y):
    # Add column of ones for bias
    X_b = np.c_[np.ones(len(X)), X]
    # Solve directly
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta[0], theta[1:]  # bias, weights

b, w = normal_equation(X_train, y_train)
```

**Pros:** 
- Exact answer in one step
- No hyperparameters to tune

**Cons:** 
- Matrix inversion is O(n³) - slow for large datasets
- Can be numerically unstable
- Doesn't work if X^T X is not invertible

### Method B: Gradient Descent (Iterative)

Update weights step by step:

```
w = w - alpha * gradient_of_loss

Where gradient:
gradient = (2/n) * X^T @ (X @ w - y)
```

```python
def gradient_descent_regression(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        # Predictions
        y_pred = X @ w + b
        
        # Gradients
        dw = (2/n_samples) * (X.T @ (y_pred - y))
        db = (2/n_samples) * np.sum(y_pred - y)
        
        # Update
        w = w - lr * dw
        b = b - lr * db
    
    return w, b
```

**Pros:** 
- Scales to massive datasets
- Works for any differentiable loss
- Foundation for neural network training

**Cons:** 
- Need to choose learning rate
- Takes multiple iterations
- May not find exact optimum

---

## Feature Scaling: Why It Matters

**Problem:** Features on different scales cause optimization issues.

```
Feature 1: House size (500-5000 sq ft)
Feature 2: Number of bedrooms (1-6)
```

### Unscaled Loss Surface

```
     w2 (bedrooms)
     |  
     |  ____________________
     | /                    /
     |/  Very elongated    /
     /  ellipse           /
    /____________________/ 
                          w1 (sqft)

Gradient descent zigzags slowly down the narrow valley
```

### Scaled Loss Surface

```
     w2
     |  
     |    ___
     |   /   \
     |  |  O  |  Nice circular contours
     |   \___/
     |____________ w1

Gradient descent goes straight to minimum
```

### Common Scaling Methods

**Standardization (Z-score):** Mean=0, Std=1
```python
X_scaled = (X - X.mean()) / X.std()
```

**Min-Max Scaling:** Range [0, 1]
```python
X_scaled = (X - X.min()) / (X.max() - X.min())
```

**Always scale features before gradient descent!**

---

## Interpreting the Model

Linear regression is **interpretable** - you can explain what each coefficient means:

```python
# Example: House price prediction
# Price = 150*sqft + 50000*bedrooms + 20000*has_garage - 100000

# Interpretation:
# - Each additional sqft adds $150 to price
# - Each additional bedroom adds $50,000
# - Having a garage adds $20,000
# - Base price (intercept) is -$100,000
```

### Coefficient Magnitude as Feature Importance

After scaling features, the magnitude of coefficients indicates relative importance:

```python
# Scaled model
weights = [0.8, 0.3, 0.1]  # for sqft, bedrooms, bathrooms

# sqft is most important (0.8)
# bathrooms is least important (0.1)
```

**Caution:** This only works for scaled features! Otherwise you're comparing apples to oranges.

---

## Polynomial Regression

Linear regression can model **curves** by adding polynomial features:

```
Original: y = w₁x + b                    (line)
Add x²:   y = w₁x + w₂x² + b             (parabola)
Add x³:   y = w₁x + w₂x² + w₃x³ + b      (cubic curve)
```

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # [x] -> [1, x, x²]

# Still "linear" in parameters!
model = LinearRegression()
model.fit(X_poly, y)
```

### The Overfitting Danger

```
Degree 1            Degree 3           Degree 15
   /                  ~~~                /\/\/\/\
  /                 /     \             Memorizes
 /                 /       \            every point!
Underfits         Good fit             Overfits
High bias         Balance              High variance
```

**Higher degree = more flexibility = higher risk of overfitting**

---

## Assumptions of Linear Regression

For the model to be valid:

1. **Linearity:** Relationship between X and y is linear
   - Check: Plot residuals vs predicted - should be random scatter

2. **Independence:** Observations are independent
   - Violated if: Time series data, grouped data

3. **Homoscedasticity:** Constant variance of errors
   - Check: Residual plot shouldn't show funnel shape

4. **Normality:** Errors are normally distributed (for inference)
   - Check: Q-Q plot of residuals

### When Assumptions are Violated

| Violation | Solution |
|-----------|----------|
| Non-linear relationship | Add polynomial features, use non-linear model |
| Correlated errors | Time series models, mixed effects models |
| Non-constant variance | Transform target, weighted regression |
| Multicollinearity | Remove features, regularization, PCA |

---

## Evaluating Regression Models

### Mean Squared Error (MSE)
```python
MSE = np.mean((y_true - y_pred)**2)
```
- Units are squared (e.g., dollars²)
- Heavily penalizes large errors

### Root Mean Squared Error (RMSE)
```python
RMSE = np.sqrt(MSE)
```
- Same units as target
- More interpretable than MSE

### Mean Absolute Error (MAE)
```python
MAE = np.mean(np.abs(y_true - y_pred))
```
- Robust to outliers
- Directly interpretable (average error magnitude)

### R² (Coefficient of Determination)
```python
SS_res = np.sum((y_true - y_pred)**2)
SS_tot = np.sum((y_true - y_true.mean())**2)
R2 = 1 - SS_res/SS_tot
```

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.0: Model is no better than predicting the mean
- R² < 0.0: Model is worse than predicting the mean (bad model!)

```
R² = 0.8 means "Model explains 80% of variance in y"
```

---

## Regularization Preview

Regularization prevents overfitting by penalizing large weights:

```python
# Ridge (L2): penalize sum of squared weights
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

# Lasso (L1): penalize sum of absolute weights (can zero out features)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
```

More details in Module 8!

---

## Linear Regression with Scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaling!

# Train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Interpret
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

---

## Files

- `linear_regression.py` - Implementation from scratch with visualization

## Key Takeaways

1. **Linear regression** finds the best hyperplane minimizing squared errors
2. **Two methods:** Normal equation (exact, slow) vs Gradient descent (iterative, scalable)
3. **Always scale features** before gradient descent
4. **Coefficients are interpretable** - that's the power of linear models
5. **Polynomial features** let you fit curves while staying "linear" in parameters
6. **R²** tells you how much variance your model explains
7. **Foundation for neural networks** - add non-linearity and you get deep learning

## What's Next?

Step 6: **Logistic Regression** - adapting linear models for classification.
