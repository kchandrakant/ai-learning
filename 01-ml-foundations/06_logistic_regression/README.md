# Step 6: Logistic Regression & Classification

## From Regression to Classification

Despite the name, logistic regression is for **classification**, not regression! It's one of the most important algorithms to understand deeply.

**Linear regression problem for classification:**
```
    y
  1 |         * * * *
    |       *
    |     *
  0 | * * *
    |_________________ x

Linear regression outputs can be:
- Greater than 1 (meaningless as probability)
- Less than 0 (meaningless as probability)
- Sensitive to outliers far from decision boundary
```

**Solution:** Squash outputs to range (0, 1) using the sigmoid function.

---

## The Sigmoid Function (Logistic Function)

```
sigmoid(z) = 1 / (1 + e^(-z))
```

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Shape and Properties

```
  1.0 |          --------
      |        /
  0.5 |------/---------- <- z=0 gives exactly 0.5
      |    /
  0.0 |----
      |_________________ z
         -inf    0    +inf
```

**Properties:**
- Output is always between 0 and 1 (valid probability!)
- sigmoid(0) = 0.5
- sigmoid(large positive) approaches 1
- sigmoid(large negative) approaches 0
- Smooth and differentiable everywhere
- Derivative: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

---

## The Logistic Regression Model

```
Step 1: Linear combination (same as linear regression)
z = w^T x + b = w₁x₁ + w₂x₂ + ... + b

Step 2: Apply sigmoid to get probability
p = sigmoid(z) = 1 / (1 + e^(-z))

Step 3: Predict class based on threshold
y_pred = 1 if p >= 0.5, else 0
```

**Interpretation:** p is the probability of belonging to class 1.

```python
def predict_proba(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def predict(X, w, b, threshold=0.5):
    proba = predict_proba(X, w, b)
    return (proba >= threshold).astype(int)
```

---

## Why Not MSE? The Cross-Entropy Loss

**Problem with MSE for classification:**

When you combine MSE with sigmoid, the loss surface becomes non-convex:

```
Loss surface with sigmoid + MSE:
    Loss
      |  \    /
      |   \  /   <- Multiple local minima!
      |    \/    <- Gradient descent can get stuck
      |_________ w
```

### Cross-Entropy (Log Loss)

```
L = -[y * log(p) + (1-y) * log(1-p)]

For a dataset:
L = -(1/n) * sum[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

```python
def cross_entropy_loss(y_true, y_pred):
    # Clip to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )
```

### Why Cross-Entropy Works

| y_true | p (predicted) | Loss | Interpretation |
|--------|---------------|------|----------------|
| 1 | 0.9 | -log(0.9) = 0.1 | Correct, low loss |
| 1 | 0.1 | -log(0.1) = 2.3 | Wrong, high loss! |
| 0 | 0.1 | -log(0.9) = 0.1 | Correct, low loss |
| 0 | 0.9 | -log(0.1) = 2.3 | Wrong, high loss! |

**Key insight:** Cross-entropy heavily penalizes confident wrong predictions.

**Loss surface with cross-entropy:**
```
    Loss
      |  \
      |   \
      |    \___  <- Convex! Single global minimum
      |_________ w
```

---

## Training with Gradient Descent

The gradient has a beautifully simple form:

```
dL/dw = (1/n) * X^T @ (predictions - actual)
dL/db = (1/n) * sum(predictions - actual)
```

**Same form as linear regression!** The sigmoid is "absorbed" into the predictions.

```python
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(epochs):
        # Forward pass
        z = X @ w + b
        y_pred = sigmoid(z)
        
        # Compute gradients
        error = y_pred - y
        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) * np.sum(error)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
    
    return w, b
```

---

## The Decision Boundary

Logistic regression finds a **linear decision boundary** - a line (2D), plane (3D), or hyperplane (higher D) that separates classes.

```
Two classes in 2D:
    x₂
     |    Class 1 (o)
     |  o  o
     | o  o  \
     |--------\---- <- Decision boundary: w₁x₁ + w₂x₂ + b = 0
     |    x  x \    
     |  x    x
     |    Class 0 (x)
     |________________ x₁
```

**The boundary is where:**
```
w^T x + b = 0
sigmoid(0) = 0.5
P(y=1) = P(y=0) = 50%
```

Points on one side: P(y=1) > 0.5 -> predict 1
Points on other side: P(y=1) < 0.5 -> predict 0

---

## Probabilistic Interpretation: Log-Odds

Logistic regression models the **log-odds** (logit):

```
log(p / (1-p)) = w^T x + b

Where p/(1-p) is the "odds ratio"
```

### Interpreting Coefficients

If w₁ = 0.5:
- For each unit increase in x₁, log-odds increase by 0.5
- Equivalently: odds multiply by e^0.5 ≈ 1.65 (65% increase in odds)

**Example: Medical risk factors**
```
log-odds(heart disease) = 0.03*age + 0.5*smoker + 0.2*bmi - 8

- Being a smoker increases odds by e^0.5 = 1.65x (65% higher odds)
- Each year of age increases odds by e^0.03 = 1.03x (3% higher odds)
```

This makes logistic regression highly interpretable for clinical, business, and policy applications.

---

## Multi-class Classification

### One-vs-Rest (OvR)

Train K binary classifiers, one for each class:

```
3 classes (A, B, C) -> 3 binary classifiers:
- Classifier 1: Is it A or not-A?
- Classifier 2: Is it B or not-B?
- Classifier 3: Is it C or not-C?

Prediction: Pick class with highest probability
```

### Softmax (Multinomial Logistic Regression)

Single model with K output values, one per class:

```
z = [z₁, z₂, ..., zₖ]  (one score per class)

softmax(zᵢ) = e^zᵢ / sum(e^zⱼ)
```

**Properties:**
- All outputs sum to 1 (valid probability distribution)
- Largest input gets largest probability
- Generalizes sigmoid to multiple classes

```python
def softmax(z):
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Example
z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)  # [0.66, 0.24, 0.10] - sums to 1
```

---

## Evaluation Metrics for Classification

### The Confusion Matrix

```
                 Predicted
              |  Pos  |  Neg  |
        ----------------------
        Pos   |  TP   |  FN   |  <- Actual Positives
Actual  ----------------------
        Neg   |  FP   |  TN   |  <- Actual Negatives

TP = True Positive (correctly predicted positive)
TN = True Negative (correctly predicted negative)
FP = False Positive (incorrectly predicted positive) - Type I error
FN = False Negative (incorrectly predicted negative) - Type II error
```

### Key Metrics

| Metric | Formula | Question Answered |
|--------|---------|-------------------|
| **Accuracy** | (TP+TN)/(All) | Overall, how often correct? |
| **Precision** | TP/(TP+FP) | When I predict positive, how often right? |
| **Recall** | TP/(TP+FN) | Of actual positives, how many did I catch? |
| **F1 Score** | 2*(P*R)/(P+R) | Harmonic mean of precision & recall |
| **Specificity** | TN/(TN+FP) | Of actual negatives, how many did I identify? |

### When to Use Which Metric

**Precision matters when false positives are costly:**
- Spam filter (don't want to lose real emails)
- Criminal conviction (don't want to jail innocent people)

**Recall matters when false negatives are costly:**
- Cancer screening (don't want to miss any cases)
- Fraud detection (don't want to miss fraud)

**F1 Score when you need balance:**
- Imbalanced classes
- Both types of errors matter

**Accuracy is often misleading:**
```
99% of transactions are legitimate, 1% are fraud.
A model that predicts "legitimate" always gets 99% accuracy!
But it catches 0% of fraud - useless!

Recall = 0/100 = 0%  <- This reveals the problem
```

---

## Threshold Tuning

Default threshold is 0.5, but you can adjust:

```python
# Conservative (predict positive only when very confident)
threshold = 0.7
# Result: Higher precision, lower recall

# Aggressive (catch more positives)
threshold = 0.3  
# Result: Higher recall, lower precision
```

### ROC Curve

Plot True Positive Rate (Recall) vs False Positive Rate at all thresholds:

```
TPR |     ___----  <- Good model (hugs top-left)
    |   /
    |  /     
    | /
    |/_______ FPR

TPR |    /
    |   /
    |  /         <- Random model (diagonal)
    | /
    |/_______ FPR
```

### AUC (Area Under ROC Curve)

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (flip predictions!)

**AUC is threshold-independent** - good for comparing models overall.

---

## Precision-Recall Curve

For imbalanced datasets, PR curve is often more informative:

```
Precision |----\
          |     \___
          |         \___
          |             \
          |______________\ Recall

As you lower threshold:
- Recall increases (catch more positives)
- Precision typically decreases (more false positives)
```

---

## Logistic Regression vs Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| Output | Continuous number | Probability (0-1) |
| Activation | None (identity) | Sigmoid |
| Loss function | MSE | Cross-entropy |
| Use case | Predict quantities | Classify categories |
| Interpretation | Unit change in x -> w change in y | Unit change in x -> w change in log-odds |

---

## Scikit-learn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score,
                           confusion_matrix, classification_report)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## Files

- `logistic_regression.py` - Implementation from scratch with visualization

## Key Takeaways

1. **Sigmoid squashes** linear output to valid probability (0, 1)
2. **Cross-entropy loss** is convex, unlike MSE for classification
3. **Decision boundary** is linear - a hyperplane in feature space
4. **Coefficients = change in log-odds** - still interpretable!
5. **Softmax** extends sigmoid to multi-class
6. **Choose metrics wisely** - accuracy isn't always appropriate
7. **Threshold tuning** lets you trade precision for recall

## What's Next?

Step 7: **Gradient Descent Deep Dive** - the optimization algorithm powering all of modern ML.
