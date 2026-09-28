# Step 4: The ML Problem Setup

## Bridging Math to Machine Learning

This module connects the mathematical foundations with actual machine learning practice. Understanding how to properly set up an ML problem is crucial - many failures come from setup mistakes, not algorithm choice.

---

## Types of Learning

### Supervised Learning

Given inputs AND outputs, learn the mapping between them.

```
Training data: (X, y) pairs
Goal: Learn function f such that f(X) ≈ y

Examples:
- Image -> Label (classification)
- House features -> Price (regression)
- Email text -> Spam/Not spam
- Medical data -> Diagnosis
```

**Key characteristic:** You have labeled data to learn from.

### Unsupervised Learning

Only inputs, find hidden structure.

```
Training data: X only (no labels)
Goal: Find patterns, clusters, representations

Examples:
- Customer segmentation (grouping similar customers)
- Anomaly detection (finding unusual transactions)
- Dimensionality reduction (compressing features)
- Topic modeling (discovering themes in documents)
```

**Key characteristic:** No "right answer" - you're exploring the data.

### Semi-supervised Learning

Mix of labeled and unlabeled data.

```
Some (X, y) pairs + Many X only
Goal: Use unlabeled data to improve learning

Why it matters: Labels are expensive!
- Labeling 1M images costs thousands of dollars
- You might have 100 labeled + 100,000 unlabeled
```

### Reinforcement Learning

Learn through interaction and rewards.

```
Agent takes action -> Environment responds -> Reward
Goal: Maximize cumulative reward over time

Examples:
- Game playing (AlphaGo)
- Robotics (learning to walk)
- Recommendation systems
```

---

## Regression vs Classification

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Output type | Continuous numbers | Discrete categories |
| Example output | 23.5, 100.7, -3.2 | Cat, Dog, Bird |
| Loss function | MSE, MAE | Cross-entropy |
| Example task | Predict temperature | Predict weather type |

### Classification Subtypes

**Binary classification:** 2 classes
```
Spam / Not spam
Fraud / Legitimate
Disease / Healthy
```

**Multi-class classification:** 3+ mutually exclusive classes
```
Cat / Dog / Bird / Fish  (exactly one)
```

**Multi-label classification:** Multiple labels per sample
```
Image tags: [contains_cat, contains_dog, outdoors, sunny]
A single image can have multiple tags
```

---

## Data Terminology

```python
# Features (X): inputs, predictors, independent variables
# Labels (y): outputs, targets, dependent variable
# Example/Sample: one (X, y) pair
# Dataset: collection of examples

# Standard shape conventions:
X.shape = (n_samples, n_features)  # e.g., (1000, 10)
y.shape = (n_samples,)             # e.g., (1000,)

# Example: 1000 houses, 10 features each
# X[0] = first house's features
# y[0] = first house's price
```

---

## Train / Validation / Test Split

This is **crucial** and often misunderstood:

```
Full Dataset
    |
    +-- Training Set (60-80%)
    |       Used to train the model (fit parameters)
    |
    +-- Validation Set (10-20%)
    |       Used to tune hyperparameters, select model
    |       You CAN look at this during development
    |
    +-- Test Set (10-20%)
            Final evaluation ONLY
            Touch ONCE at the very end!
```

### Why Three Sets?

**Training set:** Model learns patterns by adjusting weights.

**Validation set:** YOU make decisions:
- Which model architecture?
- What hyperparameters?
- When to stop training?

If you use test set for these decisions, you're "cheating" - your test score won't reflect real-world performance.

**Test set:** Unbiased estimate of real-world performance.
- Simulates data you've never seen
- Touch only once, at the very end
- Report this number as your final result

### Implementation

```python
from sklearn.model_selection import train_test_split

# First, split off the test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Then split remaining into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
# 0.25 of 0.8 = 0.2, giving us 60/20/20 split

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### The Golden Rule

**Never let information from the test set influence your model in any way.**

This includes:
- Don't tune hyperparameters on test set
- Don't select features based on test set performance
- Don't normalize using test set statistics
- Don't even look at test set until you're completely done!

---

## Bias-Variance Tradeoff

This is fundamental to understanding model behavior:

```
Total Error = Bias² + Variance + Irreducible Noise
```

### What is Bias?

**Bias = error from wrong assumptions** (model too simple)

```
High Bias (Underfitting):
Reality: curved relationship
Model: tries to fit a straight line

    *   *
  *   *   *
    *   *       <- Data
  ---------     <- Model (can't capture curve)

Symptom: Poor on BOTH train and validation
```

### What is Variance?

**Variance = error from sensitivity to training data** (model too complex)

```
High Variance (Overfitting):
Reality: slightly noisy line
Model: memorizes every bump

    *   *
  * / \ / \
   /   *   \    <- Model (memorizes noise)
  *         *

Symptom: Great on train, poor on validation
```

### The Tradeoff Visualized

```
Error
  |
  |\                    Validation Error
  | \    ___________/
  |  \  /
  |   \/
  |    \____________  Training Error
  |
  +-------------------- Model Complexity
     Simple    <-->    Complex

     High Bias         High Variance
     (Underfit)        (Overfit)
              ^
              |
          Sweet Spot
```

### Practical Implications

| Situation | Diagnosis | Solution |
|-----------|-----------|----------|
| High train error, High val error | Underfitting (high bias) | More complex model, more features |
| Low train error, High val error | Overfitting (high variance) | Regularization, more data, simpler model |
| Low train error, Low val error | Good fit! | Ship it! |

---

## Cross-Validation

Single train/val split can be noisy. Cross-validation gives more reliable estimates.

### K-Fold Cross-Validation

```
Data: [1][2][3][4][5]

Fold 1: Train on [2][3][4][5], Validate on [1]
Fold 2: Train on [1][3][4][5], Validate on [2]
Fold 3: Train on [1][2][4][5], Validate on [3]
Fold 4: Train on [1][2][3][5], Validate on [4]
Fold 5: Train on [1][2][3][4], Validate on [5]

Final score = average of all 5 validation scores
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**Benefits:**
- Every data point gets to be in validation once
- More stable estimate of performance
- Standard: 5-fold or 10-fold

---

## Loss Functions

Loss functions measure how wrong predictions are.

### For Regression

**Mean Squared Error (MSE):**
```python
MSE = mean((y_true - y_pred)²)
```
- Penalizes large errors heavily (quadratic)
- Sensitive to outliers
- Most common choice

**Mean Absolute Error (MAE):**
```python
MAE = mean(|y_true - y_pred|)
```
- Linear penalty
- Robust to outliers
- Harder to optimize (not differentiable at 0)

### For Classification

**Cross-Entropy (Log Loss):**
```python
CE = -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
```
- Standard for classification
- Heavily penalizes confident wrong predictions
- Outputs meaningful probabilities

---

## Hyperparameters vs Parameters

| Parameters | Hyperparameters |
|------------|-----------------|
| Learned from data | Set by you before training |
| Weights, biases | Learning rate, regularization strength |
| Updated by optimizer | Tuned via validation set |
| Inside the model | Outside the model |

**Examples of hyperparameters:**
- Learning rate
- Number of layers
- Number of neurons per layer
- Regularization strength (lambda)
- Batch size
- Number of epochs

---

## The ML Workflow

```
1. Define Problem
   - What are you predicting?
   - What data do you have?
   - How will success be measured?
        |
        v
2. Prepare Data
   - Clean (handle missing values, outliers)
   - Split (train/val/test)
   - Normalize/standardize features
        |
        v
3. Establish Baseline
   - Simple model (majority class, mean prediction)
   - Gives you a minimum bar to beat
        |
        v
4. Choose Model
   - Start simple! Linear/logistic regression
   - Complex models aren't always better
        |
        v
5. Train
   - Fit on training data
   - Monitor training loss
        |
        v
6. Evaluate
   - Check validation performance
   - Compare to baseline
        |
        v
7. Tune
   - Adjust hyperparameters
   - Try different model architectures
        |
        v
8. Iterate (steps 4-7)
   - Try different approaches
   - Keep track of what you tried
        |
        v
9. Final Test
   - ONE-TIME evaluation on test set
   - This is your reported performance
        |
        v
10. Deploy & Monitor
    - Put in production
    - Monitor for data drift
    - Retrain periodically
```

---

## Common Mistakes to Avoid

1. **Data leakage:** Test information leaks into training
   - Normalizing before splitting
   - Feature selection using all data
   
2. **Not establishing a baseline:** You need something to compare against

3. **Overfitting to validation set:** Tuning too much on validation

4. **Ignoring class imbalance:** 99% accuracy on 99% majority class is useless

5. **Wrong metric:** Accuracy isn't always the right measure

---

## Files

- `ml_problem_setup.py` - Data splitting and cross-validation examples

## Key Takeaways

1. **Supervised vs Unsupervised:** Do you have labels?
2. **Always split data:** Train for learning, validation for tuning, test for final evaluation
3. **Never touch test set** until you're completely done
4. **Bias-Variance tradeoff:** Underfitting vs overfitting
5. **Cross-validation** gives more reliable performance estimates
6. **Start simple:** Complex models aren't always better

## What's Next?

Step 5: **Linear Regression** - your first complete ML algorithm from scratch.
