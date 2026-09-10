# Step 3: Probability & Statistics

## Why Probability?

ML deals with uncertainty:
- Data is noisy
- We make predictions, not certainties
- We need to quantify confidence

## Basic Probability

```
P(A) = probability of event A
0 ≤ P(A) ≤ 1
P(not A) = 1 - P(A)
```

### Joint and Conditional Probability

```
P(A and B) = P(A, B) — joint probability
P(A | B) = P(A given B) — conditional probability

P(A | B) = P(A, B) / P(B)
```

## Bayes' Theorem

```
P(A | B) = P(B | A) × P(A) / P(B)

posterior = likelihood × prior / evidence
```

**Example:** Spam filtering
```
P(spam | "free money") = P("free money" | spam) × P(spam) / P("free money")
```

## Common Distributions

### Bernoulli (coin flip)
```python
# P(X=1) = p, P(X=0) = 1-p
from scipy.stats import bernoulli
samples = bernoulli.rvs(p=0.7, size=100)
```

### Gaussian (normal)
```python
# Bell curve: most common distribution
from scipy.stats import norm
samples = norm.rvs(loc=0, scale=1, size=100)  # mean=0, std=1
```

## Expected Value and Variance

```python
# Expected value: average outcome
E[X] = Σ x × P(x)

# Variance: spread of distribution
Var[X] = E[(X - E[X])²]

# Standard deviation
std = √Var[X]
```

## Maximum Likelihood Estimation

Given data, find parameters that make the data most likely.

```
θ_ML = argmax P(data | θ)
     = argmax Π P(x_i | θ)    # For independent samples
     = argmax Σ log P(x_i | θ) # Log-likelihood (easier to optimize)
```

**Example:** Estimating coin bias
```python
# Flip coin 100 times, get 70 heads
# MLE: p = 70/100 = 0.7
```

## Central Limit Theorem

Average of many samples → normal distribution.

This is why normal distributions appear everywhere!

## Hypothesis Testing (Intuition)

```
Null hypothesis: "No effect" or "Model is just guessing"
p-value: Probability of seeing this result if null is true

p < 0.05 → Reject null → Result is statistically significant
```

## Files

- `probability_statistics.py` - Distributions and sampling

## Key Takeaways

1. Probability quantifies uncertainty
2. Bayes' theorem: update beliefs with evidence
3. Gaussian distribution is central to ML
4. MLE: find parameters that maximize likelihood
5. Statistics helps evaluate models

## What's Next?

Step 4: **The ML Problem Setup** — framing learning problems.
