# Step 3: Probability & Statistics

## Why Probability in ML?

Machine learning deals with uncertainty everywhere:
- Data is noisy and incomplete
- We make predictions, not certainties
- We need to quantify confidence in our predictions

Probability gives us the language to reason about uncertainty rigorously.

---

## Basic Probability

```
P(A) = probability of event A occurring
0 <= P(A) <= 1
P(not A) = 1 - P(A)
P(certain event) = 1
P(impossible event) = 0
```

### Example: Coin Flip
```
Fair coin:
P(Heads) = 0.5
P(Tails) = 0.5
P(Heads) + P(Tails) = 1  (something must happen)
```

---

## Joint and Conditional Probability

**Joint Probability:** P(A and B) = P(A, B)
- Probability both A and B happen

**Conditional Probability:** P(A | B)
- Probability of A, given that B has happened

```
P(A | B) = P(A, B) / P(B)

"Of all the times B happens, how often does A also happen?"
```

### Example: Weather and Umbrella
```
P(Rain) = 0.3
P(Umbrella | Rain) = 0.9      (90% bring umbrella when raining)
P(Umbrella | No Rain) = 0.2   (20% bring umbrella anyway)

P(Rain and Umbrella) = P(Umbrella | Rain) × P(Rain)
                     = 0.9 × 0.3 = 0.27
```

---

## Independence

Events A and B are independent if knowing one tells you nothing about the other:

```
P(A | B) = P(A)
P(A, B) = P(A) × P(B)
```

**Example:** Two separate coin flips are independent.
```
P(Heads on flip 1 AND Heads on flip 2) = 0.5 × 0.5 = 0.25
```

**In ML:** We often assume training examples are independent (i.i.d. assumption).

---

## Bayes' Theorem: Updating Beliefs

This is one of the most important formulas in ML:

```
P(A | B) = P(B | A) × P(A) / P(B)

posterior = (likelihood × prior) / evidence
```

**What each term means:**
- **Prior P(A):** What you believed before seeing evidence
- **Likelihood P(B|A):** How likely is the evidence if A is true
- **Evidence P(B):** How likely is the evidence overall
- **Posterior P(A|B):** Updated belief after seeing evidence

### The Classic Medical Test Example

**Setup:**
- Disease prevalence: 1% of population has it
- Test sensitivity: 95% (correctly detects disease when present)
- Test specificity: 95% (correctly shows negative when no disease)

**Question:** You test positive. What's the probability you actually have the disease?

**Intuitive guess:** 95%?

**Let's calculate with Bayes:**
```
P(Disease | Positive) = P(Positive | Disease) × P(Disease) / P(Positive)

P(Positive) = P(Positive | Disease) × P(Disease) + P(Positive | No Disease) × P(No Disease)
            = 0.95 × 0.01 + 0.05 × 0.99
            = 0.0095 + 0.0495
            = 0.059

P(Disease | Positive) = 0.95 × 0.01 / 0.059 = 0.161
```

**Actual answer: Only 16%!**

### Why So Counterintuitive?

Using 10,000 people:
```
| Group           | Count | Test Positive | Test Negative |
|-----------------|-------|---------------|---------------|
| Have disease    | 100   | 95 (true +)   | 5             |
| No disease      | 9,900 | 495 (false +) | 9,405         |

Total positive: 95 + 495 = 590
Of those, actually sick: 95 / 590 = 16%
```

The **base rate** (1% prevalence) matters enormously. False positives from the large healthy population swamp true positives from the small sick population.

---

## Common Probability Distributions

### Bernoulli (Binary outcome)

Single trial with two outcomes (like a coin flip).

```python
from scipy.stats import bernoulli

# P(X=1) = p, P(X=0) = 1-p
samples = bernoulli.rvs(p=0.7, size=100)
# About 70 ones, 30 zeros
```

**Used for:** Binary classification probabilities

### Binomial (Count of successes)

Number of successes in n independent trials.

```python
from scipy.stats import binom

# 10 coin flips, P(heads)=0.5, probability of exactly 7 heads?
prob = binom.pmf(k=7, n=10, p=0.5)  # ~0.117
```

### Gaussian/Normal (Bell curve)

The most important distribution - appears everywhere due to Central Limit Theorem.

```python
from scipy.stats import norm

# Mean=0, Std=1 (standard normal)
samples = norm.rvs(loc=0, scale=1, size=1000)

# Probability density at x=0
density = norm.pdf(0, loc=0, scale=1)  # ~0.4

# Probability X < 1.96
prob = norm.cdf(1.96, loc=0, scale=1)  # ~0.975
```

```
The Bell Curve:
        ___
       /   \
      /     \    <- 68% within 1 std
     /       \   <- 95% within 2 std
    /         \  <- 99.7% within 3 std
___/           \___
   -3  -2  -1  0  1  2  3
```

**Used for:** Noise modeling, weight initialization, many natural phenomena

---

## Expected Value and Variance

### Expected Value (Mean)

The average outcome if you repeated the experiment infinitely.

```python
# Discrete: E[X] = sum of (x × P(x))
# Continuous: E[X] = integral of (x × p(x))

# For data:
E_X = np.mean(data)
```

### Variance

How spread out the distribution is.

```python
# Var[X] = E[(X - E[X])²] = E[X²] - E[X]²

variance = np.var(data)
std_dev = np.std(data)  # sqrt(variance)
```

```
Low variance:          High variance:
    ___                    _
   /   \                  / \
  /     \                /   \
_/       \_           __/     \__
Tight around mean      Spread out
```

---

## Maximum Likelihood Estimation (MLE)

Given data, find parameters that make the data most likely.

```
theta_ML = argmax P(data | theta)
         = argmax PRODUCT of P(x_i | theta)   (independent samples)
         = argmax SUM of log P(x_i | theta)   (log-likelihood, easier to optimize)
```

### Example: Estimating Coin Bias

Flip a coin 100 times, get 70 heads.

```
P(70 heads | p) = C(100,70) × p^70 × (1-p)^30

To maximize, take derivative and set to 0:
d/dp [log P] = 70/p - 30/(1-p) = 0
p = 70/100 = 0.7
```

**MLE estimate:** p = 0.7 (just the proportion of heads!)

### Connection to ML

When we minimize cross-entropy loss, we're doing MLE!

```
Minimize cross-entropy = Maximize log-likelihood
```

---

## Central Limit Theorem

**The most important theorem in statistics:**

The average of many independent random variables is approximately normally distributed, regardless of the original distribution.

```
Sample means
     |         ___
     |        /   \
     |       /     \   <- Approximately normal!
     |      /       \
     |_____/         \_____
            sample mean

Even if original data looks like:
     |
     |___
     |   |___
     |       |___
```

**Why it matters:**
- Explains why normal distribution appears everywhere
- Justifies many statistical tests
- Explains why batch averages in SGD are stable

---

## Hypothesis Testing (Intuition)

**Goal:** Determine if an observed effect is real or just random chance.

```
Null Hypothesis (H0): "No effect" or "Model is just guessing"
Alternative (H1): "There is an effect"

p-value: Probability of seeing this result (or more extreme) if H0 is true

p < 0.05 -> Reject H0 -> "Statistically significant"
p >= 0.05 -> Can't reject H0 -> "Not enough evidence"
```

**Example:** Is my model better than random?
```
If random accuracy = 50%, and my model gets 55% on 1000 samples:
- Could this happen by chance?
- p-value tells us the probability
- If p < 0.05, we conclude the model is genuinely better
```

**Caution:** p < 0.05 doesn't mean 95% probability the effect is real! It means there's only 5% chance of seeing this data if there's no effect.

---

## Correlation vs Causation

**Correlation:** Two variables tend to move together
```python
correlation = np.corrcoef(x, y)[0, 1]
# +1: perfect positive correlation
#  0: no correlation
# -1: perfect negative correlation
```

**Causation:** One variable actually causes the other

```
Correlation: Ice cream sales and drowning deaths are correlated
Causation: Ice cream does NOT cause drowning!
Hidden variable: Summer (hot weather) causes both
```

**In ML:** Models find correlations. They don't prove causation. Be careful with interpretations!

---

## Key Probability Concepts for ML

| Concept | ML Application |
|---------|----------------|
| Conditional Probability | P(label \| features) |
| Bayes' Theorem | Naive Bayes, Bayesian inference |
| Gaussian Distribution | Weight initialization, noise modeling |
| MLE | Training = maximizing likelihood |
| Independence | i.i.d. assumption for training data |
| Expected Value | Loss is expected error |

---

## Files

- `probability_statistics.py` - Distributions, sampling, and Bayes examples

## Key Takeaways

1. **Probability quantifies uncertainty** - essential for ML predictions
2. **Bayes' theorem updates beliefs** with new evidence (prior -> posterior)
3. **Base rates matter** - the disease testing paradox shows why
4. **Gaussian distribution** is central due to Central Limit Theorem
5. **MLE** = find parameters that maximize probability of data
6. **Correlation ≠ Causation** - ML finds patterns, not causes

## What's Next?

Step 4: **The ML Problem Setup** - framing learning problems correctly.
