# Module 2: Bias & Fairness

## The Challenge

What does "fair" even mean? This module explores the uncomfortable truth that there are multiple valid definitions of fairness — and they often conflict.

## Types of Bias

### Data Bias
- **Historical bias**: Data reflects past discrimination
- **Representation bias**: Some groups underrepresented
- **Measurement bias**: Features measured differently across groups

### Algorithmic Bias
- Model amplifies patterns in biased data
- Proxy discrimination (using correlated features)
- Feedback loops that reinforce bias

### Societal Bias
- Deployed systems affect society
- Society affects future training data
- Cycle continues

## Fairness Definitions

### Statistical Parity (Demographic Parity)
Equal positive prediction rates across groups:
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
```
**Problem**: Ignores actual qualification rates

### Equalized Odds
Equal error rates across groups:
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (equal TPR)
P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (equal FPR)
```
**Problem**: Requires access to true labels

### Calibration
Predictions mean the same thing for all groups:
```
P(Y=1 | Ŷ=p, A=a) = p for all a
```
**Problem**: May result in different selection rates

### The Impossibility Result
**You cannot satisfy all three simultaneously** (unless base rates are equal).

This isn't a bug — it's mathematics.

## Measuring Bias

### Disparate Impact Ratio
```python
# 80% rule from employment law
ratio = P(positive | minority) / P(positive | majority)
# Should be >= 0.8
```

### Using Fairlearn

```python
from fairlearn.metrics import MetricFrame, selection_rate

metric_frame = MetricFrame(
    metrics={"selection_rate": selection_rate},
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)
print(metric_frame.by_group)
```

## Mitigation Strategies

### Pre-processing
- Resampling/reweighting training data
- Removing or transforming sensitive features

### In-processing
- Adversarial debiasing
- Constrained optimization
- Fair representations

### Post-processing
- Threshold adjustment
- Calibrated equalized odds

## Case Studies

### COMPAS (Criminal Justice)
- Predicted recidivism risk
- Found to have different error rates by race
- Sparked debate: which fairness definition matters?

### Amazon Hiring (Employment)
- Trained on historical hiring data
- Learned to penalize women's colleges
- Ultimately scrapped

### Healthcare Algorithms
- Used healthcare costs as proxy for health needs
- Black patients had lower costs due to access barriers
- Algorithm recommended less care for sicker patients

## The Tradeoff Question

For a hiring algorithm, which matters more?
- **Equal opportunity**: Same qualified candidates should have same chance
- **Equal outcomes**: Hire proportionally from all groups

There's no universal answer — it depends on values and context.

## Exercises

1. Audit a model using Fairlearn and identify disparities
2. Apply different mitigation strategies and compare tradeoffs
3. Debate which fairness definition applies to a given scenario

## Resources

- Fairlearn documentation: https://fairlearn.org
- "Fairness and Machine Learning" textbook: https://fairmlbook.org
- "Gender Shades" paper (Buolamwini & Gebru)

## What's Next?

Module 3 explores **Interpretability** — understanding why models make the decisions they do.
