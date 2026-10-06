# Step 14: ML Project Workflow

## The Big Picture

This module ties everything together. ML isn't just about fitting models—it's a **disciplined process** from understanding the problem to monitoring deployed solutions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE ML PROJECT LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ PROBLEM  │───▶│   DATA   │───▶│   EDA    │───▶│ BASELINE │             │
│   │ FRAMING  │    │COLLECTION│    │          │    │  MODEL   │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │                                               │                    │
│        │                                               ▼                    │
│        │              ┌───────────────────────────────────────┐            │
│        │              │        ITERATION LOOP                 │            │
│        │              │                                       │            │
│        │              │  Error Analysis ──▶ Feature Eng ──┐   │            │
│        │              │       ▲                           │   │            │
│        │              │       └── Model Selection ◀───────┘   │            │
│        │              │                                       │            │
│        │              └───────────────────────────────────────┘            │
│        │                                               │                    │
│        ▼                                               ▼                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ SUCCESS  │◀───│ MONITOR  │◀───│  DEPLOY  │◀───│  FINAL   │             │
│   │ METRICS  │    │ & MAINT  │    │          │    │   EVAL   │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Problem Definition (Most Important!)

**80% of ML project failures trace back to poor problem definition.**

### Questions to Answer BEFORE Writing Code

| Question | Why It Matters |
|----------|----------------|
| What decision will this model inform? | ML should drive action, not just predict |
| What's the current baseline? | Know what you're trying to beat |
| How is success measured? | Business metric ≠ ML metric |
| What are the constraints? | Latency, interpretability, fairness |
| What data is realistically available? | At prediction time, not just training |

### The Problem Statement Template

```
We want to predict [TARGET]
for [WHO/WHAT]
using [AVAILABLE DATA]
to enable [BUSINESS DECISION]
with success measured by [METRIC]
```

**Example:**
> We want to predict **churn probability** for **existing customers** using **account history and usage patterns** to enable **targeted retention campaigns** with success measured by **recall > 70% at precision > 50%**.

### Common Mistake: Solving the Wrong Problem

```
WRONG PROBLEM                        RIGHT PROBLEM
═══════════════════════════════════════════════════════════
                                      
"Predict house prices"          →   "Help agents price listings  
                                     within 5% of sale price"
                                     
"Classify customer sentiment"   →   "Route complaints to human  
                                     agents when negative"
                                     
"Detect fraud"                  →   "Block high-risk transactions
                                     while minimizing false blocks"
                                     
The difference: RIGHT problems have clear ACTIONS attached
═══════════════════════════════════════════════════════════
```

---

## Phase 2: Data Collection & Understanding

### Data Quality Dimensions

```
THE 4 V's OF DATA QUALITY
═══════════════════════════════════════════════════════════

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   VOLUME    │  │  VARIETY    │  │  VERACITY   │  │  VELOCITY   │
│             │  │             │  │             │  │             │
│ Enough      │  │ All needed  │  │ Accurate?   │  │ Fresh       │
│ samples?    │  │ features?   │  │ Consistent? │  │ enough?     │
│             │  │             │  │             │  │             │
│ Balanced    │  │ All time    │  │ Trustworthy │  │ Available   │
│ classes?    │  │ periods?    │  │ labels?     │  │ at runtime? │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘

═══════════════════════════════════════════════════════════
```

### Data Leakage: The Silent Killer

**Data leakage** means your training data contains information that won't be available at prediction time.

```
LEAKAGE WARNING SIGNS
═══════════════════════════════════════════════════════════

1. TOO-GOOD-TO-BE-TRUE PERFORMANCE
   Train AUC: 0.99, Test AUC: 0.98 → Suspicious!
   
2. FEATURES FROM THE FUTURE
   Predicting purchase? Don't use "number of purchases"
   Predicting churn? Don't use "cancellation reason"
   
3. TARGET DERIVED FEATURES
   Predicting revenue? Don't use "revenue_per_user"
   
4. PREPROCESSING BEFORE SPLITTING
   Scaling on full dataset → Test set stats leak in
   
═══════════════════════════════════════════════════════════
```

---

## Phase 3: Exploratory Data Analysis (EDA)

### The EDA Checklist

```python
# SYSTEMATIC EDA APPROACH
# ========================

# 1. SHAPE & TYPES
df.shape                    # How many rows/columns?
df.info()                   # What are the dtypes?
df.head()                   # What does data look like?

# 2. MISSING VALUES
df.isnull().sum()           # Where are the gaps?
df.isnull().mean()          # What percentage missing?

# 3. TARGET ANALYSIS
df['target'].value_counts() # Classification: balanced?
df['target'].describe()     # Regression: distribution?

# 4. FEATURE DISTRIBUTIONS
df.describe()               # Numerical summaries
df.hist()                   # Visual distributions

# 5. RELATIONSHIPS
df.corr()['target']         # Correlations with target
df.groupby('cat_feature')['target'].mean()  # Group analysis

# 6. OUTLIERS
df.boxplot()                # Visual outlier detection
```

### EDA → Hypothesis Generation

EDA isn't just visualization—it's **forming hypotheses** about what will predict the target.

```
EDA FINDING                    HYPOTHESIS TO TEST
═══════════════════════════════════════════════════════════

"Churn higher for              → Contract type will be
 month-to-month contracts"       an important feature

"Monthly charges                → Non-linear relationship;
 show bimodal distribution"      consider binning or 
                                 polynomial features

"Support tickets                → Interaction term:
 correlate with both              tickets × tenure might
 tenure and churn"               be predictive

═══════════════════════════════════════════════════════════
```

---

## Phase 4: Baseline Model

### Why Baseline First?

```
THE BASELINE PRINCIPLE
═══════════════════════════════════════════════════════════

              Complex Model
                  │
                  │ 2% improvement... worth it?
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
Baseline at 85%            Baseline at 50%
Complex at 87%             Complex at 87%

Result: Not worth it       Result: Worth the complexity
        (diminishing returns)       (significant gain)

═══════════════════════════════════════════════════════════

BASELINE ALSO CATCHES BUGS:
If Random Forest < Logistic Regression < Dummy
→ Something is WRONG (data leakage, bug, or impossible problem)
```

### Baseline Options by Problem Type

| Problem | Baseline | Python |
|---------|----------|--------|
| Classification | Most frequent class | `DummyClassifier(strategy='most_frequent')` |
| Classification | Stratified random | `DummyClassifier(strategy='stratified')` |
| Regression | Predict mean | `DummyRegressor(strategy='mean')` |
| Regression | Predict median | `DummyRegressor(strategy='median')` |
| Time Series | Last value | `y_pred = y[-1]` |
| Time Series | Moving average | `y_pred = y[-k:].mean()` |

---

## Phase 5: The Iteration Loop

### Error Analysis Drives Improvement

```
ERROR ANALYSIS FRAMEWORK
═══════════════════════════════════════════════════════════

                    ┌─────────────────┐
                    │  MAKE PREDICTION │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ANALYZE ERRORS  │
                    │                 │
                    │ • False Positives│
                    │ • False Negatives│
                    │ • High error cases│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   HIGH   │  │   HIGH   │  │  PATTERN │
        │   BIAS   │  │ VARIANCE │  │  ERRORS  │
        │          │  │          │  │          │
        │Underfit  │  │ Overfit  │  │ Specific │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        More complex   Regularize    Engineer
        features or    or get        features for
        model          more data     those patterns

═══════════════════════════════════════════════════════════
```

### The Improvement Priority Ladder

```
IMPROVEMENT PRIORITY (Top = Highest Impact)
═══════════════════════════════════════════════════════════

     1. BETTER DATA
        ↓
     2. MORE DATA  
        ↓
     3. BETTER FEATURES
        ↓
     4. BETTER MODEL
        ↓
     5. BETTER HYPERPARAMETERS

Example: 
- Fixing a labeling error >>> tuning learning rate
- Adding a key feature >>> trying XGBoost vs Random Forest
- Getting 10x more data >>> elaborate feature engineering

═══════════════════════════════════════════════════════════
```

---

## Phase 6: Final Evaluation

### The Sacred Test Set

```
TEST SET RULES
═══════════════════════════════════════════════════════════

RULE 1: Touch it ONCE
        └─▶ Final evaluation only, after all decisions made

RULE 2: No peeking
        └─▶ Even "just checking" biases your decisions

RULE 3: Report honestly
        └─▶ Including failures and limitations

RULE 4: If you break rules 1-3
        └─▶ You need a NEW test set

WHY THIS MATTERS:
Each time you use test performance to make a decision,
you're effectively training on it. Your test score
becomes an overestimate of true performance.

═══════════════════════════════════════════════════════════
```

### Data Splitting Strategy

```
TYPICAL SPLIT STRATEGY
═══════════════════════════════════════════════════════════

Full Dataset (100%)
┌──────────────────────────────────────────────────────────┐
│██████████████████████████████████████████│░░░░░░░░░░░░░░│
└──────────────────────────────────────────────────────────┘
        Working Set (80%)                   Test (20%)
        (for development)                   (final eval)
        
        
Working Set Breakdown
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    K-FOLD CROSS VALIDATION                               │
│    ═══════════════════════                               │
│                                                          │
│    Fold 1: [VAL][ TRAIN  ][ TRAIN  ][ TRAIN  ][ TRAIN ] │
│    Fold 2: [TRAIN][VAL   ][ TRAIN  ][ TRAIN  ][ TRAIN ] │
│    Fold 3: [TRAIN][ TRAIN][  VAL   ][ TRAIN  ][ TRAIN ] │
│    Fold 4: [TRAIN][ TRAIN][ TRAIN  ][  VAL   ][ TRAIN ] │
│    Fold 5: [TRAIN][ TRAIN][ TRAIN  ][ TRAIN  ][  VAL  ] │
│                                                          │
│    Report: mean ± std across folds                       │
│                                                          │
└──────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════
```

---

## Phase 7 & 8: Deployment & Monitoring

### Deployment Checklist

```python
# DEPLOYMENT ARTIFACTS
# ====================

# 1. SAVE MODEL + PREPROCESSING TOGETHER
import joblib

artifacts = {
    'model': trained_model,
    'scaler': fitted_scaler,
    'encoder': fitted_encoder,
    'feature_names': feature_list,
    'version': '1.0.0',
    'trained_date': '2024-01-01',
    'performance': {'accuracy': 0.85, 'f1': 0.78}
}
joblib.dump(artifacts, 'model_artifacts.pkl')

# 2. CREATE PREDICTION FUNCTION
def predict(raw_input, artifacts):
    """Single entry point for predictions"""
    # Preprocess exactly as during training
    processed = preprocess(raw_input, artifacts)
    # Predict
    return artifacts['model'].predict(processed)
```

### Model Monitoring

```
WHAT TO MONITOR
═══════════════════════════════════════════════════════════

1. INPUT DRIFT (Data Changed)
   ┌─────────────────────────────────────────┐
   │ Training Distribution │ Production     │
   │     ███████          │     █████████  │
   │ Shifted! Alert!      │                │
   └─────────────────────────────────────────┘
   
2. OUTPUT DRIFT (Predictions Changed)
   ┌─────────────────────────────────────────┐
   │ Expected: 30% positive predictions      │
   │ Actual:   60% positive predictions      │
   │ Investigate!                            │
   └─────────────────────────────────────────┘
   
3. PERFORMANCE DEGRADATION
   ┌─────────────────────────────────────────┐
   │ Week 1: 85% accuracy                    │
   │ Week 4: 82% accuracy                    │
   │ Week 8: 75% accuracy  ← Retrain trigger│
   └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════
```

---

## Project Organization

### Recommended Directory Structure

```
my_ml_project/
│
├── data/
│   ├── raw/                 # Original, immutable data
│   ├── processed/           # Cleaned, transformed
│   └── external/            # Third-party data
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploration
│   ├── 02_baseline.ipynb    # Initial models
│   └── 03_experiments.ipynb # Model iteration
│
├── src/
│   ├── data/                # Data loading/processing
│   │   └── make_dataset.py
│   ├── features/            # Feature engineering
│   │   └── build_features.py
│   ├── models/              # Model training/prediction
│   │   ├── train.py
│   │   └── predict.py
│   └── evaluation/          # Metrics and analysis
│       └── evaluate.py
│
├── models/                  # Saved model artifacts
│   ├── model_v1.pkl
│   └── model_v2.pkl
│
├── reports/                 # Generated analysis
│   └── figures/
│
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── config.yaml              # Configuration
```

---

## Common Mistakes & How to Avoid Them

| Mistake | Why It's Bad | Prevention |
|---------|--------------|------------|
| **Data leakage** | Overestimates performance | Split before preprocessing; check feature availability |
| **Using test set for tuning** | No true held-out evaluation | Use CV for tuning, test ONCE at end |
| **Skipping baseline** | No reference for improvement | Always start simple |
| **Not versioning** | Can't reproduce results | Git for code, DVC for data/models |
| **Ignoring class imbalance** | Misleading accuracy | Use appropriate metrics (F1, PR-AUC) |
| **Overfitting to validation** | Poor generalization | Monitor train-val gap |
| **No error analysis** | Random improvements | Systematically analyze failures |

---

## Project Checklist

### Before Training
- [ ] Problem clearly defined with success metrics
- [ ] Data collected and understood (EDA done)
- [ ] Train/val/test split done BEFORE preprocessing
- [ ] Data leakage checked

### During Development
- [ ] Baseline established
- [ ] Multiple models compared via CV
- [ ] Error analysis performed
- [ ] Best model tuned
- [ ] Learning curves checked (overfitting?)

### Before Deployment
- [ ] Final evaluation on held-out test (once!)
- [ ] Model and preprocessing saved together
- [ ] Prediction function tested
- [ ] Model card/documentation written

### After Deployment
- [ ] Monitoring dashboards set up
- [ ] Alerting configured
- [ ] Retraining trigger defined
- [ ] A/B test framework ready

---

## Files

- `project_workflow.py` - Complete end-to-end churn prediction project demonstrating all phases

---

## 🎉 Congratulations!

You've completed the **ML Foundations** course! You now understand:

| Module | What You Learned |
|--------|-----------------|
| 1-3 | Mathematical foundations (linear algebra, calculus, probability) |
| 4 | ML problem framing and setup |
| 5-6 | Core algorithms (linear & logistic regression) |
| 7 | Optimization (gradient descent) |
| 8 | Regularization (L1, L2) |
| 9 | Trees & Ensembles (Random Forest, Gradient Boosting) |
| 10 | Support Vector Machines |
| 11 | Unsupervised Learning (Clustering, PCA) |
| 12 | Model Evaluation |
| 13 | Feature Engineering |
| 14 | End-to-End Project Workflow |

### What's Next?

**→ Deep Learning & Transformers** - Neural networks, CNNs, RNNs, attention, and modern LLMs

You have the foundations. Time to go deep! 🚀
