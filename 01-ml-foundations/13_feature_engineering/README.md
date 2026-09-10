# Step 13: Feature Engineering

## Features Often Matter More Than Algorithms

Good features > fancy model.

## Handling Missing Values

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Check for missing
df.isnull().sum()

# Strategies:
# 1. Drop rows
df.dropna()

# 2. Fill with mean/median/mode
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

# 3. Fill with indicator
df['feature_missing'] = df['feature'].isnull().astype(int)
df['feature'].fillna(df['feature'].median(), inplace=True)
```

## Encoding Categorical Variables

### One-Hot Encoding

```python
# Category → binary columns
pd.get_dummies(df, columns=['color'])

# color: red, blue, green
# becomes: color_red, color_blue, color_green
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
# red=0, blue=1, green=2
```

**Use one-hot for nominal (no order), label for ordinal.**

### Target Encoding

Replace category with mean of target.

```python
means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(means)
```

## Feature Scaling

### Standardization (Z-score)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Mean = 0, Std = 1
```

### Normalization (Min-Max)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Range [0, 1]
```

**When to scale:**
- Gradient-based methods (neural nets, SVM)
- Distance-based methods (KNN, clustering)

**Trees don't need scaling!**

## Creating New Features

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# x1, x2 → x1, x2, x1², x2², x1×x2
```

### Date/Time Features

```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

### Domain-Specific

```python
# E-commerce: price_per_unit = total_price / quantity
# Finance: debt_to_income = debt / income
# Text: word_count, avg_word_length
```

## Feature Selection

### Remove Low Variance

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

### Select K Best

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### Feature Importance from Models

```python
# From Random Forest
importances = rf.feature_importances_

# From Lasso (non-zero coefficients)
selected = lasso.coef_ != 0
```

## Files

- `feature_engineering.py` - Feature engineering utilities

## Key Takeaways

1. Handle missing values (drop or impute)
2. Encode categoricals appropriately
3. Scale for gradient/distance methods
4. Create domain-specific features
5. Select important features, remove noise

## What's Next?

Step 14: **ML Project Workflow** — end-to-end process.
