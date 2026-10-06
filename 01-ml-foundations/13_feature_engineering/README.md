# Step 13: Feature Engineering

## Why Feature Engineering Matters

Feature engineering is often the **most impactful** part of machine learning. Good features can make a simple model outperform a complex one with poor features.

```
"Applied machine learning is basically feature engineering."
                                    - Andrew Ng

Same algorithm, different features:
- Bad features: 70% accuracy
- Good features: 95% accuracy
```

---

## Types of Features

| Type | Examples | Common Operations |
|------|----------|-------------------|
| Numerical | Age, price, temperature | Scaling, binning, transforms |
| Categorical | Color, country, type | Encoding |
| Text | Reviews, descriptions | Vectorization, embeddings |
| Temporal | Dates, timestamps | Extract components |
| Spatial | Latitude, longitude | Distance features |

---

## Part 1: Handling Missing Values

### Strategies

| Method | When to Use |
|--------|-------------|
| Drop rows | Few missing, random pattern |
| Drop column | >50% missing |
| Mean imputation | Numerical, Gaussian distribution |
| Median imputation | Numerical, with outliers |
| Mode imputation | Categorical |
| Forward/backward fill | Time series |
| Model-based (KNN) | Complex patterns |

### Implementation

```python
from sklearn.impute import SimpleImputer

# Numerical: median (robust to outliers)
imputer_num = SimpleImputer(strategy='median')
X_num = imputer_num.fit_transform(X_num)

# Categorical: most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
X_cat = imputer_cat.fit_transform(X_cat)
```

**Critical:** Fit imputer on train, transform both train and test!

---

## Part 2: Encoding Categorical Variables

### Label Encoding (Ordinal Features)

Use when categories have a natural order:

```python
# Size: Small < Medium < Large
size_mapping = {'small': 0, 'medium': 1, 'large': 2}
df['size_encoded'] = df['size'].map(size_mapping)
```

### One-Hot Encoding (Nominal Features)

Use when no natural order exists:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')
# drop='first' avoids dummy variable trap
X_encoded = encoder.fit_transform(df[['color']])
```

```
Color: [Red, Blue, Green]
Result: is_blue, is_green (red is reference)
```

### When to Use Which

| Encoding | Use Case |
|----------|----------|
| Label | Ordinal features (size, education level) |
| One-Hot | Nominal features (color, city) |
| Target | High cardinality (1000+ categories) |
| Binary | Two categories |

---

## Part 3: Feature Scaling

### Why Scale?

- Distance-based algorithms (KNN, SVM) require it
- Gradient descent converges faster
- Regularization treats features fairly

### Scaling Methods

| Scaler | Formula | When to Use |
|--------|---------|-------------|
| StandardScaler | (x - mean) / std | Default choice |
| MinMaxScaler | (x - min) / (max - min) | Need [0, 1] range |
| RobustScaler | (x - median) / IQR | Data has outliers |

### Implementation

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# Robust scaling (better with outliers)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

---

## Part 4: Creating New Features

### Mathematical Transformations

```python
# Log transform (reduces right skew)
df['log_income'] = np.log1p(df['income'])

# Square root
df['sqrt_area'] = np.sqrt(df['area'])

# Power transform (Box-Cox or Yeo-Johnson)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
X_transformed = pt.fit_transform(X)
```

### Ratio and Interaction Features

```python
# Ratios
df['price_per_sqft'] = df['price'] / df['sqft']
df['rooms_per_person'] = df['rooms'] / df['occupants']

# Interactions
df['area_x_quality'] = df['area'] * df['quality_score']
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Creates: x1, x2, x1^2, x2^2, x1*x2
```

### Binning (Discretization)

```python
# Age groups
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 18, 35, 50, 65, 100],
                         labels=['child', 'young', 'middle', 'senior', 'elderly'])
```

---

## Part 5: Date/Time Features

### Extract Components

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['hour'] = df['datetime'].dt.hour
```

### Cyclical Encoding

For periodic features (month, hour, day of week):

```python
# Month as sine/cosine
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# This makes January and December close together!
```

---

## Part 6: Text Features

### Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000)
X_bow = vectorizer.fit_transform(texts)
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(texts)
```

### Simple Statistics

```python
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean([len(w) for w in x.split()])
)
```

---

## Part 7: Feature Selection

### Why Select Features?

- Remove noise (irrelevant features)
- Reduce overfitting
- Speed up training
- Improve interpretability

### Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| SelectKBest | Filter | Fast | Ignores feature interactions |
| RFE | Wrapper | Considers interactions | Slow |
| Lasso (L1) | Embedded | Built into training | Linear assumption |
| Tree importance | Embedded | Handles non-linear | Biased to high cardinality |

### Implementation

```python
# Filter method
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper method
from sklearn.feature_selection import RFE
rfe = RFE(estimator, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Tree importance
rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
```

---

## Part 8: Handling Outliers

### Detection Methods

**Z-Score Method:**
```python
z_scores = (X - X.mean()) / X.std()
outliers = np.abs(z_scores) > 3
```

**IQR Method:**
```python
Q1 = np.percentile(X, 25)
Q3 = np.percentile(X, 75)
IQR = Q3 - Q1
outliers = (X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR)
```

### Treatment Options

| Treatment | When to Use |
|-----------|-------------|
| Remove | Few outliers, likely errors |
| Clip (winsorize) | Keep data points, limit effect |
| Transform (log) | Reduce impact naturally |
| Keep | Legitimate extreme values |

---

## Part 9: Building Pipelines

### Why Use Pipelines?

- Prevents data leakage
- Reproducible preprocessing
- Clean code
- Easy deployment

### Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Use it
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

---

## Common Mistakes

1. **Data leakage:** Using test data info in feature creation
2. **Fit on all data:** Must fit transformers on train only
3. **Ignoring domain knowledge:** Best features often come from expertise
4. **Over-engineering:** Too many features lead to overfitting
5. **Not validating:** Always check if new features actually help

---

## Feature Engineering Checklist

- [ ] Handle missing values (imputation strategy based on data type)
- [ ] Encode categorical variables (ordinal vs nominal)
- [ ] Scale numerical features (if using distance-based algorithms)
- [ ] Create domain-specific features
- [ ] Handle datetime features
- [ ] Detect and treat outliers
- [ ] Select relevant features
- [ ] Build reproducible pipeline
- [ ] Validate improvement with cross-validation

---

## Files

- `feature_engineering.py` - Complete examples of all techniques

## Key Takeaways

1. **Feature engineering often beats algorithm tuning**
2. **Handle missing values** before modeling
3. **Encode categoricals** appropriately (ordinal vs nominal)
4. **Scale features** for distance-based algorithms
5. **Create meaningful features** using domain knowledge
6. **Select relevant features** to reduce noise
7. **Use pipelines** for reproducible, leak-free preprocessing
8. **Fit on train, transform both** - never leak test data

## What's Next?

Step 14: **ML Project Workflow** - putting it all together in a complete machine learning project.
