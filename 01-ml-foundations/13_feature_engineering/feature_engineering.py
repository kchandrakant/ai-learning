"""
Step 13: Feature Engineering
============================

This module covers:
1. Handling missing values
2. Encoding categorical variables
3. Feature scaling
4. Creating new features
5. Feature selection
6. Handling outliers
7. Building preprocessing pipelines

Run this file to see all concepts in action!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, LabelEncoder, PolynomialFeatures,
                                   PowerTransformer)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set random seed
np.random.seed(42)

print("=" * 60)
print("STEP 13: FEATURE ENGINEERING")
print("=" * 60)


# =============================================================================
# PART 1: Handling Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: Handling Missing Values")
print("=" * 60)

# Create sample data with missing values
np.random.seed(42)
n_samples = 100

data_missing = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples).astype(float),
    'income': np.random.randint(20000, 150000, n_samples).astype(float),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples)
})

# Introduce missing values
data_missing.loc[np.random.choice(n_samples, 15, replace=False), 'age'] = np.nan
data_missing.loc[np.random.choice(n_samples, 20, replace=False), 'income'] = np.nan
data_missing.loc[np.random.choice(n_samples, 10, replace=False), 'education'] = np.nan

print("\n--- Original Data with Missing Values ---")
print(f"Shape: {data_missing.shape}")
print(f"\nMissing values per column:")
print(data_missing.isnull().sum())

# Strategy 1: Mean/Median imputation for numerical
print("\n--- Numerical Imputation (Median) ---")
imputer_num = SimpleImputer(strategy='median')
data_missing[['age', 'income']] = imputer_num.fit_transform(data_missing[['age', 'income']])
print(f"Age missing after imputation: {data_missing['age'].isnull().sum()}")
print(f"Income missing after imputation: {data_missing['income'].isnull().sum()}")

# Strategy 2: Mode imputation for categorical
print("\n--- Categorical Imputation (Most Frequent) ---")
imputer_cat = SimpleImputer(strategy='most_frequent')
data_missing[['education']] = imputer_cat.fit_transform(data_missing[['education']])
print(f"Education missing after imputation: {data_missing['education'].isnull().sum()}")

print("\nImputation Strategies:")
print("  - Numerical: mean, median (robust to outliers), or model-based")
print("  - Categorical: most_frequent or constant value")


# =============================================================================
# PART 2: Encoding Categorical Variables
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Encoding Categorical Variables")
print("=" * 60)

# Create sample categorical data
categories = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'priority': ['low', 'medium', 'high', 'low', 'high']
})

print("\n--- Original Categorical Data ---")
print(categories)

# Label Encoding (for ordinal features)
print("\n--- Label Encoding (Ordinal: size) ---")
le = LabelEncoder()
size_encoded = le.fit_transform(categories['size'])
print(f"Original: {list(categories['size'])}")
print(f"Encoded:  {list(size_encoded)}")
print(f"Classes:  {list(le.classes_)}")

# Custom ordinal encoding with correct order
print("\n--- Custom Ordinal Encoding (with correct order) ---")
size_mapping = {'small': 0, 'medium': 1, 'large': 2}
size_ordinal = categories['size'].map(size_mapping)
print(f"Original: {list(categories['size'])}")
print(f"Encoded:  {list(size_ordinal)}")

# One-Hot Encoding (for nominal features)
print("\n--- One-Hot Encoding (Nominal: color) ---")
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy variable trap
color_encoded = ohe.fit_transform(categories[['color']])
print(f"Original: {list(categories['color'])}")
print(f"Encoded shape: {color_encoded.shape}")
print(f"Feature names: {ohe.get_feature_names_out(['color'])}")
print(f"Encoded:\n{color_encoded}")

# Pandas get_dummies (convenient alternative)
print("\n--- Pandas get_dummies ---")
color_dummies = pd.get_dummies(categories['color'], prefix='color', drop_first=True)
print(color_dummies)


# =============================================================================
# PART 3: Feature Scaling
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Feature Scaling")
print("=" * 60)

# Create data with different scales
np.random.seed(42)
data_scale = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),           # Range: 18-80
    'income': np.random.randint(20000, 200000, 100), # Range: 20k-200k
    'score': np.random.random(100) * 10              # Range: 0-10
})

# Add some outliers
data_scale.loc[0, 'income'] = 1000000  # Outlier

print("\n--- Original Data Statistics ---")
print(data_scale.describe().round(2))

# StandardScaler
print("\n--- StandardScaler (Z-score normalization) ---")
scaler_standard = StandardScaler()
data_standard = scaler_standard.fit_transform(data_scale)
print(f"Mean after scaling: {data_standard.mean(axis=0).round(4)}")
print(f"Std after scaling: {data_standard.std(axis=0).round(4)}")

# MinMaxScaler
print("\n--- MinMaxScaler (0-1 range) ---")
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data_scale)
print(f"Min after scaling: {data_minmax.min(axis=0).round(4)}")
print(f"Max after scaling: {data_minmax.max(axis=0).round(4)}")

# RobustScaler (handles outliers better)
print("\n--- RobustScaler (uses median and IQR) ---")
scaler_robust = RobustScaler()
data_robust = scaler_robust.fit_transform(data_scale)
print(f"Median after scaling: {np.median(data_robust, axis=0).round(4)}")

# Compare scalers on outlier
print("\n--- Effect on Outlier (income = 1,000,000) ---")
print(f"Original: {data_scale.loc[0, 'income']}")
print(f"StandardScaler: {data_standard[0, 1]:.4f}")
print(f"MinMaxScaler: {data_minmax[0, 1]:.4f}")
print(f"RobustScaler: {data_robust[0, 1]:.4f}")
print("\nRobustScaler is less affected by the outlier!")

# Visualize scaling comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].hist(data_scale['income'], bins=20, edgecolor='black')
axes[0].set_title('Original Income')

axes[1].hist(data_standard[:, 1], bins=20, edgecolor='black')
axes[1].set_title('StandardScaler')

axes[2].hist(data_minmax[:, 1], bins=20, edgecolor='black')
axes[2].set_title('MinMaxScaler')

axes[3].hist(data_robust[:, 1], bins=20, edgecolor='black')
axes[3].set_title('RobustScaler')

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: scaling_comparison.png")


# =============================================================================
# PART 4: Creating New Features
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Creating New Features")
print("=" * 60)

# Create housing-like dataset
np.random.seed(42)
housing = pd.DataFrame({
    'bedrooms': np.random.randint(1, 6, 100),
    'bathrooms': np.random.randint(1, 4, 100),
    'sqft': np.random.randint(500, 5000, 100),
    'price': np.random.randint(100000, 1000000, 100)
})

print("\n--- Original Features ---")
print(housing.head())

# Ratio features
print("\n--- Creating Ratio Features ---")
housing['price_per_sqft'] = housing['price'] / housing['sqft']
housing['rooms_total'] = housing['bedrooms'] + housing['bathrooms']
housing['sqft_per_room'] = housing['sqft'] / housing['rooms_total']

print(housing[['price_per_sqft', 'rooms_total', 'sqft_per_room']].head())

# Polynomial features
print("\n--- Polynomial Features ---")
X_simple = housing[['sqft', 'bedrooms']].values[:5]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_simple)

print(f"Original shape: {X_simple.shape}")
print(f"Polynomial shape: {X_poly.shape}")
print(f"Feature names: {poly.get_feature_names_out(['sqft', 'bedrooms'])}")
print(f"Original:\n{X_simple}")
print(f"Polynomial:\n{X_poly}")

# Log transformation for skewed data
print("\n--- Log Transformation ---")
skewed_data = np.array([1, 10, 100, 1000, 10000, 100000])
log_data = np.log1p(skewed_data)  # log1p = log(1+x), handles 0

print(f"Original: {skewed_data}")
print(f"Log transformed: {log_data.round(2)}")
print("Log transformation reduces skewness!")


# =============================================================================
# PART 5: Date/Time Features
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Date/Time Features")
print("=" * 60)

# Create sample datetime data
dates = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10, freq='D')
})

print("\n--- Original Dates ---")
print(dates)

# Extract components
print("\n--- Extracted Features ---")
dates['year'] = dates['date'].dt.year
dates['month'] = dates['date'].dt.month
dates['day'] = dates['date'].dt.day
dates['dayofweek'] = dates['date'].dt.dayofweek
dates['is_weekend'] = dates['dayofweek'].isin([5, 6]).astype(int)
dates['day_name'] = dates['date'].dt.day_name()

print(dates)

# Cyclical encoding for month
print("\n--- Cyclical Encoding for Month ---")
months = np.arange(1, 13)
month_sin = np.sin(2 * np.pi * months / 12)
month_cos = np.cos(2 * np.pi * months / 12)

print("Month | Sin    | Cos")
print("-" * 25)
for m, s, c in zip(months, month_sin, month_cos):
    print(f"  {m:2d}  | {s:6.3f} | {c:6.3f}")

print("\nCyclical encoding ensures Jan and Dec are close!")


# =============================================================================
# PART 6: Feature Selection
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Feature Selection")
print("=" * 60)

# Create dataset with informative and noise features
X, y = make_classification(
    n_samples=500, n_features=20, n_informative=5,
    n_redundant=5, n_clusters_per_class=2, random_state=42
)

feature_names = [f'feature_{i}' for i in range(20)]
X_df = pd.DataFrame(X, columns=feature_names)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print("5 informative, 5 redundant, 10 noise features")

# Method 1: SelectKBest (univariate filter)
print("\n--- SelectKBest (Filter Method) ---")
selector_kbest = SelectKBest(f_classif, k=10)
X_kbest = selector_kbest.fit_transform(X, y)
selected_mask = selector_kbest.get_support()
selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

# Method 2: Feature importance from Random Forest
print("\n--- Random Forest Feature Importance ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]
print("Top 10 features:")
for i in range(10):
    print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Method 3: RFE (Recursive Feature Elimination)
print("\n--- RFE (Wrapper Method) ---")
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
          n_features_to_select=10)
rfe.fit(X, y)
rfe_selected = [f for f, s in zip(feature_names, rfe.support_) if s]
print(f"RFE selected features: {rfe_selected[:5]}...")

# Method 4: L1 Regularization (Embedded)
print("\n--- Lasso (Embedded Method) ---")
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
lasso_selected = [f for f, c in zip(feature_names, lasso.coef_) if c != 0]
print(f"Lasso non-zero features: {len(lasso_selected)}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: feature_importance.png")


# =============================================================================
# PART 7: Handling Outliers
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Handling Outliers")
print("=" * 60)

# Create data with outliers
np.random.seed(42)
data_outliers = np.random.normal(50, 10, 100)
data_outliers = np.append(data_outliers, [150, 200, -50])  # Add outliers

print(f"\nData with outliers:")
print(f"  Min: {data_outliers.min():.2f}")
print(f"  Max: {data_outliers.max():.2f}")
print(f"  Mean: {data_outliers.mean():.2f}")
print(f"  Median: {np.median(data_outliers):.2f}")

# Z-score method
print("\n--- Z-Score Method (threshold = 3) ---")
z_scores = (data_outliers - data_outliers.mean()) / data_outliers.std()
outliers_zscore = np.abs(z_scores) > 3
print(f"Outliers detected: {outliers_zscore.sum()}")
print(f"Outlier values: {data_outliers[outliers_zscore]}")

# IQR method
print("\n--- IQR Method ---")
Q1 = np.percentile(data_outliers, 25)
Q3 = np.percentile(data_outliers, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = (data_outliers < lower_bound) | (data_outliers > upper_bound)
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Outliers detected: {outliers_iqr.sum()}")
print(f"Outlier values: {data_outliers[outliers_iqr]}")

# Treatment: Clipping
print("\n--- Treatment: Clipping (Winsorization) ---")
data_clipped = np.clip(data_outliers, lower_bound, upper_bound)
print(f"After clipping - Min: {data_clipped.min():.2f}, Max: {data_clipped.max():.2f}")


# =============================================================================
# PART 8: Building a Complete Pipeline
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Complete Preprocessing Pipeline")
print("=" * 60)

# Create a realistic mixed dataset
np.random.seed(42)
n = 500

data_mixed = pd.DataFrame({
    'age': np.random.randint(18, 70, n).astype(float),
    'income': np.random.randint(20000, 150000, n).astype(float),
    'years_employed': np.random.randint(0, 40, n).astype(float),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n),
})

# Add missing values
data_mixed.loc[np.random.choice(n, 50, replace=False), 'age'] = np.nan
data_mixed.loc[np.random.choice(n, 30, replace=False), 'income'] = np.nan
data_mixed.loc[np.random.choice(n, 20, replace=False), 'education'] = np.nan

# Create target
y_mixed = (data_mixed['income'].fillna(data_mixed['income'].median()) > 60000).astype(int)

print("\n--- Mixed Dataset ---")
print(f"Shape: {data_mixed.shape}")
print(f"\nMissing values:\n{data_mixed.isnull().sum()}")
print(f"\nData types:\n{data_mixed.dtypes}")

# Define column types
numeric_features = ['age', 'income', 'years_employed']
categorical_features = ['education', 'city']

# Create transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data_mixed, y_mixed, test_size=0.2, random_state=42
)

# Fit and evaluate
full_pipeline.fit(X_train, y_train)
train_score = full_pipeline.score(X_train, y_train)
test_score = full_pipeline.score(X_test, y_test)

print("\n--- Pipeline Results ---")
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(full_pipeline, data_mixed, y_mixed, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")


# =============================================================================
# PART 9: Impact of Feature Engineering
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: Impact of Feature Engineering")
print("=" * 60)

# Load California housing for realistic example
housing_data = fetch_california_housing()
X_housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y_housing = housing_data.target

print(f"\nCalifornia Housing Dataset: {X_housing.shape}")
print(f"Features: {list(X_housing.columns)}")

# Split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Baseline: No feature engineering
print("\n--- Baseline: Raw Features ---")
rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
rf_baseline.fit(X_train_h, y_train_h)
baseline_score = rf_baseline.score(X_test_h, y_test_h)
print(f"R2 Score: {baseline_score:.4f}")

# With feature engineering
print("\n--- With Feature Engineering ---")
X_train_fe = X_train_h.copy()
X_test_fe = X_test_h.copy()

# Create new features
for df in [X_train_fe, X_test_fe]:
    df['rooms_per_household'] = df['AveRooms']
    df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['HouseAge']
    df['income_per_room'] = df['MedInc'] / df['AveRooms']

# Log transform skewed features
for df in [X_train_fe, X_test_fe]:
    df['log_population'] = np.log1p(df['Population'])
    df['log_households'] = np.log1p(df['AveOccup'])

rf_engineered = RandomForestRegressor(n_estimators=100, random_state=42)
rf_engineered.fit(X_train_fe, y_train_h)
engineered_score = rf_engineered.score(X_test_fe, y_test_h)
print(f"R2 Score: {engineered_score:.4f}")

improvement = (engineered_score - baseline_score) / baseline_score * 100
print(f"\nImprovement: {improvement:.2f}%")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. MISSING VALUES
   - Numerical: median (robust) or mean
   - Categorical: most_frequent
   - Always fit on train, transform both

2. ENCODING
   - Ordinal: Label encoding with correct order
   - Nominal: One-hot encoding (drop first to avoid multicollinearity)
   - High cardinality: Target encoding (careful of leakage)

3. SCALING
   - StandardScaler: Default choice
   - MinMaxScaler: Need bounded range
   - RobustScaler: Data has outliers

4. FEATURE CREATION
   - Domain knowledge is key
   - Ratios, interactions, aggregations
   - Log transform for skewed data
   - Cyclical encoding for periodic features

5. FEATURE SELECTION
   - Filter: SelectKBest (fast, simple)
   - Wrapper: RFE (better but slower)
   - Embedded: L1 regularization, tree importance

6. OUTLIERS
   - Detection: Z-score or IQR method
   - Treatment: Remove, clip, or transform

7. PIPELINES
   - Use sklearn Pipeline and ColumnTransformer
   - Ensures no data leakage
   - Reproducible preprocessing

8. ALWAYS REMEMBER
   - Fit on train, transform both train and test
   - Feature engineering often beats algorithm tuning
   - Domain knowledge creates the best features
""")

print("\n" + "=" * 60)
print("Module 13 Complete!")
print("=" * 60)
