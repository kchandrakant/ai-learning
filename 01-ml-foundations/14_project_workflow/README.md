# Step 14: ML Project Workflow

## End-to-End Process

```
1. Define Problem → 2. Get Data → 3. Explore Data → 4. Prepare Data
→ 5. Model Selection → 6. Train & Tune → 7. Evaluate → 8. Deploy
```

## Step 1: Define the Problem

- What are you trying to predict/find?
- Classification, regression, clustering?
- What metric defines success?
- What's the baseline to beat?

## Step 2: Get Data

```python
# From files
df = pd.read_csv('data.csv')

# From databases
df = pd.read_sql(query, connection)

# From APIs
response = requests.get(api_url)
```

## Step 3: Exploratory Data Analysis (EDA)

```python
# Basic info
df.info()
df.describe()
df.head()

# Missing values
df.isnull().sum()

# Distributions
df['feature'].hist()

# Correlations
df.corr()

# Target distribution (classification)
df['target'].value_counts()
```

## Step 4: Data Preparation

```python
# Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Then fit preprocessing on training only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform, not fit_transform!
```

## Step 5: Baseline Model

Always start simple!

```python
# Classification baseline
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f"Baseline accuracy: {dummy.score(X_test, y_test):.3f}")

# Regression baseline
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean')
```

## Step 6: Train & Tune

```python
# Try multiple models
models = {
    'Logistic': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

# Tune best model
param_grid = {...}
grid = GridSearchCV(best_model, param_grid, cv=5)
grid.fit(X_train, y_train)
```

## Step 7: Final Evaluation

```python
# Only now use test set!
final_model = grid.best_estimator_
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

## Step 8: Document & Deploy

```python
# Save model
import joblib
joblib.dump(final_model, 'model.pkl')

# Save preprocessing
joblib.dump(scaler, 'scaler.pkl')

# Load for inference
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
```

## Common Mistakes to Avoid

1. **Data leakage:** Fitting scaler on all data, not just training
2. **Using test set for tuning:** Overfits to test set
3. **Ignoring baseline:** Complex model might not beat simple
4. **Not versioning:** Can't reproduce results
5. **Skipping EDA:** Garbage in, garbage out

## Project Checklist

- [ ] Problem clearly defined
- [ ] Data understood (EDA done)
- [ ] Train/val/test split before preprocessing
- [ ] Baseline established
- [ ] Multiple models compared
- [ ] Best model tuned
- [ ] Final evaluation on held-out test
- [ ] Model and preprocessing saved
- [ ] Results documented

## Files

- `project_workflow.py` - Complete example workflow

## Congratulations!

You've completed the ML Foundations course. You're ready for:
- **Deep Learning to Transformers** → Neural networks
- **Applied ML projects** → Real-world problems
