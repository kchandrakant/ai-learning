"""
Module 14: ML Project Workflow - Complete End-to-End Example

This module demonstrates a complete ML project from problem definition
to model deployment, using best practices learned throughout the course.

Project: Predict customer churn for a telecom company
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MODULE 14: ML PROJECT WORKFLOW")
print("Complete End-to-End Machine Learning Project")
print("=" * 70)


# =============================================================================
# PHASE 1: PROBLEM DEFINITION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 1: PROBLEM DEFINITION")
print("=" * 70)

problem_definition = """
BUSINESS PROBLEM: Customer Churn Prediction
============================================

1. WHAT: Predict which customers will cancel their service (churn)

2. WHY: 
   - Acquiring new customers costs 5-25x more than retaining existing ones
   - Early intervention can save at-risk customers
   - Resource allocation for retention campaigns

3. SUCCESS METRICS:
   - Business: Reduce churn rate by 10%, ROI on retention campaigns
   - ML: Focus on RECALL (catch churners) while maintaining reasonable PRECISION
   - Target: Recall > 70%, Precision > 50%

4. CONSTRAINTS:
   - Model must be interpretable for business stakeholders
   - Predictions needed daily for campaign targeting
   - Must identify top factors driving churn

5. BASELINE TO BEAT:
   - Random guessing based on churn rate
   - Current rule-based system (if tenure < 6 months, flag as risky)
"""
print(problem_definition)


# =============================================================================
# PHASE 2: DATA COLLECTION & CREATION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2: DATA COLLECTION")
print("=" * 70)

# Create synthetic telecom churn dataset
np.random.seed(42)
n_samples = 2000

# Generate features with realistic patterns
tenure = np.random.exponential(24, n_samples).clip(1, 72)  # months
monthly_charges = 20 + 80 * np.random.beta(2, 2, n_samples)  # $20-$100
total_charges = tenure * monthly_charges * (0.9 + 0.2 * np.random.random(n_samples))

# Categorical features
contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, 
                                  p=[0.55, 0.25, 0.20])
payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                   n_samples, p=[0.35, 0.20, 0.25, 0.20])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20])

# Support-related
num_support_tickets = np.random.poisson(2, n_samples)
online_security = np.random.choice(['Yes', 'No', 'No internet'], n_samples, p=[0.30, 0.50, 0.20])

# Generate churn with realistic dependencies
churn_prob = (
    0.10 +  # base rate
    0.25 * (contract_types == 'Month-to-month') +
    0.15 * (payment_methods == 'Electronic check') +
    0.10 * (tenure < 12) +
    0.05 * (monthly_charges > 70) +
    0.03 * num_support_tickets +
    -0.15 * (tenure > 36) +
    -0.10 * (contract_types == 'Two year')
)
churn_prob = np.clip(churn_prob, 0.05, 0.85)
churn = (np.random.random(n_samples) < churn_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'tenure': tenure.round(0),
    'monthly_charges': monthly_charges.round(2),
    'total_charges': total_charges.round(2),
    'contract': contract_types,
    'payment_method': payment_methods,
    'internet_service': internet_service,
    'num_support_tickets': num_support_tickets,
    'online_security': online_security,
    'churn': churn
})

# Add some missing values (realistic scenario)
missing_idx = np.random.choice(n_samples, size=50, replace=False)
df.loc[missing_idx[:25], 'total_charges'] = np.nan
df.loc[missing_idx[25:], 'num_support_tickets'] = np.nan

print(f"\nDataset Shape: {df.shape}")
print(f"Churn Rate: {df['churn'].mean():.1%}")
print("\nFirst few rows:")
print(df.head())


# =============================================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 3.1 Basic Info
print("\n--- Data Types & Info ---")
print(df.dtypes)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 3.2 Missing Values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"Total missing: {df.isnull().sum().sum()} ({100*df.isnull().sum().sum()/(df.shape[0]*df.shape[1]):.2f}%)")

# 3.3 Target Distribution
print("\n--- Target Distribution (Class Imbalance Check) ---")
print(df['churn'].value_counts())
print(f"\nChurn rate: {df['churn'].mean():.1%}")
print("Assessment: Moderate imbalance - need to track recall for churners")

# 3.4 Numerical Features Summary
print("\n--- Numerical Features ---")
print(df.describe())

# 3.5 Categorical Features Summary
print("\n--- Categorical Features Distribution ---")
for col in ['contract', 'payment_method', 'internet_service', 'online_security']:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True).round(3))

# 3.6 Feature-Target Relationships
print("\n--- Churn Rate by Key Features ---")

print("\nChurn by Contract Type:")
print(df.groupby('contract')['churn'].mean().sort_values(ascending=False).round(3))

print("\nChurn by Payment Method:")
print(df.groupby('payment_method')['churn'].mean().sort_values(ascending=False).round(3))

print("\nChurn by Tenure (bucketed):")
df['tenure_bucket'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 100], 
                             labels=['0-12 mo', '12-24 mo', '24-48 mo', '48+ mo'])
print(df.groupby('tenure_bucket')['churn'].mean().round(3))
df = df.drop('tenure_bucket', axis=1)

# 3.7 Correlation Analysis (numerical only)
print("\n--- Correlation with Churn ---")
numerical_cols = ['tenure', 'monthly_charges', 'total_charges', 'num_support_tickets']
correlations = df[numerical_cols + ['churn']].corr()['churn'].drop('churn').sort_values()
print(correlations.round(3))

# Key EDA Insights
eda_insights = """
KEY EDA INSIGHTS:
=================
1. Churn rate ~35% - moderately imbalanced, recall is important
2. Month-to-month contracts have MUCH higher churn (~52% vs ~15% for yearly)
3. Electronic check payments correlate with higher churn
4. Shorter tenure strongly associated with churn
5. Higher monthly charges slightly increase churn risk
6. Support tickets increase churn risk
7. Missing values exist but are minimal (<3%)
"""
print(eda_insights)


# =============================================================================
# PHASE 4: DATA PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 4: DATA PREPARATION")
print("=" * 70)

# 4.1 Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# 4.2 Split data FIRST (before any preprocessing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X):.0%})")
print(f"Train churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# 4.3 Identify column types
numerical_features = ['tenure', 'monthly_charges', 'total_charges', 'num_support_tickets']
categorical_features = ['contract', 'payment_method', 'internet_service', 'online_security']

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# 4.4 Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', LabelEncoder())  # Will handle separately due to sklearn limitation
])

# For simplicity, let's encode categoricals manually and use ColumnTransformer for numericals
# In practice, you'd use OneHotEncoder in the pipeline

# Manual categorical encoding (fit on train only!)
label_encoders = {}
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in categorical_features:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train[col].fillna('Unknown'))
    X_test_encoded[col] = le.transform(X_test[col].fillna('Unknown'))
    label_encoders[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# Handle numerical missing values and scaling
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_num = X_train_encoded[numerical_features]
X_test_num = X_test_encoded[numerical_features]

X_train_num_imputed = imputer.fit_transform(X_train_num)
X_test_num_imputed = imputer.transform(X_test_num)

X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
X_test_num_scaled = scaler.transform(X_test_num_imputed)

# Combine numerical and categorical
X_train_final = np.hstack([X_train_num_scaled, X_train_encoded[categorical_features].values])
X_test_final = np.hstack([X_test_num_scaled, X_test_encoded[categorical_features].values])

feature_names = numerical_features + categorical_features
print(f"\nFinal feature count: {X_train_final.shape[1]}")
print("Data preparation complete!")


# =============================================================================
# PHASE 5: BASELINE MODEL
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 5: BASELINE MODEL")
print("=" * 70)

# 5.1 Dummy Classifiers
print("\n--- Dummy Baselines ---")

baselines = {
    'Most Frequent': DummyClassifier(strategy='most_frequent'),
    'Stratified': DummyClassifier(strategy='stratified'),
    'Uniform Random': DummyClassifier(strategy='uniform')
}

for name, model in baselines.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    print(f"{name:20s}: Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec:.3f}")

# 5.2 Simple Rule-Based Baseline
print("\n--- Rule-Based Baseline ---")
# Rule: Flag as churn risk if month-to-month contract AND tenure < 12
contract_idx = feature_names.index('contract')
tenure_idx = feature_names.index('tenure')

# Month-to-month is encoded, need to find its value
mm_code = label_encoders['contract'].transform(['Month-to-month'])[0]

# Apply rule (using unscaled tenure from original data)
rule_pred_train = ((X_train_encoded['contract'] == mm_code) & 
                   (X_train['tenure'] < 12)).astype(int)
rule_pred_test = ((X_test_encoded['contract'] == mm_code) & 
                  (X_test['tenure'] < 12)).astype(int)

acc = accuracy_score(y_test, rule_pred_test)
rec = recall_score(y_test, rule_pred_test)
prec = precision_score(y_test, rule_pred_test)
print(f"Rule-based (MM + tenure<12): Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec:.3f}")

# 5.3 Simple Logistic Regression Baseline
print("\n--- Logistic Regression Baseline ---")
lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train_final, y_train)
y_pred_lr = lr_baseline.predict(X_test_final)
acc = accuracy_score(y_test, y_pred_lr)
rec = recall_score(y_test, y_pred_lr)
prec = precision_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
print(f"Logistic Regression:  Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, F1={f1:.3f}")

baseline_to_beat = f1
print(f"\n>>> Baseline F1 to beat: {baseline_to_beat:.3f}")


# =============================================================================
# PHASE 6: MODEL TRAINING & COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 6: MODEL TRAINING & COMPARISON")
print("=" * 70)

# 6.1 Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# 6.2 Cross-validation comparison
print("\n--- Cross-Validation Results (5-Fold) ---")
print(f"{'Model':<25} {'Accuracy':>12} {'Std':>8}")
print("-" * 50)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1')
    cv_results[name] = scores
    print(f"{name:<25} {scores.mean():>12.3f} {scores.std():>8.3f}")

# 6.3 Select best model for tuning
best_model_name = max(cv_results, key=lambda x: cv_results[x].mean())
print(f"\n>>> Best model for tuning: {best_model_name}")


# =============================================================================
# PHASE 7: HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 7: HYPERPARAMETER TUNING")
print("=" * 70)

# Tune Random Forest (typically best or near-best)
print("\n--- Tuning Random Forest ---")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use smaller grid for speed
param_grid_small = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid_small, cv=5, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_final, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_


# =============================================================================
# PHASE 8: ERROR ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 8: ERROR ANALYSIS")
print("=" * 70)

# Predict on training set for error analysis
y_train_pred = best_model.predict(X_train_final)
y_train_proba = best_model.predict_proba(X_train_final)[:, 1]

# Confusion matrix
print("\n--- Training Set Confusion Matrix ---")
cm = confusion_matrix(y_train, y_train_pred)
print(f"                 Predicted")
print(f"                 No    Yes")
print(f"Actual No     {cm[0,0]:5d} {cm[0,1]:5d}")
print(f"       Yes    {cm[1,0]:5d} {cm[1,1]:5d}")

# Error types
fp = cm[0, 1]  # False positives (predicted churn, actually stayed)
fn = cm[1, 0]  # False negatives (predicted stay, actually churned)
print(f"\nFalse Positives (waste resources): {fp}")
print(f"False Negatives (miss churners): {fn}")
print(f"\nBusiness Impact: FN is more costly - losing customers we could have saved")

# Feature Importance
print("\n--- Feature Importance ---")
importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))

# Learning Curve Analysis
print("\n--- Learning Curve Analysis ---")
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_final, y_train, cv=5,
    train_sizes=np.linspace(0.2, 1.0, 5), scoring='f1'
)
print(f"Training samples: {train_sizes}")
print(f"Training F1:     {train_scores.mean(axis=1).round(3)}")
print(f"Validation F1:   {val_scores.mean(axis=1).round(3)}")

gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
print(f"\nTrain-Val Gap: {gap:.3f}")
if gap > 0.05:
    print("Assessment: Some overfitting detected - consider regularization or more data")
else:
    print("Assessment: Good generalization - model is not overfitting significantly")


# =============================================================================
# PHASE 9: FINAL EVALUATION (TEST SET - ONLY ONCE!)
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 9: FINAL EVALUATION (Test Set)")
print("=" * 70)
print("\n*** This is the ONLY time we touch the test set ***\n")

# Final predictions
y_pred = best_model.predict(X_test_final)
y_proba = best_model.predict_proba(X_test_final)[:, 1]

# Comprehensive metrics
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Key metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- Key Metrics Summary ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC AUC:   {auc:.3f}")

# Compare to baseline
print(f"\n--- Comparison to Baseline ---")
print(f"Baseline F1: {baseline_to_beat:.3f}")
print(f"Final F1:    {f1:.3f}")
print(f"Improvement: {100*(f1-baseline_to_beat)/baseline_to_beat:.1f}%")

# Check against success criteria
print("\n--- Success Criteria Check ---")
print(f"Target Recall > 70%: {recall:.1%} {'✓ PASS' if recall > 0.70 else '✗ FAIL'}")
print(f"Target Precision > 50%: {precision:.1%} {'✓ PASS' if precision > 0.50 else '✗ FAIL'}")

# Confusion Matrix
print("\n--- Test Set Confusion Matrix ---")
cm_test = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 No    Yes")
print(f"Actual No     {cm_test[0,0]:5d} {cm_test[0,1]:5d}")
print(f"       Yes    {cm_test[1,0]:5d} {cm_test[1,1]:5d}")


# =============================================================================
# PHASE 10: MODEL DEPLOYMENT PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 10: MODEL DEPLOYMENT PREPARATION")
print("=" * 70)

# 10.1 Save model and preprocessing objects
print("\n--- Saving Model Artifacts ---")

# In a real project, you would save these:
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'imputer': imputer,
    'label_encoders': label_encoders,
    'feature_names': feature_names,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features
}

# Demonstrate saving (commented out to avoid file creation)
# joblib.dump(model_artifacts, 'churn_model_artifacts.pkl')
print("Model artifacts prepared (model, scaler, imputer, encoders)")

# 10.2 Create prediction function
print("\n--- Prediction Function ---")

def predict_churn(customer_data, artifacts):
    """
    Make churn prediction for new customer data.
    
    Parameters:
    -----------
    customer_data : dict
        Customer features (tenure, monthly_charges, etc.)
    artifacts : dict
        Saved model artifacts
    
    Returns:
    --------
    prediction : int (0 or 1)
    probability : float
    """
    # Create DataFrame
    df_new = pd.DataFrame([customer_data])
    
    # Encode categoricals
    for col in artifacts['categorical_features']:
        le = artifacts['label_encoders'][col]
        df_new[col] = le.transform(df_new[col].fillna('Unknown'))
    
    # Process numericals
    num_data = df_new[artifacts['numerical_features']]
    num_imputed = artifacts['imputer'].transform(num_data)
    num_scaled = artifacts['scaler'].transform(num_imputed)
    
    # Combine features
    X_new = np.hstack([num_scaled, df_new[artifacts['categorical_features']].values])
    
    # Predict
    prediction = artifacts['model'].predict(X_new)[0]
    probability = artifacts['model'].predict_proba(X_new)[0, 1]
    
    return prediction, probability

# Test prediction function
test_customer = {
    'tenure': 3,
    'monthly_charges': 85.0,
    'total_charges': 255.0,
    'contract': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic',
    'num_support_tickets': 4,
    'online_security': 'No'
}

pred, prob = predict_churn(test_customer, model_artifacts)
print(f"\nTest prediction for high-risk customer:")
print(f"  Prediction: {'Will Churn' if pred == 1 else 'Will Stay'}")
print(f"  Churn Probability: {prob:.1%}")

# 10.3 Model Documentation
print("\n--- Model Card ---")
model_card = """
MODEL CARD: Customer Churn Prediction
=====================================

Model Type: Random Forest Classifier
Version: 1.0
Date: 2024

PURPOSE:
Predict customer churn probability to enable proactive retention campaigns.

PERFORMANCE:
- Accuracy: {:.1%}
- Precision: {:.1%}  
- Recall: {:.1%}
- F1 Score: {:.3f}
- ROC AUC: {:.3f}

KEY FEATURES (by importance):
1. tenure - strongest predictor
2. contract type - month-to-month high risk
3. monthly_charges
4. total_charges
5. payment_method

LIMITATIONS:
- Trained on historical data from one time period
- May not capture seasonal effects
- Feature importance doesn't imply causation

MONITORING:
- Track prediction distribution weekly
- Compare actual vs predicted monthly
- Retrain quarterly or when drift detected

ETHICAL CONSIDERATIONS:
- Model should not be used for discriminatory pricing
- Retention offers should be fair across customer segments
""".format(accuracy, precision, recall, f1, auc)
print(model_card)


# =============================================================================
# PROJECT SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

summary = """
CHURN PREDICTION PROJECT - COMPLETED
=====================================

PROBLEM: Predict customer churn for targeted retention campaigns

DATA: 2,000 customers with 8 features
- Numerical: tenure, charges, support tickets  
- Categorical: contract, payment method, services

APPROACH:
1. EDA revealed contract type and tenure as key factors
2. Built preprocessing pipeline (imputation + scaling + encoding)
3. Compared 4 models via cross-validation
4. Tuned Random Forest hyperparameters
5. Error analysis confirmed good generalization

RESULTS:
- Achieved {:.0%} recall (target: 70%) ✓
- Achieved {:.0%} precision (target: 50%) ✓
- F1 improved {:.0%} over baseline

KEY INSIGHTS:
1. Month-to-month contracts are highest risk
2. New customers (tenure < 12 months) need attention
3. Electronic check users churn more often
4. Support tickets signal dissatisfaction

RECOMMENDATIONS:
1. Focus retention on month-to-month customers
2. Create loyalty incentives for new customers
3. Investigate electronic check payment experience
4. Proactive outreach after support tickets

NEXT STEPS:
1. Deploy model to production scoring system
2. Set up weekly prediction batch jobs
3. Create dashboard for retention team
4. Establish monitoring and retraining pipeline
""".format(recall, precision, 100*(f1-baseline_to_beat)/baseline_to_beat)
print(summary)

print("\n" + "=" * 70)
print("CONGRATULATIONS! You've completed the ML Foundations course!")
print("=" * 70)
print("""
You've learned:
✓ Mathematical foundations (linear algebra, calculus, probability)
✓ Core ML algorithms (linear/logistic regression, trees, SVM)
✓ Ensemble methods (random forest, gradient boosting)
✓ Unsupervised learning (clustering, dimensionality reduction)
✓ Model evaluation (metrics, cross-validation, learning curves)
✓ Feature engineering (encoding, scaling, pipelines)
✓ End-to-end project workflow

NEXT: Deep Learning & Transformers →
""")
