"""
Step 12: Model Evaluation
=========================

This module covers:
1. Classification metrics (accuracy, precision, recall, F1)
2. Confusion matrix analysis
3. ROC curve and AUC
4. Precision-Recall curve
5. Regression metrics (MSE, RMSE, MAE, R2)
6. Cross-validation
7. Learning curves
8. Hyperparameter tuning

Run this file to see all concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     learning_curve, GridSearchCV,
                                     StratifiedKFold)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score, roc_auc_score,
                            mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(42)

print("=" * 60)
print("STEP 12: MODEL EVALUATION")
print("=" * 60)


# =============================================================================
# PART 1: Classification Metrics Basics
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: Classification Metrics Basics")
print("=" * 60)

# Create a balanced dataset
X_balanced, y_balanced = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Train a simple model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Basic Classification Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")


# =============================================================================
# PART 2: Confusion Matrix
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("""
         Predicted
         Neg   Pos
Actual Neg [ TN   FP ]
       Pos [ FN   TP ]
""")

# Extract components
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")

# Calculate metrics from confusion matrix
print("\n--- Metrics from Confusion Matrix ---")
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0

print(f"Accuracy:  {accuracy_manual:.4f}")
print(f"Precision: {precision_manual:.4f}")
print(f"Recall:    {recall_manual:.4f}")
print(f"F1 Score:  {f1_manual:.4f}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', 
                      color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=20)

plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: confusion_matrix.png")


# =============================================================================
# PART 3: Imbalanced Data - Why Accuracy Fails
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Imbalanced Data - Why Accuracy Fails")
print("=" * 60)

# Create highly imbalanced dataset (95% class 0, 5% class 1)
X_imb, y_imb = make_classification(
    n_samples=2000, n_features=20, n_informative=10,
    n_redundant=5, weights=[0.95, 0.05], random_state=42
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42
)

print(f"\nClass distribution in test set:")
print(f"  Class 0: {sum(y_test_imb == 0)} ({sum(y_test_imb == 0)/len(y_test_imb)*100:.1f}%)")
print(f"  Class 1: {sum(y_test_imb == 1)} ({sum(y_test_imb == 1)/len(y_test_imb)*100:.1f}%)")

# Naive model: always predict majority class
y_pred_naive = np.zeros_like(y_test_imb)

print("\n--- Naive Model (Always Predict 0) ---")
print(f"Accuracy:  {accuracy_score(y_test_imb, y_pred_naive):.4f} <- Looks great!")
print(f"Precision: {precision_score(y_test_imb, y_pred_naive, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test_imb, y_pred_naive):.4f} <- Catches nothing!")
print(f"F1 Score:  {f1_score(y_test_imb, y_pred_naive):.4f}")

# Train actual model
model_imb = LogisticRegression(random_state=42, max_iter=1000)
model_imb.fit(X_train_imb, y_train_imb)
y_pred_imb = model_imb.predict(X_test_imb)

print("\n--- Logistic Regression Model ---")
print(f"Accuracy:  {accuracy_score(y_test_imb, y_pred_imb):.4f}")
print(f"Precision: {precision_score(y_test_imb, y_pred_imb):.4f}")
print(f"Recall:    {recall_score(y_test_imb, y_pred_imb):.4f}")
print(f"F1 Score:  {f1_score(y_test_imb, y_pred_imb):.4f}")

print("\nLesson: Accuracy is misleading on imbalanced data!")
print("Use Precision, Recall, F1, or AUC-PR instead.")


# =============================================================================
# PART 4: Precision-Recall Tradeoff
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Precision-Recall Tradeoff")
print("=" * 60)

# Get probabilities
y_proba = model_imb.predict_proba(X_test_imb)[:, 1]

print("\n--- Effect of Threshold on Metrics ---")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 48)

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    prec = precision_score(y_test_imb, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test_imb, y_pred_thresh)
    f1 = f1_score(y_test_imb, y_pred_thresh)
    print(f"{threshold:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

print("\nLower threshold -> Higher recall, Lower precision")
print("Higher threshold -> Higher precision, Lower recall")


# =============================================================================
# PART 5: ROC Curve and AUC
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: ROC Curve and AUC")
print("=" * 60)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_imb, y_proba)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC Score: {roc_auc:.4f}")
print("  1.0 = Perfect classifier")
print("  0.5 = Random guessing")
print("  <0.5 = Worse than random")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: roc_curve.png")


# =============================================================================
# PART 6: Precision-Recall Curve
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Precision-Recall Curve")
print("=" * 60)

# Calculate PR curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test_imb, y_proba)
avg_precision = average_precision_score(y_test_imb, y_proba)

print(f"\nAverage Precision Score: {avg_precision:.4f}")
print("PR curve is better than ROC for imbalanced data!")

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, 
         label=f'PR Curve (AP = {avg_precision:.3f})')
plt.axhline(y=sum(y_test_imb)/len(y_test_imb), color='gray', linestyle='--',
           label=f'Random ({sum(y_test_imb)/len(y_test_imb):.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: pr_curve.png")


# =============================================================================
# PART 7: Comparing Multiple Models
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Comparing Multiple Models")
print("=" * 60)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Scale data for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

print("\n--- Model Comparison ---")
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 73)

for name, clf in models.items():
    if 'KNN' in name:
        clf.fit(X_train_scaled, y_train)
        y_pred_model = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred_model = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred_model)
    prec = precision_score(y_test, y_pred_model)
    rec = recall_score(y_test, y_pred_model)
    f1 = f1_score(y_test, y_pred_model)
    
    print(f"{name:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")


# =============================================================================
# PART 8: Regression Metrics
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Regression Metrics")
print("=" * 60)

# Create regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

# Calculate metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n--- Regression Metrics ---")
print(f"MSE (Mean Squared Error):     {mse:.4f}")
print(f"RMSE (Root MSE):              {rmse:.4f}")
print(f"MAE (Mean Absolute Error):    {mae:.4f}")
print(f"R2 (Coefficient of Determination): {r2:.4f}")

print("\nInterpretation:")
print(f"  RMSE: On average, predictions are off by {rmse:.2f} units")
print(f"  R2: Model explains {r2*100:.1f}% of variance in target")

# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 
         'r--', lw=2, label='Perfect predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Regression: Actual vs Predicted (R2 = {r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_predictions.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: regression_predictions.png")


# =============================================================================
# PART 9: Cross-Validation
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: Cross-Validation")
print("=" * 60)

print("\n--- K-Fold Cross-Validation ---")

# Simple cross-validation
cv_scores = cross_val_score(
    LogisticRegression(random_state=42, max_iter=1000),
    X_balanced, y_balanced, cv=5, scoring='accuracy'
)

print(f"\n5-Fold CV Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Compare different metrics
print("\n--- Cross-Validation with Different Metrics ---")
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in metrics:
    scores = cross_val_score(
        LogisticRegression(random_state=42, max_iter=1000),
        X_balanced, y_balanced, cv=5, scoring=metric
    )
    print(f"{metric:>12}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")


# =============================================================================
# PART 10: Learning Curves
# =============================================================================

print("\n" + "=" * 60)
print("PART 10: Learning Curves")
print("=" * 60)

print("\nGenerating learning curves (this may take a moment)...")

train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(random_state=42, max_iter=1000),
    X_balanced, y_balanced,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: learning_curve.png")

print("\nHow to interpret learning curves:")
print("  - Large gap = Overfitting (high variance)")
print("  - Both low = Underfitting (high bias)")
print("  - Converging = Good fit")


# =============================================================================
# PART 11: Overfitting Detection
# =============================================================================

print("\n" + "=" * 60)
print("PART 11: Overfitting Detection")
print("=" * 60)

print("\n--- Comparing Train vs Test Performance ---")

models_overfit = {
    'Decision Tree (no limit)': DecisionTreeClassifier(random_state=42),
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f"\n{'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'Diagnosis'}")
print("-" * 70)

for name, clf in models_overfit.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    # Diagnose
    gap = train_acc - test_acc
    if gap > 0.15:
        diagnosis = "OVERFITTING"
    elif train_acc < 0.7:
        diagnosis = "Underfitting"
    else:
        diagnosis = "Good fit"
    
    print(f"{name:<30} {train_acc:<12.4f} {test_acc:<12.4f} {diagnosis}")


# =============================================================================
# PART 12: Hyperparameter Tuning with Grid Search
# =============================================================================

print("\n" + "=" * 60)
print("PART 12: Hyperparameter Tuning")
print("=" * 60)

print("\n--- Grid Search CV ---")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
print(f"Test F1 score: {f1_score(y_test, grid_search.predict(X_test)):.4f}")


# =============================================================================
# PART 13: Classification Report
# =============================================================================

print("\n" + "=" * 60)
print("PART 13: Full Classification Report")
print("=" * 60)

# Use best model from grid search
y_pred_best = grid_search.predict(X_test)

print("\n--- Sklearn Classification Report ---")
print(classification_report(y_test, y_pred_best, target_names=['Class 0', 'Class 1']))


# =============================================================================
# PART 14: Metric Selection Guide
# =============================================================================

print("\n" + "=" * 60)
print("PART 14: Metric Selection Guide")
print("=" * 60)

print("""
CLASSIFICATION METRICS GUIDE:

| Scenario                    | Primary Metric  | Why                              |
|-----------------------------|-----------------|----------------------------------|
| Balanced classes            | Accuracy, F1    | All errors matter equally        |
| Imbalanced (care about pos) | Recall, F1      | Don't want to miss positives     |
| False positives costly      | Precision       | e.g., Spam filter                |
| False negatives costly      | Recall          | e.g., Cancer detection           |
| Ranking/probability         | ROC-AUC         | Threshold-independent            |
| Imbalanced + ranking        | PR-AUC          | Better than ROC for imbalanced   |

REGRESSION METRICS GUIDE:

| Scenario                    | Primary Metric  | Why                              |
|-----------------------------|-----------------|----------------------------------|
| Standard case               | RMSE, R2        | Penalizes large errors           |
| Outliers present            | MAE             | Robust to outliers               |
| Relative error matters      | MAPE            | Percentage-based                 |
| Model comparison            | R2              | Normalized, easy to interpret    |
""")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. ACCURACY IS NOT ENOUGH
   - Misleading on imbalanced data
   - Use precision, recall, F1 for classification
   - Consider business costs of different errors

2. PRECISION vs RECALL TRADEOFF
   - Can't maximize both
   - Adjust threshold based on use case
   - F1 balances both

3. ROC vs PR CURVES
   - ROC for balanced data and model comparison
   - PR curve better for imbalanced data
   - AUC summarizes overall performance

4. CROSS-VALIDATION
   - More reliable than single split
   - Get uncertainty estimates (std dev)
   - Use stratified for classification

5. OVERFITTING DETECTION
   - Compare train vs test performance
   - Large gap = overfitting
   - Learning curves show the trend

6. REGRESSION METRICS
   - RMSE: standard, penalizes large errors
   - MAE: robust to outliers
   - R2: proportion of variance explained

7. ALWAYS CONSIDER CONTEXT
   - What are the costs of different errors?
   - What metric aligns with business goals?
   - Technical best != business best
""")

print("\n" + "=" * 60)
print("Module 12 Complete!")
print("=" * 60)
