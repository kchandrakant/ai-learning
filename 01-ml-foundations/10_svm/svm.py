"""
Step 10: Support Vector Machines (SVM)
======================================

This module covers:
1. Linear SVM and maximum margin concept
2. Soft margin and the C parameter
3. Kernel trick for non-linear boundaries
4. Different kernels (linear, RBF, polynomial)
5. Hyperparameter tuning
6. Multi-class classification

Run this file to see all concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("STEP 10: SUPPORT VECTOR MACHINES (SVM)")
print("=" * 60)


# =============================================================================
# PART 1: Linear SVM - Maximum Margin Concept
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: Linear SVM - Maximum Margin")
print("=" * 60)

# Create linearly separable data
X_linear, y_linear = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1,
    class_sep=2.0, random_state=42
)

# Train linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_linear, y_linear)

print(f"\nNumber of support vectors: {len(svm_linear.support_vectors_)}")
print(f"Support vectors per class: {svm_linear.n_support_}")


def plot_svm_decision_boundary(model, X, y, title, ax):
    """Plot SVM decision boundary with margins and support vectors."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Try to plot margin lines for linear kernel
    if hasattr(model, 'decision_function'):
        try:
            Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z_decision = Z_decision.reshape(xx.shape)
            ax.contour(xx, yy, Z_decision, colors='k', levels=[-1, 0, 1],
                      linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
        except:
            pass
    
    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    
    # Highlight support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
              s=200, facecolors='none', edgecolors='green', linewidths=2,
              label='Support Vectors')
    
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)


# Plot linear SVM
fig, ax = plt.subplots(figsize=(8, 6))
plot_svm_decision_boundary(svm_linear, X_linear, y_linear,
                           'Linear SVM: Maximum Margin Classifier', ax)
plt.tight_layout()
plt.savefig('svm_linear.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_linear.png")


# =============================================================================
# PART 2: The C Parameter - Soft Margin
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: The C Parameter (Soft Margin)")
print("=" * 60)

# Create data with some overlap
X_overlap, y_overlap = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1,
    class_sep=0.8, flip_y=0.1, random_state=42
)

print("\nEffect of C parameter:")
print("- Small C: Wide margin, more misclassifications allowed")
print("- Large C: Narrow margin, fewer misclassifications")

# Compare different C values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
C_values = [0.01, 1, 100]

for ax, C in zip(axes, C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_overlap, y_overlap)
    plot_svm_decision_boundary(svm, X_overlap, y_overlap,
                               f'C = {C}\n({len(svm.support_vectors_)} support vectors)', ax)

plt.suptitle('Effect of C Parameter on Decision Boundary', fontsize=14)
plt.tight_layout()
plt.savefig('svm_c_parameter.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_c_parameter.png")

# Quantitative comparison
print("\n--- C Parameter Comparison ---")
print(f"{'C':>10} {'Support Vectors':>20} {'Train Accuracy':>15}")
print("-" * 50)

for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_overlap, y_overlap)
    acc = svm.score(X_overlap, y_overlap)
    print(f"{C:>10} {len(svm.support_vectors_):>20} {acc:>15.4f}")


# =============================================================================
# PART 3: Non-Linear Data - The Need for Kernels
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Non-Linear Data - The Need for Kernels")
print("=" * 60)

# Create non-linearly separable data
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

print("\nLinear SVM fails on non-linear data:")

# Try linear SVM on circles
svm_linear_circles = SVC(kernel='linear', C=1)
svm_linear_circles.fit(X_circles, y_circles)
print(f"Linear SVM on circles: {svm_linear_circles.score(X_circles, y_circles):.4f}")

# RBF kernel works!
svm_rbf_circles = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf_circles.fit(X_circles, y_circles)
print(f"RBF SVM on circles: {svm_rbf_circles.score(X_circles, y_circles):.4f}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Circles dataset
plot_svm_decision_boundary(svm_linear_circles, X_circles, y_circles,
                           'Linear SVM on Circles (Fails)', axes[0, 0])
plot_svm_decision_boundary(svm_rbf_circles, X_circles, y_circles,
                           'RBF SVM on Circles (Works!)', axes[0, 1])

# Moons dataset
svm_linear_moons = SVC(kernel='linear', C=1)
svm_linear_moons.fit(X_moons, y_moons)

svm_rbf_moons = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf_moons.fit(X_moons, y_moons)

plot_svm_decision_boundary(svm_linear_moons, X_moons, y_moons,
                           'Linear SVM on Moons', axes[1, 0])
plot_svm_decision_boundary(svm_rbf_moons, X_moons, y_moons,
                           'RBF SVM on Moons', axes[1, 1])

plt.suptitle('Linear vs RBF Kernel on Non-Linear Data', fontsize=14)
plt.tight_layout()
plt.savefig('svm_kernel_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_kernel_comparison.png")


# =============================================================================
# PART 4: Different Kernels
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Different Kernels")
print("=" * 60)

print("\nCommon kernels:")
print("- Linear: K(x,y) = x.y")
print("- Polynomial: K(x,y) = (gamma*x.y + r)^d")
print("- RBF: K(x,y) = exp(-gamma*||x-y||^2)")
print("- Sigmoid: K(x,y) = tanh(gamma*x.y + r)")

# Compare kernels on moons dataset
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
kernels = [
    ('linear', {}),
    ('poly', {'degree': 3}),
    ('rbf', {}),
    ('sigmoid', {})
]

print("\n--- Kernel Comparison on Moons Data ---")
for ax, (kernel, params) in zip(axes.flat, kernels):
    svm = SVC(kernel=kernel, C=1, gamma='scale', **params)
    svm.fit(X_moons, y_moons)
    acc = svm.score(X_moons, y_moons)
    print(f"{kernel:>10} kernel: Accuracy = {acc:.4f}, Support Vectors = {len(svm.support_vectors_)}")
    plot_svm_decision_boundary(svm, X_moons, y_moons,
                               f'{kernel.capitalize()} Kernel (Acc: {acc:.2f})', ax)

plt.suptitle('Comparison of Different SVM Kernels', fontsize=14)
plt.tight_layout()
plt.savefig('svm_kernels.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_kernels.png")


# =============================================================================
# PART 5: The Gamma Parameter (RBF Kernel)
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: The Gamma Parameter")
print("=" * 60)

print("\nGamma controls the 'reach' of each training example:")
print("- Small gamma: Far reach, smooth boundary (underfitting)")
print("- Large gamma: Close reach, wiggly boundary (overfitting)")

# Compare gamma values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
gamma_values = [0.1, 1, 10, 100]

print("\n--- Gamma Comparison ---")
for ax, gamma in zip(axes, gamma_values):
    svm = SVC(kernel='rbf', C=1, gamma=gamma)
    svm.fit(X_moons, y_moons)
    acc = svm.score(X_moons, y_moons)
    print(f"gamma = {gamma:>5}: Accuracy = {acc:.4f}, Support Vectors = {len(svm.support_vectors_)}")
    plot_svm_decision_boundary(svm, X_moons, y_moons,
                               f'gamma = {gamma}', ax)

plt.suptitle('Effect of Gamma on RBF Kernel Decision Boundary', fontsize=14)
plt.tight_layout()
plt.savefig('svm_gamma.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_gamma.png")


# =============================================================================
# PART 6: Feature Scaling is Critical!
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Feature Scaling")
print("=" * 60)

# Create data with very different scales
X_unscaled = np.random.randn(200, 2)
X_unscaled[:, 0] *= 100  # Feature 0: range ~[-300, 300]
X_unscaled[:, 1] *= 1    # Feature 1: range ~[-3, 3]
y_scale_demo = (X_unscaled[:, 0] + X_unscaled[:, 1] * 50 > 0).astype(int)

# Split data
X_train_scale, X_test_scale, y_train_scale, y_test_scale = train_test_split(
    X_unscaled, y_scale_demo, test_size=0.3, random_state=42
)

# Without scaling
svm_unscaled = SVC(kernel='rbf', C=1, gamma='scale')
svm_unscaled.fit(X_train_scale, y_train_scale)
acc_unscaled = svm_unscaled.score(X_test_scale, y_test_scale)

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scale)
X_test_scaled = scaler.transform(X_test_scale)

svm_scaled = SVC(kernel='rbf', C=1, gamma='scale')
svm_scaled.fit(X_train_scaled, y_train_scale)
acc_scaled = svm_scaled.score(X_test_scaled, y_test_scale)

print("\n--- Effect of Feature Scaling ---")
print(f"Without scaling: {acc_unscaled:.4f}")
print(f"With scaling:    {acc_scaled:.4f}")
print("\nALWAYS scale your features before using SVM!")


# =============================================================================
# PART 7: Hyperparameter Tuning with Grid Search
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Hyperparameter Tuning")
print("=" * 60)

# Create a more realistic dataset
X_tune, y_tune = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, n_clusters_per_class=2, random_state=42
)

X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(
    X_tune, y_tune, test_size=0.2, random_state=42
)

# Scale features
scaler_tune = StandardScaler()
X_train_tune_scaled = scaler_tune.fit_transform(X_train_tune)
X_test_tune_scaled = scaler_tune.transform(X_test_tune)

# Grid search
print("\n--- Grid Search for SVM ---")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_tune_scaled, y_train_tune)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test_tune_scaled, y_test_tune):.4f}")


# =============================================================================
# PART 8: Multi-Class Classification
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Multi-Class Classification")
print("=" * 60)

# Create multi-class data
from sklearn.datasets import make_blobs

X_multi, y_multi = make_blobs(n_samples=300, centers=4, n_features=2,
                               cluster_std=1.5, random_state=42)

# Scale
scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

# Train multi-class SVM (uses OvO by default)
svm_multi = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
svm_multi.fit(X_multi_scaled, y_multi)

print(f"\nMulti-class SVM (4 classes):")
print(f"Training accuracy: {svm_multi.score(X_multi_scaled, y_multi):.4f}")
print(f"Number of support vectors per class: {svm_multi.n_support_}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
h = 0.02
x_min, x_max = X_multi_scaled[:, 0].min() - 1, X_multi_scaled[:, 0].max() + 1
y_min, y_max = X_multi_scaled[:, 1].min() - 1, X_multi_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_multi.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax.scatter(X_multi_scaled[:, 0], X_multi_scaled[:, 1], 
                     c=y_multi, cmap='viridis', edgecolors='black', s=50)
ax.set_title('Multi-Class SVM (4 Classes)')
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig('svm_multiclass.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: svm_multiclass.png")


# =============================================================================
# PART 9: SVM vs Other Classifiers
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: SVM vs Other Classifiers")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Use a larger dataset for comparison
X_compare, y_compare = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
    X_compare, y_compare, test_size=0.2, random_state=42
)

# Scale for SVM and KNN
scaler_cmp = StandardScaler()
X_train_cmp_scaled = scaler_cmp.fit_transform(X_train_cmp)
X_test_cmp_scaled = scaler_cmp.transform(X_test_cmp)

# Compare classifiers
classifiers = {
    'SVM (RBF)': SVC(kernel='rbf', C=1, gamma='scale'),
    'SVM (Linear)': SVC(kernel='linear', C=1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\n--- Classifier Comparison ---")
print(f"{'Classifier':<25} {'Train Acc':>12} {'Test Acc':>12}")
print("-" * 50)

for name, clf in classifiers.items():
    # Use scaled data for SVM and KNN, unscaled for others
    if 'SVM' in name or 'KNN' in name:
        clf.fit(X_train_cmp_scaled, y_train_cmp)
        train_acc = clf.score(X_train_cmp_scaled, y_train_cmp)
        test_acc = clf.score(X_test_cmp_scaled, y_test_cmp)
    else:
        clf.fit(X_train_cmp, y_train_cmp)
        train_acc = clf.score(X_train_cmp, y_train_cmp)
        test_acc = clf.score(X_test_cmp, y_test_cmp)
    
    print(f"{name:<25} {train_acc:>12.4f} {test_acc:>12.4f}")


# =============================================================================
# PART 10: When to Use SVM
# =============================================================================

print("\n" + "=" * 60)
print("PART 10: When to Use SVM")
print("=" * 60)

print("""
GOOD USE CASES:
- Medium-sized datasets (< 100k samples)
- High-dimensional data (text classification, genomics)
- When you need a clear margin of separation
- Binary classification or small multi-class problems

NOT IDEAL FOR:
- Very large datasets (slow: O(n^2) to O(n^3))
- When you need probability estimates
- Lots of noise with overlapping classes
- When interpretability is crucial

PRACTICAL TIPS:
1. ALWAYS scale your features (StandardScaler)
2. Start with RBF kernel for non-linear problems
3. Use LinearSVC for large datasets (faster)
4. Tune C and gamma with grid search
5. Use cross-validation to avoid overfitting
""")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. MAXIMUM MARGIN
   - SVM finds the boundary with largest margin
   - Support vectors are the critical points
   - More robust than arbitrary separators

2. SOFT MARGIN (C PARAMETER)
   - C controls margin width vs. errors tradeoff
   - Small C = wide margin, more errors allowed
   - Large C = narrow margin, few errors allowed

3. KERNEL TRICK
   - Enables non-linear boundaries
   - RBF is the go-to for non-linear problems
   - Linear kernel for high-dimensional sparse data

4. GAMMA PARAMETER
   - Controls reach of each training example
   - Small gamma = smooth boundary (underfit)
   - Large gamma = wiggly boundary (overfit)

5. CRITICAL: FEATURE SCALING
   - SVM uses distances, scales must be similar
   - Always use StandardScaler or similar
   - Fit on train, transform both train and test

6. HYPERPARAMETER TUNING
   - Grid search over C and gamma
   - Use cross-validation
   - Start with reasonable ranges
""")

print("\n" + "=" * 60)
print("Module 10 Complete!")
print("=" * 60)
