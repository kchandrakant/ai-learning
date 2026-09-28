"""
Step 9: Decision Trees & Ensemble Methods
=========================================

This module covers:
1. Decision Tree from scratch
2. Gini impurity and information gain
3. Random Forest
4. Gradient Boosting basics
5. XGBoost/LightGBM usage
6. Feature importance

Run this file to see all concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("STEP 9: DECISION TREES & ENSEMBLE METHODS")
print("=" * 60)


# =============================================================================
# PART 1: Decision Tree Fundamentals
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: Decision Tree Fundamentals")
print("=" * 60)


def gini_impurity(y):
    """
    Calculate Gini impurity of a node.
    
    Gini = 1 - sum(p_i^2) for all classes
    
    - Gini = 0: Pure node (all same class)
    - Gini = 0.5: Maximum impurity for binary (50/50 split)
    """
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    total = len(y)
    
    gini = 1.0
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    
    return gini


def entropy(y):
    """
    Calculate entropy of a node.
    
    Entropy = -sum(p_i * log2(p_i)) for all classes
    
    - Entropy = 0: Pure node
    - Entropy = 1: Maximum impurity for binary (50/50)
    """
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    total = len(y)
    
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * np.log2(p)
    
    return ent


# Demonstrate impurity measures
print("\n--- Impurity Measures ---")

# Pure node
pure = [1, 1, 1, 1, 1]
print(f"Pure node {pure}:")
print(f"  Gini: {gini_impurity(pure):.4f}")
print(f"  Entropy: {entropy(pure):.4f}")

# 50/50 split
mixed = [0, 0, 0, 1, 1, 1]
print(f"\n50/50 mixed {mixed}:")
print(f"  Gini: {gini_impurity(mixed):.4f}")
print(f"  Entropy: {entropy(mixed):.4f}")

# 70/30 split
partial = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
print(f"\n70/30 split {partial}:")
print(f"  Gini: {gini_impurity(partial):.4f}")
print(f"  Entropy: {entropy(partial):.4f}")


def information_gain(y, left_indices, right_indices, criterion='gini'):
    """
    Calculate information gain from a split.
    
    IG = parent_impurity - weighted_avg(child_impurities)
    """
    if criterion == 'gini':
        impurity_func = gini_impurity
    else:
        impurity_func = entropy
    
    parent_impurity = impurity_func(y)
    
    n = len(y)
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    left_impurity = impurity_func(y[left_indices])
    right_impurity = impurity_func(y[right_indices])
    
    weighted_child = (n_left / n) * left_impurity + (n_right / n) * right_impurity
    
    return parent_impurity - weighted_child


# =============================================================================
# PART 2: Decision Tree From Scratch
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Decision Tree From Scratch")
print("=" * 60)


class DecisionTreeNode:
    """A node in the decision tree."""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left            # Left child (feature <= threshold)
        self.right = right          # Right child (feature > threshold)
        self.value = value          # Prediction value (for leaf nodes)
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implemented from scratch.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        """Build the decision tree."""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        """Recursively grow the tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            return DecisionTreeNode(value=self._most_common_label(y))
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return DecisionTreeNode(value=self._most_common_label(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check min_samples_leaf
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            return DecisionTreeNode(value=self._most_common_label(y))
        
        # Recursively build children
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                left_indices = np.where(left_mask)[0]
                right_indices = np.where(~left_mask)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gain = information_gain(y, left_indices, right_indices)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _most_common_label(self, y):
        """Return the most common class label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# Test our decision tree
print("\n--- Testing Our Decision Tree ---")

# Create simple dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=200, n_features=4, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Split data
train_size = 150
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train our tree
tree_scratch = DecisionTreeClassifierScratch(max_depth=5, min_samples_leaf=5)
tree_scratch.fit(X_train, y_train)

# Predictions
y_pred_scratch = tree_scratch.predict(X_test)
accuracy_scratch = np.mean(y_pred_scratch == y_test)
print(f"Our Decision Tree Accuracy: {accuracy_scratch:.4f}")

# Compare with sklearn
from sklearn.tree import DecisionTreeClassifier

tree_sklearn = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
tree_sklearn.fit(X_train, y_train)
y_pred_sklearn = tree_sklearn.predict(X_test)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test)
print(f"Sklearn Decision Tree Accuracy: {accuracy_sklearn:.4f}")


# =============================================================================
# PART 3: Visualizing Decision Boundaries
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Visualizing Decision Boundaries")
print("=" * 60)


def plot_decision_boundary(model, X, y, title, ax):
    """Plot decision boundary for 2D data."""
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.set_title(title)


# Create 2D dataset for visualization
X_2d, y_2d = make_classification(
    n_samples=300, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Different tree depths
for ax, depth in zip(axes, [1, 3, 10]):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_2d, y_2d)
    plot_decision_boundary(tree, X_2d, y_2d, f'Depth = {depth}', ax)

plt.suptitle('Decision Tree: Effect of Max Depth on Decision Boundary')
plt.tight_layout()
plt.savefig('tree_depth_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: tree_depth_comparison.png")


# =============================================================================
# PART 4: Random Forest
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Random Forest")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

# Create more challenging dataset
X_rf, y_rf = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_clusters_per_class=2, random_state=42
)

# Split
X_train_rf, X_test_rf = X_rf[:800], X_rf[800:]
y_train_rf, y_test_rf = y_rf[:800], y_rf[800:]

print("\n--- Single Tree vs Random Forest ---")

# Single decision tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train_rf, y_train_rf)
print(f"Single Tree Accuracy: {single_tree.score(X_test_rf, y_test_rf):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
print(f"Random Forest (100 trees) Accuracy: {rf.score(X_test_rf, y_test_rf):.4f}")

# Effect of number of trees
print("\n--- Effect of Number of Trees ---")
n_trees_list = [1, 5, 10, 25, 50, 100, 200]
accuracies = []

for n_trees in n_trees_list:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_rf, y_train_rf)
    acc = rf_temp.score(X_test_rf, y_test_rf)
    accuracies.append(acc)
    print(f"  {n_trees:3d} trees: {acc:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Trees')
plt.ylabel('Test Accuracy')
plt.title('Random Forest: Effect of Number of Trees')
plt.grid(True, alpha=0.3)
plt.savefig('rf_num_trees.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: rf_num_trees.png")


# =============================================================================
# PART 5: Feature Importance
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Feature Importance")
print("=" * 60)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- Feature Ranking ---")
for i in range(min(10, len(indices))):
    print(f"  {i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [f'F{i}' for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: feature_importance.png")


# =============================================================================
# PART 6: Gradient Boosting
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Gradient Boosting")
print("=" * 60)

from sklearn.ensemble import GradientBoostingClassifier

print("\n--- Gradient Boosting vs Random Forest ---")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train_rf, y_train_rf)
print(f"Gradient Boosting Accuracy: {gb.score(X_test_rf, y_test_rf):.4f}")
print(f"Random Forest Accuracy: {rf.score(X_test_rf, y_test_rf):.4f}")

# Effect of learning rate
print("\n--- Effect of Learning Rate ---")
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]

for lr in learning_rates:
    gb_temp = GradientBoostingClassifier(
        n_estimators=100, learning_rate=lr, max_depth=3, random_state=42
    )
    gb_temp.fit(X_train_rf, y_train_rf)
    print(f"  LR = {lr:.2f}: Train = {gb_temp.score(X_train_rf, y_train_rf):.4f}, "
          f"Test = {gb_temp.score(X_test_rf, y_test_rf):.4f}")


# =============================================================================
# PART 7: XGBoost and LightGBM
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: XGBoost and LightGBM")
print("=" * 60)

try:
    import xgboost as xgb
    
    print("\n--- XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_rf, y_train_rf)
    print(f"XGBoost Accuracy: {xgb_model.score(X_test_rf, y_test_rf):.4f}")
    
except ImportError:
    print("\nXGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    
    print("\n--- LightGBM ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_rf, y_train_rf)
    print(f"LightGBM Accuracy: {lgb_model.score(X_test_rf, y_test_rf):.4f}")
    
except ImportError:
    print("\nLightGBM not installed. Install with: pip install lightgbm")


# =============================================================================
# PART 8: Hyperparameter Tuning Example
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Hyperparameter Tuning")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

print("\n--- Grid Search for Random Forest ---")

# Define parameter grid (small for demo)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 5]
}

rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_grid, param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_rf, y_train_rf)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test_rf, y_test_rf):.4f}")


# =============================================================================
# PART 9: Comparison Summary
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: Model Comparison Summary")
print("=" * 60)

# Train all models
models = {
    'Single Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n--- Final Comparison ---")
print(f"{'Model':<25} {'Train Acc':>12} {'Test Acc':>12}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train_rf, y_train_rf)
    train_acc = model.score(X_train_rf, y_train_rf)
    test_acc = model.score(X_test_rf, y_test_rf)
    print(f"{name:<25} {train_acc:>12.4f} {test_acc:>12.4f}")


# =============================================================================
# PART 10: Key Takeaways
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. DECISION TREES
   - Split data to maximize information gain (reduce impurity)
   - Interpretable but prone to overfitting
   - Control with max_depth, min_samples_leaf

2. RANDOM FOREST (Bagging)
   - Ensemble of trees on bootstrap samples
   - Random feature selection decorrelates trees
   - Robust, hard to overfit, parallel training

3. GRADIENT BOOSTING
   - Sequential trees, each correcting previous errors
   - Often higher accuracy than Random Forest
   - Needs careful tuning (learning_rate, n_estimators)

4. PRACTICAL TIPS
   - Start with Random Forest (robust defaults)
   - Move to XGBoost/LightGBM for better performance
   - Use early stopping with boosting
   - Feature importance helps interpretability

5. WHEN TO USE
   - Great for tabular data
   - Mixed feature types (numerical + categorical)
   - When you need feature importance
   - Kaggle competitions on structured data!
""")

print("\n" + "=" * 60)
print("Module 9 Complete!")
print("=" * 60)
