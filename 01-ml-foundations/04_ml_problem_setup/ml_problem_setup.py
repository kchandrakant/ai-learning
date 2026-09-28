"""
The ML Problem Setup
====================

This module demonstrates the fundamental concepts of machine learning:
- Types of learning (supervised, unsupervised, reinforcement)
- Train/validation/test splits
- Bias-variance tradeoff
- Underfitting vs overfitting

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: TYPES OF LEARNING
# =============================================================================

def demonstrate_learning_types():
    """Visualize different types of machine learning."""
    print("\n" + "=" * 60)
    print("PART 1: TYPES OF MACHINE LEARNING")
    print("=" * 60)
    
    print("""
1. SUPERVISED LEARNING
   - Given: inputs X AND outputs y
   - Goal: learn mapping f such that f(X) ~ y
   - Examples: classification, regression
   
2. UNSUPERVISED LEARNING
   - Given: only inputs X (no labels)
   - Goal: find structure/patterns in data
   - Examples: clustering, dimensionality reduction
   
3. REINFORCEMENT LEARNING
   - Given: environment, actions, rewards
   - Goal: learn policy to maximize cumulative reward
   - Examples: game playing, robotics
""")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Supervised Learning - Classification
    ax = axes[0]
    np.random.seed(42)
    # Two classes
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    ax.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0', alpha=0.7)
    ax.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1', alpha=0.7)
    # Decision boundary
    ax.plot([-4, 4], [4, -4], 'k--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Supervised Learning\n(Classification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # Unsupervised Learning - Clustering
    ax = axes[1]
    np.random.seed(42)
    # Three clusters (no labels)
    X1 = np.random.randn(30, 2) + np.array([3, 3])
    X2 = np.random.randn(30, 2) + np.array([-3, 3])
    X3 = np.random.randn(30, 2) + np.array([0, -3])
    X = np.vstack([X1, X2, X3])
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7, s=50)
    # Show discovered clusters
    ax.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.3, s=100)
    ax.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.3, s=100)
    ax.scatter(X3[:, 0], X3[:, 1], c='green', alpha=0.3, s=100)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Unsupervised Learning\n(Clustering - no labels given)')
    ax.grid(True, alpha=0.3)
    
    # Reinforcement Learning - simple illustration
    ax = axes[2]
    # Grid world
    grid_size = 5
    for i in range(grid_size + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5)
        ax.axvline(x=i, color='gray', linewidth=0.5)
    # Agent, goal, obstacles
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7))
    ax.text(0.5, 0.5, 'Agent', ha='center', va='center', fontsize=10, color='white')
    ax.add_patch(plt.Rectangle((4, 4), 1, 1, color='green', alpha=0.7))
    ax.text(4.5, 4.5, 'Goal\n+100', ha='center', va='center', fontsize=9, color='white')
    ax.add_patch(plt.Rectangle((2, 2), 1, 1, color='red', alpha=0.7))
    ax.text(2.5, 2.5, 'Trap\n-50', ha='center', va='center', fontsize=9, color='white')
    # Path
    path = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (1.5, 2.5), (1.5, 3.5), (2.5, 3.5), (3.5, 3.5), (4.5, 3.5), (4.5, 4.5)]
    for i in range(len(path) - 1):
        ax.annotate('', xy=path[i+1], xytext=path[i],
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_title('Reinforcement Learning\n(Agent learns from rewards)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'learning_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'learning_types.png'}")


# =============================================================================
# PART 2: TRAIN/VALIDATION/TEST SPLIT
# =============================================================================

def demonstrate_data_splits():
    """Show how to properly split data."""
    print("\n" + "=" * 60)
    print("PART 2: TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)
    
    print("""
Data Split Strategy:
    
    All Data
    |-- Training Set (60-80%)    -> Train the model
    |-- Validation Set (10-20%)  -> Tune hyperparameters  
    |-- Test Set (10-20%)        -> Final evaluation (ONLY USE ONCE!)
    
GOLDEN RULE: Never tune your model based on test set performance!
             Otherwise you're "cheating" and overestimating generalization.
""")
    
    # Generate sample data
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    
    # Method 1: Two-stage split for train/val/test
    print("\n--- Method 1: Manual Split ---")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Total samples: {n_samples}")
    print(f"Training set:   {len(X_train)} samples ({len(X_train)/n_samples*100:.0f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/n_samples*100:.0f}%)")
    print(f"Test set:       {len(X_test)} samples ({len(X_test)/n_samples*100:.0f}%)")
    
    # Visualize the split
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data split visualization
    ax = axes[0]
    sizes = [len(X_train), len(X_val), len(X_test)]
    labels = [f'Train\n{sizes[0]} ({sizes[0]/n_samples*100:.0f}%)', 
              f'Validation\n{sizes[1]} ({sizes[1]/n_samples*100:.0f}%)',
              f'Test\n{sizes[2]} ({sizes[2]/n_samples*100:.0f}%)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax.set_title('Data Split (60/20/20)')
    
    # Plot 2: Why we need splits
    ax = axes[1]
    # Simulate model complexity vs error
    complexity = np.linspace(1, 20, 100)
    train_error = 1 / complexity
    val_error = 0.3 + 0.05 * (complexity - 5)**2 / 25
    
    ax.plot(complexity, train_error, 'b-', linewidth=2, label='Training Error')
    ax.plot(complexity, val_error, 'r-', linewidth=2, label='Validation Error')
    ax.axvline(x=5, color='green', linestyle='--', label='Optimal Complexity')
    ax.fill_between(complexity[:25], 0, 1, alpha=0.2, color='blue', label='Underfitting')
    ax.fill_between(complexity[75:], 0, 1, alpha=0.2, color='red', label='Overfitting')
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('Error')
    ax.set_title('Why Validation Set Matters\n(Find optimal complexity)')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'data_splits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'data_splits.png'}")


# =============================================================================
# PART 3: BIAS-VARIANCE TRADEOFF
# =============================================================================

def demonstrate_bias_variance():
    """Visualize the bias-variance tradeoff."""
    print("\n" + "=" * 60)
    print("PART 3: BIAS-VARIANCE TRADEOFF")
    print("=" * 60)
    
    print("""
The Fundamental Equation:

    Total Error = Bias^2 + Variance + Irreducible Noise
    
BIAS: Error from wrong assumptions
    - Model too simple -> misses patterns
    - High bias = underfitting
    
VARIANCE: Error from sensitivity to training data
    - Model too complex -> memorizes noise
    - High variance = overfitting
    
GOAL: Find the sweet spot that minimizes TOTAL error!
""")
    
    # Generate true function with noise
    np.random.seed(42)
    n_points = 30
    X = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * X)
    y = y_true + 0.3 * np.random.randn(n_points)
    
    # Fit polynomials of different degrees
    degrees = [1, 4, 15]
    titles = ['Underfitting (High Bias)', 'Good Fit', 'Overfitting (High Variance)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    X_plot = np.linspace(0, 1, 100)
    
    for idx, (degree, title) in enumerate(zip(degrees, titles)):
        ax = axes[0, idx]
        
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_plot_poly)
        
        # Training error
        train_mse = mean_squared_error(y, model.predict(X_poly))
        
        # Plot
        ax.scatter(X, y, c='blue', alpha=0.7, label='Training data')
        ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', linewidth=2, label='True function')
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'{title}\nDegree={degree}, Train MSE={train_mse:.3f}')
        ax.legend()
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
    
    # Show how different training sets affect each model
    ax = axes[1, 0]
    ax.text(0.5, 0.7, 'HIGH BIAS', fontsize=20, ha='center', va='center', fontweight='bold', color='blue')
    ax.text(0.5, 0.5, 'Predictions are consistently\nwrong (same direction)', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.3, 'Model is too simple\nto capture pattern', fontsize=12, ha='center', va='center')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.text(0.5, 0.7, 'GOOD BALANCE', fontsize=20, ha='center', va='center', fontweight='bold', color='green')
    ax.text(0.5, 0.5, 'Predictions are accurate\nand consistent', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.3, 'Model complexity\nmatches the problem', fontsize=12, ha='center', va='center')
    ax.axis('off')
    
    ax = axes[1, 2]
    ax.text(0.5, 0.7, 'HIGH VARIANCE', fontsize=20, ha='center', va='center', fontweight='bold', color='red')
    ax.text(0.5, 0.5, 'Predictions change wildly\nwith different training data', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.3, 'Model is too complex,\nfitting noise', fontsize=12, ha='center', va='center')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bias_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'bias_variance.png'}")


# =============================================================================
# PART 4: CROSS-VALIDATION
# =============================================================================

def demonstrate_cross_validation():
    """Show k-fold cross-validation."""
    print("\n" + "=" * 60)
    print("PART 4: CROSS-VALIDATION")
    print("=" * 60)
    
    print("""
K-Fold Cross-Validation:
    
    1. Split data into K equal parts (folds)
    2. For each fold:
       - Use that fold as validation
       - Use remaining K-1 folds as training
       - Compute validation score
    3. Average all K scores
    
Benefits:
    - Uses all data for both training and validation
    - More reliable estimate than single split
    - Reduces variance in performance estimate
""")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary
    
    # Visualize K-fold
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show the K splits
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        if fold_idx < 5:
            ax = axes[fold_idx // 3, fold_idx % 3]
            
            # Plot all points
            ax.scatter(X[train_idx, 0], X[train_idx, 1], c='blue', alpha=0.5, s=30, label='Training')
            ax.scatter(X[val_idx, 0], X[val_idx, 1], c='red', alpha=0.8, s=50, label='Validation')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'Fold {fold_idx + 1}: Val size = {len(val_idx)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Summary in last plot
    ax = axes[1, 2]
    
    # Use sklearn's cross_val_score
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=5)
    
    ax.bar(range(1, 6), scores, color='steelblue', alpha=0.7)
    ax.axhline(y=scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
    ax.fill_between([0.5, 5.5], scores.mean() - scores.std(), scores.mean() + scores.std(), 
                    alpha=0.2, color='red', label=f'Std: {scores.std():.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'5-Fold CV Results\nAccuracy: {scores.mean():.3f} +/- {scores.std():.3f}')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cross_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'cross_validation.png'}")
    
    print(f"\nCross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all ML problem setup demonstrations."""
    print("=" * 60)
    print("THE ML PROBLEM SETUP")
    print("=" * 60)
    print("\nUnderstanding the fundamentals before diving into algorithms.")
    
    demonstrate_learning_types()
    demonstrate_data_splits()
    demonstrate_bias_variance()
    demonstrate_cross_validation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. TYPES OF LEARNING:
   - Supervised: learn from labeled examples
   - Unsupervised: find structure without labels
   - Reinforcement: learn from rewards

2. DATA SPLITS:
   - Train: fit the model
   - Validation: tune hyperparameters
   - Test: final evaluation (use ONCE!)

3. BIAS-VARIANCE TRADEOFF:
   - High bias = underfitting (too simple)
   - High variance = overfitting (too complex)
   - Goal: minimize total error

4. CROSS-VALIDATION:
   - More reliable than single split
   - K-fold: use each fold as validation once
   - Reports mean +/- std of scores

Before building any model, ask:
   - What type of problem is this?
   - How will I split my data?
   - How will I know if I'm overfitting?
   - What metric defines success?
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
