"""
Regularization & Generalization
================================

This module explores techniques to prevent overfitting:
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Elastic Net
- Early stopping

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: THE OVERFITTING PROBLEM
# =============================================================================

def demonstrate_overfitting():
    """Show what overfitting looks like."""
    print("\n" + "=" * 60)
    print("PART 1: THE OVERFITTING PROBLEM")
    print("=" * 60)
    
    print("""
Overfitting: Model memorizes training data, fails on new data.

Signs of overfitting:
    - Training error: LOW
    - Validation/Test error: HIGH
    - Gap between train and test error is LARGE

The model learned the noise, not the pattern!
""")
    
    # Generate data
    np.random.seed(42)
    n = 20
    X = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * X)
    y = y_true + 0.3 * np.random.randn(n)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fit different complexity models
    degrees = [1, 4, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    X_plot = np.linspace(0, 1, 100)
    
    for ax, degree in zip(axes, degrees):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
        test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
        
        ax.scatter(X_train, y_train, c='blue', label='Train', alpha=0.7)
        ax.scatter(X_test, y_test, c='red', label='Test', alpha=0.7)
        ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', label='True', linewidth=2)
        ax.plot(X_plot, model.predict(X_plot_poly), 'k-', label='Fit', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'Degree {degree}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
        ax.legend()
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overfitting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'overfitting.png'}")


# =============================================================================
# PART 2: L2 REGULARIZATION (RIDGE)
# =============================================================================

def demonstrate_ridge():
    """Show L2 regularization (Ridge regression)."""
    print("\n" + "=" * 60)
    print("PART 2: L2 REGULARIZATION (RIDGE)")
    print("=" * 60)
    
    print("""
Ridge Regression adds L2 penalty:

    Loss = MSE + lambda * sum(w^2)
    
Effect: Shrinks weights toward zero (but never exactly zero)
    - Large lambda: more shrinkage, simpler model
    - Small lambda: less shrinkage, closer to OLS
""")
    
    # Generate data
    np.random.seed(42)
    n = 30
    X = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * X) + 0.3 * np.random.randn(n)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # High degree polynomial (prone to overfitting)
    degree = 15
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    
    # Compare different alpha values
    alphas = [0, 0.001, 0.1, 10]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    X_plot = np.linspace(0, 1, 100)
    X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
    
    for ax, alpha in zip(axes, alphas):
        if alpha == 0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=alpha)
        
        model.fit(X_train_poly, y_train)
        
        train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
        test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
        
        ax.scatter(X_train, y_train, c='blue', alpha=0.7)
        ax.scatter(X_test, y_test, c='red', alpha=0.7)
        ax.plot(X_plot, model.predict(X_plot_poly), 'k-', linewidth=2)
        ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'alpha={alpha}\nTest MSE: {test_mse:.3f}')
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ridge.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'ridge.png'}")


# =============================================================================
# PART 3: L1 REGULARIZATION (LASSO)
# =============================================================================

def demonstrate_lasso():
    """Show L1 regularization (Lasso regression)."""
    print("\n" + "=" * 60)
    print("PART 3: L1 REGULARIZATION (LASSO)")
    print("=" * 60)
    
    print("""
Lasso Regression adds L1 penalty:

    Loss = MSE + lambda * sum(|w|)
    
Effect: Can shrink weights exactly to ZERO (feature selection!)
    - Produces sparse models
    - Useful when you have many irrelevant features
""")
    
    # Generate data with many features, only few relevant
    np.random.seed(42)
    n = 100
    n_features = 20
    n_relevant = 5
    
    X = np.random.randn(n, n_features)
    true_weights = np.zeros(n_features)
    true_weights[:n_relevant] = np.array([3, -2, 1.5, -1, 0.5])
    y = X @ true_weights + 0.5 * np.random.randn(n)
    
    # Fit Ridge vs Lasso
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    
    ridge.fit(X, y)
    lasso.fit(X, y)
    
    # Visualize coefficients
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.bar(range(n_features), true_weights, color='green', alpha=0.7)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient')
    ax.set_title('True Weights\n(Only 5 are non-zero)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.bar(range(n_features), ridge.coef_, color='blue', alpha=0.7)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient')
    ax.set_title('Ridge Coefficients\n(All non-zero, but small)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.bar(range(n_features), lasso.coef_, color='red', alpha=0.7)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Lasso Coefficients\n({np.sum(lasso.coef_ != 0)} non-zero = feature selection!)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lasso.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'lasso.png'}")
    
    print(f"\nTrue non-zero features: {n_relevant}")
    print(f"Ridge non-zero features: {np.sum(np.abs(ridge.coef_) > 0.01)}")
    print(f"Lasso non-zero features: {np.sum(lasso.coef_ != 0)}")


# =============================================================================
# PART 4: ELASTIC NET
# =============================================================================

def demonstrate_elastic_net():
    """Show Elastic Net (L1 + L2 combined)."""
    print("\n" + "=" * 60)
    print("PART 4: ELASTIC NET")
    print("=" * 60)
    
    print("""
Elastic Net combines L1 and L2:

    Loss = MSE + lambda1 * sum(|w|) + lambda2 * sum(w^2)
    
l1_ratio controls the mix:
    - l1_ratio = 1: pure Lasso
    - l1_ratio = 0: pure Ridge
    - l1_ratio = 0.5: equal mix
    
Best of both worlds when features are correlated.
""")
    
    # Generate correlated features
    np.random.seed(42)
    n = 100
    X1 = np.random.randn(n)
    X2 = X1 + 0.1 * np.random.randn(n)  # Highly correlated with X1
    X3 = np.random.randn(n)
    X = np.column_stack([X1, X2, X3])
    y = 2 * X1 + 3 * X3 + 0.5 * np.random.randn(n)
    
    # Compare models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    print("\nCoefficients (true: w1=2, w2=0, w3=3):")
    for name, model in models.items():
        model.fit(X, y)
        print(f"  {name}: {model.coef_.round(3)}")


# =============================================================================
# PART 5: EARLY STOPPING
# =============================================================================

def demonstrate_early_stopping():
    """Show early stopping as regularization."""
    print("\n" + "=" * 60)
    print("PART 5: EARLY STOPPING")
    print("=" * 60)
    
    print("""
Early Stopping: Stop training when validation loss stops improving.

    - Training loss keeps decreasing
    - Validation loss decreases, then INCREASES (overfitting!)
    - Stop at the minimum validation loss
    
This is implicit regularization - simpler than adding penalty terms.
""")
    
    # Simulate training curves
    np.random.seed(42)
    epochs = np.arange(1, 101)
    
    # Training loss decreases
    train_loss = 2 * np.exp(-0.05 * epochs) + 0.1 + 0.02 * np.random.randn(100)
    
    # Validation loss: decreases then increases
    val_loss = 2 * np.exp(-0.05 * epochs) + 0.3 + 0.01 * (epochs - 30)**2 / 100 + 0.03 * np.random.randn(100)
    val_loss = np.maximum(val_loss, 0.2)
    
    # Find best epoch
    best_epoch = np.argmin(val_loss) + 1
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
               label=f'Early Stop (epoch {best_epoch})')
    
    ax.fill_between(epochs[best_epoch:], 0, 3, alpha=0.2, color='red', label='Overfitting zone')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Early Stopping\n(Stop when validation loss starts increasing)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'early_stopping.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'early_stopping.png'}")
    
    print(f"\nBest epoch: {best_epoch}")
    print("Without early stopping, we would train to epoch 100 and overfit!")


# =============================================================================
# PART 6: CHOOSING REGULARIZATION STRENGTH
# =============================================================================

def demonstrate_hyperparameter_tuning():
    """Show how to choose regularization strength."""
    print("\n" + "=" * 60)
    print("PART 6: CHOOSING REGULARIZATION STRENGTH")
    print("=" * 60)
    
    print("""
Use cross-validation to find optimal lambda (alpha):

    1. Try different alpha values
    2. For each alpha, compute CV score
    3. Choose alpha with best CV score
""")
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 10)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n)
    
    # Grid search
    alphas = np.logspace(-4, 2, 50)
    
    param_grid = {'alpha': alphas}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    print(f"\nBest alpha: {grid_search.best_params_['alpha']:.4f}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    mean_scores = -grid_search.cv_results_['mean_test_score']
    std_scores = grid_search.cv_results_['std_test_score']
    
    ax.semilogx(alphas, mean_scores, 'b-', linewidth=2)
    ax.fill_between(alphas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
    ax.axvline(x=grid_search.best_params_['alpha'], color='red', linestyle='--', 
               label=f"Best alpha = {grid_search.best_params_['alpha']:.4f}")
    
    ax.set_xlabel('Alpha (regularization strength)')
    ax.set_ylabel('Mean CV MSE')
    ax.set_title('Cross-Validation to Choose Regularization Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'hyperparameter_tuning.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all regularization demonstrations."""
    print("=" * 60)
    print("REGULARIZATION & GENERALIZATION")
    print("=" * 60)
    
    demonstrate_overfitting()
    demonstrate_ridge()
    demonstrate_lasso()
    demonstrate_elastic_net()
    demonstrate_early_stopping()
    demonstrate_hyperparameter_tuning()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. OVERFITTING: Low train error, high test error
   - Model learned noise, not signal

2. L2 REGULARIZATION (RIDGE):
   - Loss = MSE + lambda * sum(w^2)
   - Shrinks weights toward zero

3. L1 REGULARIZATION (LASSO):
   - Loss = MSE + lambda * sum(|w|)
   - Can zero out weights (feature selection)

4. ELASTIC NET:
   - Combines L1 + L2
   - Good for correlated features

5. EARLY STOPPING:
   - Stop when validation loss increases
   - Simple and effective

6. CHOOSING LAMBDA:
   - Use cross-validation
   - Balance bias vs variance

Rule of thumb:
   - Start with Ridge (L2)
   - Use Lasso if you want feature selection
   - Use cross-validation to tune strength
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
