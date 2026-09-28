"""
Linear Regression
=================

This module implements linear regression from scratch:
- Hypothesis function and MSE loss
- Gradient descent implementation
- Closed-form solution (normal equation)
- Comparison with sklearn

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: THE LINEAR REGRESSION MODEL
# =============================================================================

def demonstrate_linear_model():
    """Introduce the linear regression hypothesis."""
    print("\n" + "=" * 60)
    print("PART 1: THE LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    print("""
Linear Regression Hypothesis:

    y_hat = w * x + b
    
    y_hat: predicted value
    w: weight (slope)
    b: bias (intercept)
    x: input feature
    
For multiple features:
    y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
    y_hat = X @ w + b  (matrix form)
""")
    
    # Generate simple data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y_true = 2 * X + 3  # True relationship: y = 2x + 3
    y = y_true + np.random.randn(50) * 2  # Add noise
    
    print(f"\nTrue relationship: y = 2*x + 3")
    print(f"Generated {len(X)} data points with Gaussian noise")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and true line
    ax = axes[0]
    ax.scatter(X, y, c='blue', alpha=0.7, label='Data (with noise)')
    ax.plot(X, y_true, 'g--', linewidth=2, label='True: y = 2x + 3')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Problem\n(Find the best line)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Different lines
    ax = axes[1]
    ax.scatter(X, y, c='blue', alpha=0.5)
    
    # Show different parameter choices
    params = [(1, 5, 'red'), (2, 3, 'green'), (3, 0, 'orange')]
    for w, b, color in params:
        y_line = w * X + b
        mse = np.mean((y - y_line)**2)
        ax.plot(X, y_line, color=color, linewidth=2, label=f'w={w}, b={b}, MSE={mse:.1f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Which Line is Best?\n(Minimize Mean Squared Error)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'linear_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'linear_model.png'}")


# =============================================================================
# PART 2: LOSS FUNCTION - MEAN SQUARED ERROR
# =============================================================================

def demonstrate_mse_loss():
    """Visualize the MSE loss function."""
    print("\n" + "=" * 60)
    print("PART 2: LOSS FUNCTION - MEAN SQUARED ERROR")
    print("=" * 60)
    
    print("""
Mean Squared Error (MSE):

    L = (1/n) * sum( (y_pred - y_true)^2 )
    
Why squared?
    - Penalizes large errors more
    - Differentiable everywhere
    - Has a unique minimum
""")
    
    # Generate data
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2.1, 4.0, 5.8, 8.1, 9.9])  # Roughly y = 2x
    
    # Compute MSE for different w values (with b=0)
    w_values = np.linspace(0, 4, 100)
    mse_values = []
    
    for w in w_values:
        y_pred = w * X
        mse = np.mean((y_pred - y)**2)
        mse_values.append(mse)
    
    # Find minimum
    best_w_idx = np.argmin(mse_values)
    best_w = w_values[best_w_idx]
    best_mse = mse_values[best_w_idx]
    
    print(f"\nData: X = {X}, y = {y}")
    print(f"Searching for best w (assuming b=0)")
    print(f"Best w = {best_w:.3f} with MSE = {best_mse:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss surface
    ax = axes[0]
    ax.plot(w_values, mse_values, 'b-', linewidth=2)
    ax.scatter([best_w], [best_mse], color='red', s=100, zorder=5, label=f'Minimum at w={best_w:.2f}')
    ax.axvline(x=best_w, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('w (weight)')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Surface\n(Find the minimum)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 2D loss surface (w and b)
    ax = axes[1]
    w_range = np.linspace(0, 4, 50)
    b_range = np.linspace(-2, 2, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    MSE = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = W[i, j] * X + B[i, j]
            MSE[i, j] = np.mean((y_pred - y)**2)
    
    contour = ax.contour(W, B, MSE, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Mark minimum
    min_idx = np.unravel_index(np.argmin(MSE), MSE.shape)
    ax.scatter([W[min_idx]], [B[min_idx]], color='red', s=100, marker='*', label='Minimum')
    
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('2D Loss Surface\n(Contour plot)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mse_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'mse_loss.png'}")


# =============================================================================
# PART 3: GRADIENT DESCENT FOR LINEAR REGRESSION
# =============================================================================

def demonstrate_gradient_descent():
    """Implement gradient descent for linear regression."""
    print("\n" + "=" * 60)
    print("PART 3: GRADIENT DESCENT FOR LINEAR REGRESSION")
    print("=" * 60)
    
    print("""
Gradient Descent Update Rule:

    Gradients:
        dL/dw = (2/n) * sum( (y_pred - y) * x )
        dL/db = (2/n) * sum( y_pred - y )
    
    Updates:
        w = w - learning_rate * dL/dw
        b = b - learning_rate * dL/db
""")
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n)
    y = 2 * X + 3 + 0.5 * np.random.randn(n)  # y = 2x + 3 + noise
    
    # Gradient descent implementation
    def gradient_descent(X, y, lr=0.1, n_iterations=100):
        """Train linear regression using gradient descent."""
        w = 0.0
        b = 0.0
        n = len(X)
        
        history = {'w': [w], 'b': [b], 'loss': []}
        
        for i in range(n_iterations):
            # Forward pass
            y_pred = w * X + b
            
            # Compute loss
            loss = np.mean((y_pred - y)**2)
            history['loss'].append(loss)
            
            # Compute gradients
            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)
            
            # Update parameters
            w = w - lr * dw
            b = b - lr * db
            
            history['w'].append(w)
            history['b'].append(b)
        
        return w, b, history
    
    # Train
    w_final, b_final, history = gradient_descent(X, y, lr=0.1, n_iterations=50)
    
    print(f"\nInitial: w=0, b=0")
    print(f"After training: w={w_final:.3f}, b={b_final:.3f}")
    print(f"True values: w=2, b=3")
    print(f"Final MSE: {history['loss'][-1]:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Data and fitted line
    ax = axes[0]
    ax.scatter(X, y, c='blue', alpha=0.5, label='Data')
    
    X_line = np.linspace(X.min(), X.max(), 100)
    ax.plot(X_line, 2 * X_line + 3, 'g--', linewidth=2, label='True: y = 2x + 3')
    ax.plot(X_line, w_final * X_line + b_final, 'r-', linewidth=2, 
            label=f'Learned: y = {w_final:.2f}x + {b_final:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gradient Descent Result')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curve
    ax = axes[1]
    ax.plot(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Decreases During Training')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter trajectory on loss surface
    ax = axes[2]
    w_range = np.linspace(-1, 4, 50)
    b_range = np.linspace(0, 6, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    MSE = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = W[i, j] * X + B[i, j]
            MSE[i, j] = np.mean((y_pred - y)**2)
    
    contour = ax.contour(W, B, MSE, levels=20, cmap='viridis', alpha=0.7)
    ax.plot(history['w'], history['b'], 'ro-', markersize=3, linewidth=1, label='GD path')
    ax.scatter([history['w'][0]], [history['b'][0]], color='green', s=100, marker='s', label='Start', zorder=5)
    ax.scatter([history['w'][-1]], [history['b'][-1]], color='red', s=100, marker='*', label='End', zorder=5)
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Gradient Descent Path')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gradient_descent_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'gradient_descent_lr.png'}")
    
    return w_final, b_final


# =============================================================================
# PART 4: CLOSED-FORM SOLUTION (NORMAL EQUATION)
# =============================================================================

def demonstrate_normal_equation():
    """Show the closed-form solution for linear regression."""
    print("\n" + "=" * 60)
    print("PART 4: CLOSED-FORM SOLUTION (NORMAL EQUATION)")
    print("=" * 60)
    
    print("""
Normal Equation:

    theta = (X^T * X)^(-1) * X^T * y
    
Where:
    - X has a column of 1s added (for bias term)
    - theta = [b, w1, w2, ...]
    
Pros: No iterations, exact solution
Cons: Slow for large datasets (matrix inversion is O(n^3))
""")
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n)
    y = 2 * X + 3 + 0.5 * np.random.randn(n)
    
    # Normal equation implementation
    def normal_equation(X, y):
        """Solve linear regression using normal equation."""
        # Add bias column
        X_b = np.c_[np.ones(len(X)), X]
        
        # theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        return theta[0], theta[1]  # b, w
    
    b_closed, w_closed = normal_equation(X, y)
    
    print(f"\nClosed-form solution:")
    print(f"  w = {w_closed:.4f}")
    print(f"  b = {b_closed:.4f}")
    print(f"  (True: w=2, b=3)")
    
    # Compare with sklearn
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    
    print(f"\nsklearn LinearRegression:")
    print(f"  w = {model.coef_[0]:.4f}")
    print(f"  b = {model.intercept_:.4f}")
    
    # Verify they're the same
    print(f"\nDifference: w={abs(w_closed - model.coef_[0]):.10f}, b={abs(b_closed - model.intercept_):.10f}")


# =============================================================================
# PART 5: POLYNOMIAL REGRESSION
# =============================================================================

def demonstrate_polynomial_regression():
    """Show polynomial features for nonlinear relationships."""
    print("\n" + "=" * 60)
    print("PART 5: POLYNOMIAL REGRESSION")
    print("=" * 60)
    
    print("""
For nonlinear relationships, create polynomial features:

    Original: x
    Degree 2: x, x^2
    Degree 3: x, x^2, x^3
    
Then apply linear regression to the expanded features!
    y = w0 + w1*x + w2*x^2 + w3*x^3 + ...
""")
    
    # Generate nonlinear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 50)
    y = 0.5 * X**3 - 2 * X**2 + X + 2 + np.random.randn(50) * 2  # Cubic relationship
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    degrees = [1, 3, 15]
    titles = ['Degree 1 (Underfitting)', 'Degree 3 (Good)', 'Degree 15 (Overfitting)']
    
    for idx, (degree, title) in enumerate(zip(degrees, titles)):
        ax = axes[idx]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict
        X_plot = np.linspace(-3, 3, 100)
        X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
        y_pred = model.predict(X_plot_poly)
        
        # Compute metrics
        y_train_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_train_pred)
        r2 = r2_score(y, y_train_pred)
        
        ax.scatter(X, y, c='blue', alpha=0.7, label='Data')
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title}\nMSE={mse:.2f}, R2={r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-30, 30)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'polynomial_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'polynomial_regression.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all linear regression demonstrations."""
    print("=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)
    print("\nThe simplest and most fundamental supervised learning algorithm.")
    
    demonstrate_linear_model()
    demonstrate_mse_loss()
    demonstrate_gradient_descent()
    demonstrate_normal_equation()
    demonstrate_polynomial_regression()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. LINEAR MODEL: y = wx + b
   - Simple hypothesis function
   - Foundation for neural networks

2. MSE LOSS: L = (1/n) * sum((y_pred - y)^2)
   - Measures prediction error
   - Has a unique global minimum

3. GRADIENT DESCENT:
   - Iteratively update w and b
   - Move in direction of steepest descent
   - Learning rate controls step size

4. NORMAL EQUATION: theta = (X^T X)^(-1) X^T y
   - Closed-form solution
   - No iterations needed
   - Slow for large datasets

5. POLYNOMIAL FEATURES:
   - Handle nonlinear relationships
   - Be careful of overfitting!

Linear regression is simple but powerful:
   - Easy to interpret
   - Fast to train
   - Foundation for understanding more complex models
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
