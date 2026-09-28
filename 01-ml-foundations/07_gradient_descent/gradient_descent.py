"""
Gradient Descent Deep Dive
==========================

This module explores gradient descent variants:
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Momentum
- Adam optimizer

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: BATCH GRADIENT DESCENT
# =============================================================================

def demonstrate_batch_gd():
    """Show batch gradient descent."""
    print("\n" + "=" * 60)
    print("PART 1: BATCH GRADIENT DESCENT")
    print("=" * 60)
    
    print("""
Batch Gradient Descent:
    
    Compute gradient using ALL training examples:
    
    gradient = (1/n) * sum( gradient for each example )
    theta = theta - lr * gradient
    
Pros:
    - Stable updates
    - Guaranteed to converge (convex functions)
    
Cons:
    - Slow for large datasets
    - Each update requires full dataset pass
""")
    
    # Generate data
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    y = 2 * X + 3 + 0.5 * np.random.randn(n)
    
    def batch_gradient_descent(X, y, lr=0.1, n_iterations=50):
        """Batch gradient descent for linear regression."""
        w, b = 0.0, 0.0
        n = len(X)
        
        history = {'w': [w], 'b': [b], 'loss': []}
        
        for _ in range(n_iterations):
            # Compute predictions
            y_pred = w * X + b
            
            # Compute loss
            loss = np.mean((y_pred - y)**2)
            history['loss'].append(loss)
            
            # Compute gradients (using ALL data)
            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)
            
            # Update
            w = w - lr * dw
            b = b - lr * db
            
            history['w'].append(w)
            history['b'].append(b)
        
        return w, b, history
    
    w, b, history = batch_gradient_descent(X, y, lr=0.1, n_iterations=30)
    
    print(f"\nResult: w={w:.3f}, b={b:.3f}")
    print(f"True:   w=2.000, b=3.000")
    
    return X, y, history


# =============================================================================
# PART 2: STOCHASTIC GRADIENT DESCENT
# =============================================================================

def demonstrate_sgd(X, y):
    """Show stochastic gradient descent."""
    print("\n" + "=" * 60)
    print("PART 2: STOCHASTIC GRADIENT DESCENT (SGD)")
    print("=" * 60)
    
    print("""
Stochastic Gradient Descent:
    
    Compute gradient using ONE example at a time:
    
    for each example (x_i, y_i):
        gradient = gradient for (x_i, y_i)
        theta = theta - lr * gradient
    
Pros:
    - Fast updates
    - Can escape local minima (noise helps!)
    - Works well for large datasets
    
Cons:
    - Noisy updates
    - May not converge exactly
""")
    
    def sgd(X, y, lr=0.01, n_epochs=5):
        """Stochastic gradient descent for linear regression."""
        w, b = 0.0, 0.0
        n = len(X)
        
        history = {'w': [w], 'b': [b], 'loss': [], 'updates': 0}
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            
            for i in indices:
                x_i, y_i = X[i], y[i]
                
                # Compute prediction for this example
                y_pred = w * x_i + b
                
                # Compute gradients (from single example)
                dw = 2 * (y_pred - y_i) * x_i
                db = 2 * (y_pred - y_i)
                
                # Update
                w = w - lr * dw
                b = b - lr * db
                
                history['w'].append(w)
                history['b'].append(b)
                history['updates'] += 1
            
            # Compute loss at end of epoch
            y_pred = w * X + b
            loss = np.mean((y_pred - y)**2)
            history['loss'].append(loss)
        
        return w, b, history
    
    w, b, history = sgd(X, y, lr=0.01, n_epochs=5)
    
    print(f"\nResult: w={w:.3f}, b={b:.3f}")
    print(f"Updates: {history['updates']}")
    
    return history


# =============================================================================
# PART 3: MINI-BATCH GRADIENT DESCENT
# =============================================================================

def demonstrate_mini_batch(X, y):
    """Show mini-batch gradient descent."""
    print("\n" + "=" * 60)
    print("PART 3: MINI-BATCH GRADIENT DESCENT")
    print("=" * 60)
    
    print("""
Mini-Batch Gradient Descent:
    
    Compute gradient using BATCH_SIZE examples:
    
    for each mini-batch of size B:
        gradient = (1/B) * sum( gradient for batch )
        theta = theta - lr * gradient
    
Typical batch sizes: 32, 64, 128, 256

Best of both worlds:
    - More stable than SGD
    - Faster than batch GD
    - Allows GPU parallelization
""")
    
    def mini_batch_gd(X, y, batch_size=32, lr=0.05, n_epochs=10):
        """Mini-batch gradient descent for linear regression."""
        w, b = 0.0, 0.0
        n = len(X)
        
        history = {'w': [w], 'b': [b], 'loss': []}
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Compute predictions
                y_pred = w * X_batch + b
                
                # Compute gradients
                dw = (2/len(X_batch)) * np.sum((y_pred - y_batch) * X_batch)
                db = (2/len(X_batch)) * np.sum(y_pred - y_batch)
                
                # Update
                w = w - lr * dw
                b = b - lr * db
                
                history['w'].append(w)
                history['b'].append(b)
            
            # Compute loss at end of epoch
            y_pred = w * X + b
            loss = np.mean((y_pred - y)**2)
            history['loss'].append(loss)
        
        return w, b, history
    
    w, b, history = mini_batch_gd(X, y, batch_size=32, lr=0.05, n_epochs=10)
    
    print(f"\nResult: w={w:.3f}, b={b:.3f}")
    
    return history


# =============================================================================
# PART 4: MOMENTUM
# =============================================================================

def demonstrate_momentum():
    """Show SGD with momentum."""
    print("\n" + "=" * 60)
    print("PART 4: MOMENTUM")
    print("=" * 60)
    
    print("""
SGD with Momentum:
    
    Accumulate velocity in consistent gradient direction:
    
    v = beta * v + (1 - beta) * gradient   # or just: v = beta * v - lr * gradient
    theta = theta + v
    
    beta: momentum coefficient (typically 0.9)
    
Benefits:
    - Accelerates in consistent directions
    - Dampens oscillations
    - Helps escape saddle points
""")
    
    # Create a "ravine" loss surface
    def loss_function(w1, w2):
        """Elongated bowl (ravine)."""
        return 10 * w1**2 + w2**2
    
    def gradient(w1, w2):
        return np.array([20 * w1, 2 * w2])
    
    def sgd_basic(start, lr, n_steps):
        """Basic SGD without momentum."""
        w = np.array(start, dtype=float)
        path = [w.copy()]
        
        for _ in range(n_steps):
            grad = gradient(w[0], w[1])
            w = w - lr * grad
            path.append(w.copy())
        
        return np.array(path)
    
    def sgd_momentum(start, lr, momentum, n_steps):
        """SGD with momentum."""
        w = np.array(start, dtype=float)
        v = np.zeros(2)
        path = [w.copy()]
        
        for _ in range(n_steps):
            grad = gradient(w[0], w[1])
            v = momentum * v - lr * grad
            w = w + v
            path.append(w.copy())
        
        return np.array(path)
    
    # Run both
    start = [1.0, 1.0]
    path_basic = sgd_basic(start, lr=0.08, n_steps=30)
    path_momentum = sgd_momentum(start, lr=0.08, momentum=0.9, n_steps=30)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create contour plot
    w1_range = np.linspace(-1.5, 1.5, 100)
    w2_range = np.linspace(-1.5, 1.5, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = loss_function(W1, W2)
    
    for ax, path, title in [(axes[0], path_basic, 'SGD without Momentum\n(oscillates in ravine)'),
                            (axes[1], path_momentum, 'SGD with Momentum (beta=0.9)\n(accelerates along ravine)')]:
        ax.contour(W1, W2, Z, levels=30, cmap='viridis', alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=1)
        ax.scatter([start[0]], [start[1]], color='green', s=100, marker='s', label='Start', zorder=5)
        ax.scatter([path[-1, 0]], [path[-1, 1]], color='red', s=100, marker='*', label='End', zorder=5)
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'momentum.png'}")


# =============================================================================
# PART 5: ADAM OPTIMIZER
# =============================================================================

def demonstrate_adam():
    """Show the Adam optimizer."""
    print("\n" + "=" * 60)
    print("PART 5: ADAM OPTIMIZER")
    print("=" * 60)
    
    print("""
Adam (Adaptive Moment Estimation):
    
    Combines momentum with adaptive learning rates:
    
    m = beta1 * m + (1 - beta1) * gradient      # First moment (momentum)
    v = beta2 * v + (1 - beta2) * gradient^2    # Second moment (RMSprop)
    
    m_hat = m / (1 - beta1^t)  # Bias correction
    v_hat = v / (1 - beta2^t)
    
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    
Default hyperparameters:
    - lr = 0.001
    - beta1 = 0.9
    - beta2 = 0.999
    - eps = 1e-8

Adam is the default optimizer for deep learning!
""")
    
    def adam(start, loss_fn, grad_fn, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=100):
        """Adam optimizer."""
        w = np.array(start, dtype=float)
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        
        path = [w.copy()]
        
        for t in range(1, n_steps + 1):
            grad = grad_fn(w[0], w[1])
            
            # Update biased moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Update
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
            path.append(w.copy())
        
        return np.array(path)
    
    # Test on Rosenbrock-like function (harder optimization)
    def loss_function(w1, w2):
        return (1 - w1)**2 + 10 * (w2 - w1**2)**2
    
    def gradient(w1, w2):
        dw1 = -2 * (1 - w1) - 40 * w1 * (w2 - w1**2)
        dw2 = 20 * (w2 - w1**2)
        return np.array([dw1, dw2])
    
    # Run Adam
    start = [-1.0, 1.0]
    path_adam = adam(start, loss_function, gradient, lr=0.1, n_steps=200)
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    w1_range = np.linspace(-2, 2, 100)
    w2_range = np.linspace(-1, 3, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = loss_function(W1, W2)
    
    ax.contour(W1, W2, Z, levels=np.logspace(-1, 3, 30), cmap='viridis', alpha=0.7)
    ax.plot(path_adam[:, 0], path_adam[:, 1], 'ro-', markersize=2, linewidth=0.5, alpha=0.7)
    ax.scatter([start[0]], [start[1]], color='green', s=100, marker='s', label='Start', zorder=5)
    ax.scatter([1], [1], color='blue', s=100, marker='*', label='Global minimum', zorder=5)
    ax.scatter([path_adam[-1, 0]], [path_adam[-1, 1]], color='red', s=100, marker='*', label='End', zorder=5)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_title('Adam Optimizer on Rosenbrock Function\n(Challenging non-convex optimization)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'adam.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'adam.png'}")


# =============================================================================
# PART 6: COMPARISON
# =============================================================================

def demonstrate_comparison():
    """Compare all optimizers."""
    print("\n" + "=" * 60)
    print("PART 6: OPTIMIZER COMPARISON")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.5 * np.random.randn(n)
    
    def compute_loss(w, b, X, y):
        return np.mean((X @ w + b - y)**2)
    
    def train_batch_gd(X, y, lr, n_epochs):
        w = np.zeros(2)
        b = 0.0
        losses = []
        
        for _ in range(n_epochs):
            y_pred = X @ w + b
            losses.append(compute_loss(w, b, X, y))
            
            dw = (2/len(X)) * (X.T @ (y_pred - y))
            db = (2/len(X)) * np.sum(y_pred - y)
            
            w = w - lr * dw
            b = b - lr * db
        
        return losses
    
    def train_sgd(X, y, lr, n_epochs):
        w = np.zeros(2)
        b = 0.0
        losses = []
        
        for _ in range(n_epochs):
            losses.append(compute_loss(w, b, X, y))
            indices = np.random.permutation(len(X))
            
            for i in indices:
                y_pred = X[i] @ w + b
                dw = 2 * (y_pred - y[i]) * X[i]
                db = 2 * (y_pred - y[i])
                
                w = w - lr * dw
                b = b - lr * db
        
        return losses
    
    def train_sgd_momentum(X, y, lr, momentum, n_epochs):
        w = np.zeros(2)
        b = 0.0
        vw = np.zeros(2)
        vb = 0.0
        losses = []
        
        for _ in range(n_epochs):
            losses.append(compute_loss(w, b, X, y))
            indices = np.random.permutation(len(X))
            
            for i in indices:
                y_pred = X[i] @ w + b
                dw = 2 * (y_pred - y[i]) * X[i]
                db = 2 * (y_pred - y[i])
                
                vw = momentum * vw - lr * dw
                vb = momentum * vb - lr * db
                
                w = w + vw
                b = b + vb
        
        return losses
    
    def train_adam(X, y, lr, n_epochs):
        w = np.zeros(2)
        b = 0.0
        mw, mb = np.zeros(2), 0.0
        vw, vb = np.zeros(2), 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        losses = []
        t = 0
        
        for _ in range(n_epochs):
            losses.append(compute_loss(w, b, X, y))
            indices = np.random.permutation(len(X))
            
            for i in indices:
                t += 1
                y_pred = X[i] @ w + b
                dw = 2 * (y_pred - y[i]) * X[i]
                db = 2 * (y_pred - y[i])
                
                mw = beta1 * mw + (1 - beta1) * dw
                mb = beta1 * mb + (1 - beta1) * db
                vw = beta2 * vw + (1 - beta2) * dw**2
                vb = beta2 * vb + (1 - beta2) * db**2
                
                mw_hat = mw / (1 - beta1**t)
                mb_hat = mb / (1 - beta1**t)
                vw_hat = vw / (1 - beta2**t)
                vb_hat = vb / (1 - beta2**t)
                
                w = w - lr * mw_hat / (np.sqrt(vw_hat) + eps)
                b = b - lr * mb_hat / (np.sqrt(vb_hat) + eps)
        
        return losses
    
    # Train with different optimizers
    n_epochs = 20
    
    losses_batch = train_batch_gd(X, y, lr=0.1, n_epochs=n_epochs)
    losses_sgd = train_sgd(X, y, lr=0.001, n_epochs=n_epochs)
    losses_momentum = train_sgd_momentum(X, y, lr=0.001, momentum=0.9, n_epochs=n_epochs)
    losses_adam = train_adam(X, y, lr=0.01, n_epochs=n_epochs)
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(losses_batch, 'b-', linewidth=2, label='Batch GD (lr=0.1)')
    ax.plot(losses_sgd, 'r-', linewidth=2, label='SGD (lr=0.001)')
    ax.plot(losses_momentum, 'g-', linewidth=2, label='SGD+Momentum (lr=0.001, m=0.9)')
    ax.plot(losses_adam, 'purple', linewidth=2, label='Adam (lr=0.01)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Optimizer Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'comparison.png'}")
    
    print("""
Optimizer Recommendations:

| Optimizer      | Use When                          |
|----------------|-----------------------------------|
| Batch GD       | Small datasets, need stability    |
| SGD            | Large datasets, simple baseline   |
| SGD + Momentum | Most cases, good default          |
| Adam           | Deep learning default, works well |
| AdamW          | Adam with proper weight decay     |
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all gradient descent demonstrations."""
    print("=" * 60)
    print("GRADIENT DESCENT DEEP DIVE")
    print("=" * 60)
    print("\nUnderstanding the optimization algorithms that train ML models.")
    
    X, y, batch_history = demonstrate_batch_gd()
    sgd_history = demonstrate_sgd(X, y)
    mini_batch_history = demonstrate_mini_batch(X, y)
    demonstrate_momentum()
    demonstrate_adam()
    demonstrate_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. BATCH GD: Use all data for each update
   - Stable but slow for large datasets

2. SGD: Use one example per update
   - Fast but noisy

3. MINI-BATCH: Use small batches (32-256)
   - Best of both worlds
   - Enables GPU parallelization

4. MOMENTUM: Accumulate velocity
   - Accelerates in consistent directions
   - Dampens oscillations

5. ADAM: Adaptive learning rates + momentum
   - Works well out of the box
   - Default choice for deep learning

The optimization formula:
   theta_new = theta_old - learning_rate * gradient

Everything else is about:
   - How to compute the gradient (batch/mini-batch/stochastic)
   - How to adjust the update (momentum, adaptive lr)
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
