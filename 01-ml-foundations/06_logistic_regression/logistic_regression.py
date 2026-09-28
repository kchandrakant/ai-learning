"""
Logistic Regression & Classification
=====================================

This module implements logistic regression from scratch:
- Sigmoid function
- Cross-entropy loss
- Gradient descent for classification
- Decision boundaries
- Multi-class with softmax

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: THE SIGMOID FUNCTION
# =============================================================================

def demonstrate_sigmoid():
    """Introduce the sigmoid activation function."""
    print("\n" + "=" * 60)
    print("PART 1: THE SIGMOID FUNCTION")
    print("=" * 60)
    
    print("""
The Problem with Linear Regression for Classification:
    - Linear regression outputs any real number
    - Classification needs probabilities (0 to 1)
    
Solution: The Sigmoid Function

    sigmoid(z) = 1 / (1 + e^(-z))
    
Properties:
    - Always outputs between 0 and 1
    - sigmoid(0) = 0.5
    - sigmoid(large positive) -> 1
    - sigmoid(large negative) -> 0
""")
    
    def sigmoid(z):
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-z))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sigmoid function
    ax = axes[0]
    z = np.linspace(-10, 10, 100)
    ax.plot(z, sigmoid(z), 'b-', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate key points
    ax.scatter([0], [0.5], color='red', s=100, zorder=5)
    ax.annotate('sigmoid(0) = 0.5', xy=(0, 0.5), xytext=(2, 0.6), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('z')
    ax.set_ylabel('sigmoid(z)')
    ax.set_title('The Sigmoid Function\nsigmoid(z) = 1 / (1 + e^(-z))')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 2: Sigmoid derivative
    ax = axes[1]
    sigmoid_derivative = sigmoid(z) * (1 - sigmoid(z))
    ax.plot(z, sigmoid(z), 'b-', linewidth=2, label='sigmoid(z)')
    ax.plot(z, sigmoid_derivative, 'r-', linewidth=2, label="sigmoid'(z)")
    ax.set_xlabel('z')
    ax.set_ylabel('value')
    ax.set_title("Sigmoid and Its Derivative\nsigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'sigmoid.png'}")
    
    # Show example values
    print("\nExample values:")
    for z_val in [-5, -2, 0, 2, 5]:
        print(f"  sigmoid({z_val:2d}) = {sigmoid(z_val):.4f}")


# =============================================================================
# PART 2: LOGISTIC REGRESSION MODEL
# =============================================================================

def demonstrate_logistic_model():
    """Show the logistic regression model."""
    print("\n" + "=" * 60)
    print("PART 2: LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    print("""
Logistic Regression:

    z = w * x + b           (linear combination)
    P(y=1|x) = sigmoid(z)   (probability)
    
Prediction:
    - If P(y=1|x) >= 0.5: predict class 1
    - If P(y=1|x) < 0.5:  predict class 0
    
Decision Boundary: where P = 0.5, i.e., z = 0
    w * x + b = 0
""")
    
    # Generate 2D data for visualization
    np.random.seed(42)
    n = 100
    X_class0 = np.random.randn(n, 2) + np.array([-1, -1])
    X_class1 = np.random.randn(n, 2) + np.array([1, 1])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n + [1] * n)
    
    # Train logistic regression
    model = LogisticRegression()
    model.fit(X, y)
    
    # Get coefficients
    w = model.coef_[0]
    b = model.intercept_[0]
    print(f"\nTrained model:")
    print(f"  w = [{w[0]:.3f}, {w[1]:.3f}]")
    print(f"  b = {b:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and decision boundary
    ax = axes[0]
    ax.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', alpha=0.7, label='Class 0')
    ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', alpha=0.7, label='Class 1')
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    x1_line = np.linspace(-4, 4, 100)
    x2_line = -(w[0] * x1_line + b) / w[1]
    ax.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision boundary')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Logistic Regression Decision Boundary\n(Where P(y=1) = 0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    
    # Plot 2: Probability surface
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='P(y=1)')
    ax.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', alpha=0.5, edgecolors='white', s=30)
    ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', alpha=0.5, edgecolors='white', s=30)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Predicted Probability Surface\n(Black line = decision boundary)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'logistic_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'logistic_model.png'}")


# =============================================================================
# PART 3: CROSS-ENTROPY LOSS
# =============================================================================

def demonstrate_cross_entropy():
    """Explain and visualize cross-entropy loss."""
    print("\n" + "=" * 60)
    print("PART 3: CROSS-ENTROPY LOSS")
    print("=" * 60)
    
    print("""
Binary Cross-Entropy Loss:

    L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )
    
Where:
    - y is true label (0 or 1)
    - p is predicted probability
    
Intuition:
    - If y=1 and p=0.9: loss = -log(0.9) = 0.105 (small)
    - If y=1 and p=0.1: loss = -log(0.1) = 2.303 (large!)
    
Cross-entropy penalizes confident wrong predictions severely.
""")
    
    def cross_entropy(y_true, y_pred):
        """Compute binary cross-entropy loss."""
        eps = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Visualize loss vs predicted probability
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss for y=1
    ax = axes[0]
    p = np.linspace(0.01, 0.99, 100)
    loss_y1 = -np.log(p)  # When y=1
    loss_y0 = -np.log(1 - p)  # When y=0
    
    ax.plot(p, loss_y1, 'b-', linewidth=2, label='Loss when y=1: -log(p)')
    ax.plot(p, loss_y0, 'r-', linewidth=2, label='Loss when y=0: -log(1-p)')
    ax.set_xlabel('Predicted probability p')
    ax.set_ylabel('Loss')
    ax.set_title('Cross-Entropy Loss\n(Penalizes confident wrong predictions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    
    # Annotate
    ax.annotate('Good: y=1, p=0.9\nLoss=0.11', xy=(0.9, -np.log(0.9)), xytext=(0.7, 1.5),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Bad: y=1, p=0.1\nLoss=2.30', xy=(0.1, -np.log(0.1)), xytext=(0.3, 3),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Plot 2: Why not MSE for classification?
    ax = axes[1]
    mse_y1 = (1 - p)**2  # MSE when y=1
    
    ax.plot(p, loss_y1, 'b-', linewidth=2, label='Cross-entropy (y=1)')
    ax.plot(p, mse_y1, 'r--', linewidth=2, label='MSE (y=1)')
    ax.set_xlabel('Predicted probability p')
    ax.set_ylabel('Loss')
    ax.set_title('Cross-Entropy vs MSE\n(CE gives stronger gradients for wrong predictions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cross_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'cross_entropy.png'}")


# =============================================================================
# PART 4: TRAINING FROM SCRATCH
# =============================================================================

def demonstrate_training():
    """Implement logistic regression training from scratch."""
    print("\n" + "=" * 60)
    print("PART 4: TRAINING FROM SCRATCH")
    print("=" * 60)
    
    print("""
Gradient Descent for Logistic Regression:

    Forward pass:
        z = X @ w + b
        p = sigmoid(z)
        loss = cross_entropy(y, p)
    
    Gradients:
        dw = (1/n) * X^T @ (p - y)
        db = (1/n) * sum(p - y)
    
    Update:
        w = w - lr * dw
        b = b - lr * db
""")
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def train_logistic_regression(X, y, lr=0.1, n_iterations=1000):
        """Train logistic regression using gradient descent."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        
        history = {'loss': [], 'accuracy': []}
        
        for i in range(n_iterations):
            # Forward pass
            z = X @ w + b
            p = sigmoid(z)
            
            # Compute loss (with numerical stability)
            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            
            # Compute accuracy
            predictions = (p >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            # Compute gradients
            dw = (1 / n_samples) * (X.T @ (p - y))
            db = np.mean(p - y)
            
            # Update parameters
            w = w - lr * dw
            b = b - lr * db
        
        return w, b, history
    
    # Generate data
    np.random.seed(42)
    n = 200
    X_class0 = np.random.randn(n, 2) + np.array([-1.5, -1.5])
    X_class1 = np.random.randn(n, 2) + np.array([1.5, 1.5])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n + [1] * n)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    # Train
    w, b, history = train_logistic_regression(X, y, lr=0.5, n_iterations=100)
    
    print(f"\nTrained parameters:")
    print(f"  w = [{w[0]:.3f}, {w[1]:.3f}]")
    print(f"  b = {b:.3f}")
    print(f"  Final accuracy: {history['accuracy'][-1]:.1%}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training curves
    ax = axes[0]
    ax.plot(history['loss'], 'b-', linewidth=2, label='Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Accuracy
    ax = axes[1]
    ax.plot(history['accuracy'], 'g-', linewidth=2, label='Accuracy')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Final decision boundary
    ax = axes[2]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.5, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.5, label='Class 1')
    
    # Decision boundary
    x1_line = np.linspace(-5, 5, 100)
    x2_line = -(w[0] * x1_line + b) / w[1]
    ax.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision boundary')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Final Result\nAccuracy: {history["accuracy"][-1]:.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'training.png'}")


# =============================================================================
# PART 5: MULTI-CLASS CLASSIFICATION (SOFTMAX)
# =============================================================================

def demonstrate_softmax():
    """Show softmax for multi-class classification."""
    print("\n" + "=" * 60)
    print("PART 5: MULTI-CLASS CLASSIFICATION (SOFTMAX)")
    print("=" * 60)
    
    print("""
Softmax Function (for K classes):

    softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
    
Properties:
    - Outputs sum to 1 (probability distribution)
    - Each output is in [0, 1]
    - Generalizes sigmoid to multiple classes
""")
    
    def softmax(z):
        """Compute softmax (numerically stable)."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Example
    z = np.array([[2.0, 1.0, 0.5]])  # Raw scores for 3 classes
    probs = softmax(z)
    
    print(f"\nExample:")
    print(f"  Raw scores z = {z[0]}")
    print(f"  Softmax(z) = {probs[0]}")
    print(f"  Sum = {probs.sum():.4f}")
    print(f"  Predicted class: {np.argmax(probs)}")
    
    # Generate multi-class data
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
    
    # Train multi-class logistic regression
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Decision regions
    ax = axes[0]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='white')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Multi-class Decision Regions\n(3 classes)')
    
    # Plot 2: Softmax output example
    ax = axes[1]
    classes = ['Cat', 'Dog', 'Bird']
    probs_example = [0.7, 0.2, 0.1]
    
    bars = ax.bar(classes, probs_example, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_ylabel('Probability')
    ax.set_title('Softmax Output Example\n(Probabilities sum to 1)')
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probs_example):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{prob:.1%}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'softmax.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'softmax.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all logistic regression demonstrations."""
    print("=" * 60)
    print("LOGISTIC REGRESSION & CLASSIFICATION")
    print("=" * 60)
    print("\nFrom regression to classification with the sigmoid function.")
    
    demonstrate_sigmoid()
    demonstrate_logistic_model()
    demonstrate_cross_entropy()
    demonstrate_training()
    demonstrate_softmax()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. SIGMOID: Squashes linear output to [0, 1]
   sigmoid(z) = 1 / (1 + e^(-z))

2. LOGISTIC REGRESSION:
   P(y=1|x) = sigmoid(w*x + b)
   Output is a probability!

3. CROSS-ENTROPY LOSS:
   L = -mean(y*log(p) + (1-y)*log(1-p))
   Penalizes confident wrong predictions

4. DECISION BOUNDARY:
   Where P(y=1) = 0.5
   Linear boundary in feature space

5. SOFTMAX for multi-class:
   Generalizes sigmoid to K classes
   Outputs sum to 1

Connection to Neural Networks:
   - Sigmoid is an "activation function"
   - Softmax is used for classification output
   - Cross-entropy is the standard classification loss
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
