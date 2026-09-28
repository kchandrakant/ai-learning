"""
Calculus for Machine Learning
=============================

This module demonstrates calculus concepts essential for ML:
- Derivatives and their geometric meaning
- Partial derivatives and gradients
- The chain rule (critical for backpropagation!)
- Gradient descent visualization

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# PART 1: DERIVATIVES
# =============================================================================

def demonstrate_derivatives():
    """Visualize what derivatives mean geometrically."""
    print("\n" + "=" * 60)
    print("PART 1: DERIVATIVES")
    print("=" * 60)
    
    # Define a simple function
    def f(x):
        return x**2
    
    def f_derivative(x):
        return 2*x
    
    print("\nFunction: f(x) = x^2")
    print("Derivative: f'(x) = 2x")
    print("\nThe derivative gives the SLOPE at any point:")
    
    for x_val in [-2, 0, 1, 3]:
        slope = f_derivative(x_val)
        print(f"  At x={x_val}: slope = 2*{x_val} = {slope}")
    
    # Visualize
    x = np.linspace(-3, 3, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Function and tangent lines
    ax = axes[0]
    ax.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x^2')
    
    # Draw tangent lines at specific points
    points = [-2, 0, 2]
    colors = ['red', 'green', 'purple']
    for x0, color in zip(points, colors):
        y0 = f(x0)
        slope = f_derivative(x0)
        
        # Tangent line: y - y0 = slope * (x - x0)
        x_tangent = np.linspace(x0 - 1.5, x0 + 1.5, 50)
        y_tangent = y0 + slope * (x_tangent - x0)
        
        ax.plot(x_tangent, y_tangent, color=color, linestyle='--', 
                label=f'Tangent at x={x0} (slope={slope})')
        ax.scatter([x0], [y0], color=color, s=100, zorder=5)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 9)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Derivative = Slope of Tangent Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Function and its derivative
    ax = axes[1]
    ax.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x^2')
    ax.plot(x, f_derivative(x), 'r-', linewidth=2, label="f'(x) = 2x")
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Annotate
    ax.annotate("f'(x) < 0\nfunction decreasing", xy=(-2, f(-2)), 
                xytext=(-2.5, 7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate("f'(x) > 0\nfunction increasing", xy=(2, f(2)), 
                xytext=(1.5, 7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate("f'(x) = 0\nlocal minimum!", xy=(0, f(0)), 
                xytext=(0.5, 2), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-5, 9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Function vs Its Derivative')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'derivatives.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'derivatives.png'}")
    
    # Common derivatives
    print("\n--- Common Derivatives (Memorize These!) ---")
    print("f(x) = c       -> f'(x) = 0")
    print("f(x) = x^n     -> f'(x) = n * x^(n-1)")
    print("f(x) = e^x     -> f'(x) = e^x")
    print("f(x) = ln(x)   -> f'(x) = 1/x")
    print("f(x) = sin(x)  -> f'(x) = cos(x)")
    print("f(x) = sigmoid(x) -> f'(x) = sigmoid(x) * (1 - sigmoid(x))")


# =============================================================================
# PART 2: THE CHAIN RULE
# =============================================================================

def demonstrate_chain_rule():
    """The chain rule - the heart of backpropagation."""
    print("\n" + "=" * 60)
    print("PART 2: THE CHAIN RULE")
    print("=" * 60)
    
    print("""
The Chain Rule: For f(g(x)), the derivative is
    
    d/dx[f(g(x))] = f'(g(x)) * g'(x)
    
    "Derivative of outside * derivative of inside"
""")
    
    # Example 1: Simple composed function
    print("--- Example 1: f(x) = (3x + 2)^2 ---")
    print("Let u = 3x + 2, so f = u^2")
    print("f'(x) = d(u^2)/du * du/dx")
    print("     = 2u * 3")
    print("     = 2(3x + 2) * 3")
    print("     = 6(3x + 2)")
    
    # Verify numerically
    def f1(x): return (3*x + 2)**2
    def f1_derivative(x): return 6*(3*x + 2)
    
    x_test = 2.0
    numerical = (f1(x_test + 0.0001) - f1(x_test)) / 0.0001
    analytical = f1_derivative(x_test)
    print(f"\nAt x=2: Numerical = {numerical:.4f}, Analytical = {analytical:.4f}")
    
    # Example 2: Neural network layer (this is backprop!)
    print("\n--- Example 2: Neural Network (Why This Matters!) ---")
    print("""
Consider a simple network:
    
    Input x -> Linear: z = wx + b -> Sigmoid: a = sigmoid(z) -> Loss: L = (a - y)^2
    
To train, we need dL/dw. Apply chain rule:
    
    dL/dw = dL/da * da/dz * dz/dw
    
Each piece:
    dL/da = 2(a - y)           [derivative of squared error]
    da/dz = sigmoid(z)*(1-sigmoid(z))  [derivative of sigmoid]
    dz/dw = x                  [derivative of wx + b wrt w]
    
So: dL/dw = 2(a - y) * sigmoid(z)*(1-sigmoid(z)) * x

THIS IS BACKPROPAGATION! Just chain rule through the network.
""")
    
    # Visualize the computation graph
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Draw boxes for each operation
    boxes = [
        (1, 0.5, 'x\n(input)'),
        (3, 0.5, 'z = wx + b\n(linear)'),
        (5, 0.5, 'a = sig(z)\n(activation)'),
        (7, 0.5, 'L = (a-y)^2\n(loss)')
    ]
    
    for x_pos, y_pos, text in boxes:
        ax.add_patch(plt.Rectangle((x_pos - 0.5, y_pos - 0.3), 1.5, 0.6, 
                                    fill=True, facecolor='lightblue', edgecolor='blue'))
        ax.text(x_pos + 0.25, y_pos, text, ha='center', va='center', fontsize=10)
    
    # Forward arrows
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1][0] - 0.5, boxes[i+1][1]), 
                    xytext=(boxes[i][0] + 1, boxes[i][1]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Backward arrows (gradients)
    for i in range(len(boxes) - 1, 0, -1):
        ax.annotate('', xy=(boxes[i-1][0] + 1, boxes[i-1][1] - 0.1), 
                    xytext=(boxes[i][0] - 0.5, boxes[i][1] - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Labels
    ax.text(2, 0.8, 'dz/dw = x', fontsize=9, color='red')
    ax.text(4, 0.8, "da/dz = sig'(z)", fontsize=9, color='red')
    ax.text(6, 0.8, 'dL/da = 2(a-y)', fontsize=9, color='red')
    
    ax.text(4, -0.1, 'Forward: compute predictions (green)', fontsize=11, ha='center', color='green')
    ax.text(4, -0.3, 'Backward: compute gradients using chain rule (red)', fontsize=11, ha='center', color='red')
    
    ax.set_xlim(0, 8.5)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Backpropagation = Chain Rule Through Computation Graph', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'chain_rule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'chain_rule.png'}")


# =============================================================================
# PART 3: PARTIAL DERIVATIVES AND GRADIENTS
# =============================================================================

def demonstrate_gradients():
    """Partial derivatives and the gradient vector."""
    print("\n" + "=" * 60)
    print("PART 3: PARTIAL DERIVATIVES AND GRADIENTS")
    print("=" * 60)
    
    print("""
When f depends on multiple variables, take derivative with respect 
to each one separately (treating others as constants).

Example: f(x, y) = x^2 + 2xy + y^2

    df/dx = 2x + 2y    (treat y as constant)
    df/dy = 2x + 2y    (treat x as constant)

The GRADIENT is the vector of all partial derivatives:
    grad_f = [df/dx, df/dy]
""")
    
    # Define a 2D function (like a loss surface)
    def f(x, y):
        return x**2 + y**2  # Bowl shape - simplest loss function
    
    def gradient(x, y):
        return np.array([2*x, 2*y])
    
    # Create grid for visualization
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Loss Surface: f(x,y) = x^2 + y^2')
    
    # Plot 2: Contour with gradient arrows
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=15, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Draw gradient vectors at several points
    points = [(-2, -2), (-2, 1), (1, 2), (2, -1), (0, 2)]
    for px, py in points:
        grad = gradient(px, py)
        # Normalize for visualization
        grad_norm = grad / np.linalg.norm(grad) * 0.8
        ax2.arrow(px, py, grad_norm[0], grad_norm[1], head_width=0.15, 
                  head_length=0.1, fc='red', ec='red')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gradient Points UPHILL\n(red arrows)')
    ax2.set_aspect('equal')
    
    # Plot 3: Gradient descent path
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)
    
    # Run gradient descent
    lr = 0.1
    point = np.array([2.5, 2.5])
    path = [point.copy()]
    
    for _ in range(20):
        grad = gradient(point[0], point[1])
        point = point - lr * grad  # Move OPPOSITE to gradient (downhill)
        path.append(point.copy())
    
    path = np.array(path)
    ax3.plot(path[:, 0], path[:, 1], 'ro-', markersize=5, label='GD path')
    ax3.scatter([0], [0], color='green', s=100, marker='*', label='Minimum')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Gradient Descent: Move Opposite to Gradient\ntheta_new = theta_old - lr * grad')
    ax3.legend()
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gradients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'gradients.png'}")
    
    print("\n--- Key Insight ---")
    print("The gradient grad_f points in the direction of STEEPEST INCREASE.")
    print("To MINIMIZE: move in the OPPOSITE direction (-grad_f).")
    print("This is gradient descent!")


# =============================================================================
# PART 4: GRADIENT DESCENT IN ACTION
# =============================================================================

def demonstrate_gradient_descent():
    """Visualize gradient descent on different loss surfaces."""
    print("\n" + "=" * 60)
    print("PART 4: GRADIENT DESCENT IN ACTION")
    print("=" * 60)
    
    # Define loss functions with different characteristics
    def quadratic(x):
        """Simple bowl - single global minimum"""
        return x**2
    
    def quadratic_grad(x):
        return 2*x
    
    def non_convex(x):
        """Has local minima"""
        return x**4 - 3*x**2 + x
    
    def non_convex_grad(x):
        return 4*x**3 - 6*x + 1
    
    def run_gradient_descent(f, grad, x_init, lr, n_steps):
        """Run gradient descent and return path."""
        x = x_init
        path = [x]
        losses = [f(x)]
        
        for _ in range(n_steps):
            x = x - lr * grad(x)
            path.append(x)
            losses.append(f(x))
        
        return np.array(path), np.array(losses)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gradient descent on quadratic
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 100)
    ax.plot(x, quadratic(x), 'b-', linewidth=2, label='f(x) = x^2')
    
    path, losses = run_gradient_descent(quadratic, quadratic_grad, 2.5, 0.1, 20)
    ax.plot(path, [quadratic(p) for p in path], 'ro-', markersize=5, label='GD path')
    ax.scatter([0], [0], color='green', s=100, marker='*', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Convex Function: Converges to Global Minimum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate comparison
    ax = axes[0, 1]
    ax.plot(x, quadratic(x), 'b-', linewidth=2, alpha=0.5)
    
    for lr, color, label in [(0.01, 'green', 'lr=0.01 (slow)'), 
                              (0.1, 'blue', 'lr=0.1 (good)'),
                              (0.9, 'red', 'lr=0.9 (oscillates)')]:
        path, _ = run_gradient_descent(quadratic, quadratic_grad, 2.5, lr, 15)
        ax.plot(path, [quadratic(p) for p in path], f'{color}o-', 
                markersize=4, label=label, alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Effect of Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Non-convex function - local minima
    ax = axes[1, 0]
    x = np.linspace(-2, 2, 100)
    ax.plot(x, non_convex(x), 'b-', linewidth=2, label='f(x) = x^4 - 3x^2 + x')
    
    # Different starting points lead to different minima
    for x_init, color in [(-1.5, 'red'), (1.5, 'green')]:
        path, _ = run_gradient_descent(non_convex, non_convex_grad, x_init, 0.05, 50)
        ax.plot(path, [non_convex(p) for p in path], f'{color}o-', 
                markersize=4, label=f'Start: {x_init}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Non-Convex: Starting Point Matters!\n(Can get stuck in local minima)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss curve
    ax = axes[1, 1]
    
    path, losses = run_gradient_descent(quadratic, quadratic_grad, 2.5, 0.1, 30)
    ax.plot(losses, 'b-o', markersize=4, label='lr=0.1')
    
    path, losses = run_gradient_descent(quadratic, quadratic_grad, 2.5, 0.01, 30)
    ax.plot(losses, 'g-o', markersize=4, label='lr=0.01')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Decreases Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gradient_descent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'gradient_descent.png'}")
    
    print("""
Key Insights:
1. Learning rate too small -> slow convergence
2. Learning rate too large -> oscillation or divergence
3. Non-convex functions have local minima
4. Starting point can affect which minimum is found
5. Loss should decrease over training iterations
""")


# =============================================================================
# PART 5: NUMERICAL VS ANALYTICAL GRADIENTS
# =============================================================================

def demonstrate_numerical_gradient():
    """Compare numerical and analytical gradients."""
    print("\n" + "=" * 60)
    print("PART 5: NUMERICAL VS ANALYTICAL GRADIENTS")
    print("=" * 60)
    
    print("""
Numerical gradient: Approximate using finite differences
    
    df/dx approx= [f(x + eps) - f(x - eps)] / (2*eps)

This is useful for:
1. Checking your analytical gradient is correct (gradient checking)
2. When analytical gradient is hard to derive
""")
    
    def f(x, y):
        return x**2 * y + np.sin(x) * np.cos(y)
    
    def analytical_gradient(x, y):
        df_dx = 2*x*y + np.cos(x)*np.cos(y)
        df_dy = x**2 - np.sin(x)*np.sin(y)
        return np.array([df_dx, df_dy])
    
    def numerical_gradient(f, x, y, epsilon=1e-5):
        df_dx = (f(x + epsilon, y) - f(x - epsilon, y)) / (2 * epsilon)
        df_dy = (f(x, y + epsilon) - f(x, y - epsilon)) / (2 * epsilon)
        return np.array([df_dx, df_dy])
    
    # Test at a point
    x, y = 2.0, 1.0
    
    analytical = analytical_gradient(x, y)
    numerical = numerical_gradient(f, x, y)
    
    print(f"\nAt point (x={x}, y={y}):")
    print(f"  Analytical gradient: {analytical}")
    print(f"  Numerical gradient:  {numerical}")
    print(f"  Difference:          {np.abs(analytical - numerical)}")
    print(f"  Relative error:      {np.abs(analytical - numerical) / np.abs(analytical)}")
    
    print("\n--- Gradient Checking in Practice ---")
    print("""
When implementing backpropagation:
1. Compute gradient analytically (your implementation)
2. Compute gradient numerically (slow but reliable)
3. Compare: relative error should be < 1e-5

If they don't match, you have a bug in your gradient computation!
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all calculus demonstrations."""
    print("=" * 60)
    print("CALCULUS FOR MACHINE LEARNING")
    print("=" * 60)
    print("\nCalculus is the mathematics of change.")
    print("In ML, we use it to find parameters that minimize loss.")
    
    demonstrate_derivatives()
    demonstrate_chain_rule()
    demonstrate_gradients()
    demonstrate_gradient_descent()
    demonstrate_numerical_gradient()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. DERIVATIVE measures rate of change (slope)
   - f'(x) > 0: function increasing
   - f'(x) < 0: function decreasing
   - f'(x) = 0: possible minimum/maximum

2. CHAIN RULE: derivative of composed functions
   d/dx[f(g(x))] = f'(g(x)) * g'(x)
   
   THIS IS BACKPROPAGATION!

3. GRADIENT: vector of all partial derivatives
   Points in direction of steepest increase
   
4. GRADIENT DESCENT: theta_new = theta_old - lr * grad_L
   Move opposite to gradient to minimize loss
   
5. LEARNING RATE: 
   - Too small -> slow convergence
   - Too large -> oscillation/divergence
   - Just right -> efficient optimization

The entire training process is just:
    1. Forward pass: compute predictions and loss
    2. Backward pass: compute gradients using chain rule
    3. Update: move parameters opposite to gradient
    4. Repeat until loss is small enough
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
