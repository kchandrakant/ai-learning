"""
Linear Algebra Essentials for Machine Learning
===============================================

This module provides interactive examples and visualizations for
understanding linear algebra concepts used in ML.

Topics covered:
- Vectors and vector operations
- Dot products and their geometric meaning
- Matrices and matrix multiplication
- Transformations and eigenvalues

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# PART 1: VECTORS
# =============================================================================

def demonstrate_vectors():
    """Visualize vectors and basic operations."""
    print("\n" + "=" * 60)
    print("PART 1: VECTORS")
    print("=" * 60)
    
    # Create vectors
    a = np.array([3, 2])
    b = np.array([1, 4])
    
    print("\nVector a:", a)
    print("Vector b:", b)
    
    # Vector addition
    c = a + b
    print("\na + b =", c)
    
    # Scalar multiplication
    scaled = 2 * a
    print("2 * a =", scaled)
    
    # Magnitude (length)
    magnitude_a = np.linalg.norm(a)
    print(f"\n||a|| = sqrt(3^2 + 2^2) = {magnitude_a:.3f}")
    
    # Unit vector (direction)
    unit_a = a / magnitude_a
    print(f"Unit vector a_hat = a/||a|| = {unit_a}")
    print(f"||a_hat|| = {np.linalg.norm(unit_a):.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Basic vectors
    ax = axes[0]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label=f'a = {a}')
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, 
              color='red', label=f'b = {b}')
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Vectors as Arrows')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 2: Vector addition
    ax = axes[1]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label='a', width=0.02)
    ax.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1, 
              color='red', label='b (from a)', width=0.02)
    ax.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, 
              color='purple', label=f'a + b = {c}', width=0.02)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Vector Addition (Tip-to-Tail)')
    
    # Plot 3: Scalar multiplication
    ax = axes[2]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label='a', width=0.02)
    ax.quiver(0, 0, scaled[0], scaled[1], angles='xy', scale_units='xy', scale=1, 
              color='green', label='2a', width=0.02, alpha=0.7)
    ax.quiver(0, 0, -a[0], -a[1], angles='xy', scale_units='xy', scale=1, 
              color='orange', label='-a', width=0.02)
    ax.set_xlim(-4, 8)
    ax.set_ylim(-3, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Scalar Multiplication')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'vectors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'vectors.png'}")


# =============================================================================
# PART 2: DOT PRODUCT
# =============================================================================

def demonstrate_dot_product():
    """Visualize dot product and its geometric meaning."""
    print("\n" + "=" * 60)
    print("PART 2: DOT PRODUCT")
    print("=" * 60)
    
    # Create vectors
    a = np.array([4, 2])
    b = np.array([1, 3])
    
    # Compute dot product
    dot_ab = np.dot(a, b)
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"\na . b = {a[0]}*{b[0]} + {a[1]}*{b[1]} = {dot_ab}")
    
    # Geometric interpretation
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    cos_theta = dot_ab / (mag_a * mag_b)
    theta = np.arccos(cos_theta)
    
    print(f"\n||a|| = {mag_a:.3f}")
    print(f"||b|| = {mag_b:.3f}")
    print(f"cos(theta) = (a . b) / (||a|| * ||b||) = {cos_theta:.3f}")
    print(f"theta = {np.degrees(theta):.1f} degrees")
    
    # Show different angle cases
    print("\n--- Dot Product Sign Interpretation ---")
    print("a . b > 0: Vectors point in similar directions (angle < 90 deg)")
    print("a . b = 0: Vectors are perpendicular (angle = 90 deg)")
    print("a . b < 0: Vectors point in opposite directions (angle > 90 deg)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Case 1: Positive dot product
    ax = axes[0]
    v1, v2 = np.array([3, 1]), np.array([2, 2])
    dot_val = np.dot(v1, v2)
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.03)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'Positive: a.b = {dot_val}\n(Same direction)')
    
    # Case 2: Zero dot product (perpendicular)
    ax = axes[1]
    v1, v2 = np.array([3, 0]), np.array([0, 2])
    dot_val = np.dot(v1, v2)
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.03)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'Zero: a.b = {dot_val}\n(Perpendicular)')
    
    # Case 3: Negative dot product
    ax = axes[2]
    v1, v2 = np.array([3, 1]), np.array([-2, 1])
    dot_val = np.dot(v1, v2)
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.03)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
    ax.set_xlim(-3, 5)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'Negative: a.b = {dot_val}\n(Opposite directions)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'dot_product.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'dot_product.png'}")
    
    # ML Application: Similarity
    print("\n--- ML Application: Similarity ---")
    doc1 = np.array([3, 0, 2, 1])  # Word counts for document 1
    doc2 = np.array([2, 1, 3, 0])  # Word counts for document 2
    doc3 = np.array([0, 5, 0, 4])  # Word counts for document 3
    
    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print(f"Document 1 features: {doc1}")
    print(f"Document 2 features: {doc2}")
    print(f"Document 3 features: {doc3}")
    print(f"\nCosine similarity (doc1, doc2): {cosine_similarity(doc1, doc2):.3f}")
    print(f"Cosine similarity (doc1, doc3): {cosine_similarity(doc1, doc3):.3f}")
    print("-> Higher similarity means more similar content!")


# =============================================================================
# PART 3: MATRICES
# =============================================================================

def demonstrate_matrices():
    """Show matrix operations and their role in ML."""
    print("\n" + "=" * 60)
    print("PART 3: MATRICES")
    print("=" * 60)
    
    # Create matrices
    A = np.array([[1, 2], 
                  [3, 4]])
    B = np.array([[5, 6], 
                  [7, 8]])
    
    print("\nMatrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    # Matrix multiplication
    C = A @ B  # or np.dot(A, B)
    print("\nA @ B =")
    print(C)
    
    # Show computation
    print("\nHow matrix multiplication works:")
    print(f"C[0,0] = A[0,:] . B[:,0] = {A[0,:]} . {B[:,0]} = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
    print(f"C[0,1] = A[0,:] . B[:,1] = {A[0,:]} . {B[:,1]} = {A[0,0]*B[0,1] + A[0,1]*B[1,1]}")
    
    # Shape rules
    print("\n--- Shape Rules ---")
    M1 = np.random.randn(3, 4)
    M2 = np.random.randn(4, 2)
    result = M1 @ M2
    print(f"(3x4) @ (4x2) = (3x2)")
    print(f"Inner dimensions must match: 4 = 4 [OK]")
    print(f"Outer dimensions give result shape")
    
    # Special matrices
    print("\n--- Special Matrices ---")
    I = np.eye(3)
    print("Identity matrix I (3x3):")
    print(I)
    print("Property: A @ I = I @ A = A")
    
    # Transpose
    print("\nTranspose (flip rows and columns):")
    print(f"A =\n{A}")
    print(f"A.T =\n{A.T}")
    
    # Inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse A^(-1):")
    print(A_inv)
    print(f"\nA @ A^(-1) =")
    print(np.round(A @ A_inv, 10))  # Should be identity
    
    # Neural network layer simulation
    print("\n--- ML Application: Neural Network Layer ---")
    # Input: batch of 4 samples, 3 features each
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]])
    
    # Weights: 3 inputs -> 2 outputs
    W = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6]])
    
    # Bias
    b = np.array([0.1, 0.2])
    
    # Forward pass: Y = X @ W + b
    Y = X @ W + b
    
    print(f"Input X shape: {X.shape} (4 samples, 3 features)")
    print(f"Weights W shape: {W.shape} (3 inputs, 2 outputs)")
    print(f"Output Y shape: {Y.shape} (4 samples, 2 outputs)")
    print(f"\nThis is exactly what a neural network layer does!")


# =============================================================================
# PART 4: LINEAR TRANSFORMATIONS
# =============================================================================

def demonstrate_transformations():
    """Visualize how matrices transform space."""
    print("\n" + "=" * 60)
    print("PART 4: LINEAR TRANSFORMATIONS")
    print("=" * 60)
    
    # Create a grid of points
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    
    # Define transformation matrices
    transformations = {
        'Rotation (45 deg)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        'Scaling (2x, 0.5y)': np.array([[2, 0],
                                         [0, 0.5]]),
        'Shear': np.array([[1, 0.5],
                           [0, 1]]),
        'Reflection (y-axis)': np.array([[-1, 0],
                                          [0, 1]])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (name, matrix) in enumerate(transformations.items()):
        ax = axes[idx]
        
        # Original grid
        ax.scatter(points[0], points[1], c='blue', alpha=0.3, s=10, label='Original')
        
        # Transformed grid
        transformed = matrix @ points
        ax.scatter(transformed[0], transformed[1], c='red', alpha=0.5, s=10, label='Transformed')
        
        # Show basis vectors
        e1 = np.array([1, 0])
        e2 = np.array([0, 1])
        e1_transformed = matrix @ e1
        e2_transformed = matrix @ e2
        
        ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
                  color='blue', width=0.03, alpha=0.7)
        ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
                  color='blue', width=0.03, alpha=0.7)
        ax.quiver(0, 0, e1_transformed[0], e1_transformed[1], angles='xy', 
                  scale_units='xy', scale=1, color='red', width=0.03)
        ax.quiver(0, 0, e2_transformed[0], e2_transformed[1], angles='xy', 
                  scale_units='xy', scale=1, color='red', width=0.03)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.legend()
        ax.set_title(f'{name}\n{matrix[0]}\n{matrix[1]}')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'transformations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'transformations.png'}")
    
    print("\nKey insight: Matrices are TRANSFORMATIONS of space!")
    print("- Rotation: rotates points around origin")
    print("- Scaling: stretches/compresses along axes")
    print("- Shear: slants the space")
    print("- Reflection: mirrors across an axis")


# =============================================================================
# PART 5: EIGENVALUES AND EIGENVECTORS
# =============================================================================

def demonstrate_eigenvalues():
    """Visualize eigenvalues and eigenvectors."""
    print("\n" + "=" * 60)
    print("PART 5: EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)
    
    # Create a matrix
    A = np.array([[3, 1],
                  [0, 2]])
    
    print("Matrix A:")
    print(A)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (columns):")
    print(eigenvectors)
    
    # Verify: A @ v = lambda * v
    print("\n--- Verification: A*v = lambda*v ---")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        Av = A @ v
        lam_v = lam * v
        print(f"\nEigenvector {i+1}: v = {v}")
        print(f"Eigenvalue {i+1}: lambda = {lam}")
        print(f"A @ v = {Av}")
        print(f"lambda * v = {lam_v}")
        print(f"Equal? {np.allclose(Av, lam_v)}")
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw unit circle to show transformation
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    transformed = A @ circle
    
    ax.plot(circle[0], circle[1], 'b-', alpha=0.5, label='Unit circle')
    ax.plot(transformed[0], transformed[1], 'r-', alpha=0.7, label='Transformed')
    
    # Draw eigenvectors
    colors = ['green', 'purple']
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        # Original eigenvector
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                  color=colors[i], width=0.02, label=f'v{i+1} (lambda={lam:.1f})')
        # Transformed (just scaled)
        ax.quiver(0, 0, lam*v[0], lam*v[1], angles='xy', scale_units='xy', scale=1, 
                  color=colors[i], width=0.02, alpha=0.5, linestyle='--')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Eigenvectors: Only Scaled, Not Rotated by A')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'eigenvalues.png'}")
    
    print("\n--- Key Insight ---")
    print("Eigenvectors are special directions that only get SCALED by the matrix.")
    print("The eigenvalue tells you the scaling factor.")
    print("\nML Applications:")
    print("- PCA: Eigenvectors of covariance matrix are principal components")
    print("- PageRank: Eigenvector of link matrix gives page importance")
    print("- Spectral clustering: Eigenvectors of graph Laplacian")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("LINEAR ALGEBRA FOR MACHINE LEARNING")
    print("=" * 60)
    print("\nThis module demonstrates the linear algebra foundations")
    print("that power machine learning algorithms.")
    
    demonstrate_vectors()
    demonstrate_dot_product()
    demonstrate_matrices()
    demonstrate_transformations()
    demonstrate_eigenvalues()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:
1. Vectors: Ordered lists of numbers (features in ML)
2. Dot product: Measures similarity, projects one vector onto another
3. Matrix multiplication: Many dot products at once
4. Transformations: Matrices transform space (rotate, scale, shear)
5. Eigenvalues/vectors: Special directions that only scale under transformation

Neural Network Connection:
    y = activation(W @ x + b)
    
This is just:
    - Matrix multiplication (W @ x)
    - Vector addition (+ b)
    - Element-wise function (activation)
    
All linear algebra!
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
