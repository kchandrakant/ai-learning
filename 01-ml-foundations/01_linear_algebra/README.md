# Step 1: Linear Algebra Essentials

## Why Linear Algebra?

Neural networks are built on matrix operations. Every layer is:
```
output = activation(W × input + b)
```

Without linear algebra, ML is a black box.

## Vectors

A vector is an ordered list of numbers.

```python
import numpy as np

# Column vector (3 dimensions)
v = np.array([1, 2, 3])

# Vector addition
a = np.array([1, 2])
b = np.array([3, 4])
c = a + b  # [4, 6]

# Scalar multiplication
2 * a  # [2, 4]
```

### Geometric Interpretation
- Vector = arrow from origin to point
- Length (magnitude): ||v|| = √(v₁² + v₂² + ...)
- Direction: unit vector v/||v||

## Dot Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Or equivalently
dot = a @ b
```

### What Does It Mean?
```
a · b = ||a|| × ||b|| × cos(θ)

If a · b = 0 → vectors are perpendicular
If a · b > 0 → vectors point same general direction
If a · b < 0 → vectors point opposite directions
```

**In ML:** Dot product measures similarity between vectors.

## Matrices

A matrix is a 2D array of numbers.

```python
# 2x3 matrix (2 rows, 3 columns)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Shape
A.shape  # (2, 3)
```

## Matrix Multiplication

```python
A = np.array([[1, 2], [3, 4]])  # 2x2
B = np.array([[5, 6], [7, 8]])  # 2x2

C = A @ B  # Matrix multiplication
# C[i,j] = sum of (row i of A) × (column j of B)
```

**Rule:** (m × n) @ (n × p) = (m × p)

Inner dimensions must match!

### In Neural Networks
```python
# Input: 100 samples, 784 features
X = np.random.randn(100, 784)

# Weight matrix: 784 inputs → 256 outputs
W = np.random.randn(784, 256)

# Forward pass
output = X @ W  # Shape: (100, 256)
```

## Special Matrices

```python
# Identity matrix (I × A = A)
I = np.eye(3)

# Transpose (flip rows and columns)
A_T = A.T

# Inverse (A × A⁻¹ = I)
A_inv = np.linalg.inv(A)
```

## Eigenvalues and Eigenvectors (Intuition)

For a matrix A, eigenvector v satisfies:
```
A × v = λ × v
```

The matrix only scales v, doesn't change its direction.

**In ML:** PCA uses eigenvectors to find principal components.

## Files

- `linear_algebra.py` - Interactive examples and visualizations

## Key Takeaways

1. Vectors are lists of numbers (features in ML)
2. Dot product measures similarity
3. Matrix multiplication = many dot products
4. Neural network layers are matrix operations
5. NumPy makes this efficient

## What's Next?

Step 2: **Calculus for ML** — how we optimize.
