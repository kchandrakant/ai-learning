# Step 1: Linear Algebra Essentials

## Why Linear Algebra?

Neural networks are built on matrix operations. Every layer is:
```
output = activation(W × input + b)
```

Without linear algebra, ML is a black box. Understanding these operations lets you:
- Debug dimension mismatches
- Understand what layers actually compute
- Read research papers
- Implement algorithms from scratch

---

## Vectors

A vector is an ordered list of numbers representing a point or direction in space.

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

```
    y
    |     * (3, 4)
    |    /
    |   /  <- This arrow IS the vector
    |  /
    | /
    +---------- x
```

- **Vector = arrow** from origin to point
- **Length (magnitude):** ||v|| = sqrt(v1² + v2² + ...)
- **Direction:** unit vector v/||v||

### In ML Context

Each data point is a vector! A house with 3 bedrooms, 1500 sqft, and $300k price is:
```python
house = np.array([3, 1500, 300000])  # Feature vector
```

A dataset of 1000 houses = 1000 vectors stacked into a matrix.

---

## Dot Product

The dot product combines two vectors into a single number.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Or equivalently
dot = a @ b
```

### What Does It Mean?

**Geometric interpretation:**
```
a . b = ||a|| × ||b|| × cos(theta)

If a . b = 0  -> vectors are perpendicular (90 degrees)
If a . b > 0  -> vectors point same general direction
If a . b < 0  -> vectors point opposite directions
```

**In ML:** Dot product measures **similarity** between vectors.

```
High dot product = similar direction = similar features
Low/negative dot product = different direction = different features
```

This is why dot products appear everywhere in ML:
- Cosine similarity in NLP
- Attention mechanisms in transformers
- Linear layer computations

---

## Matrices

A matrix is a 2D array of numbers - think of it as a stack of vectors.

```python
# 2x3 matrix (2 rows, 3 columns)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Shape
A.shape  # (2, 3)
```

### Matrix as Transformation

A matrix represents a **transformation** of space:
- Rotation
- Scaling
- Shearing
- Projection

When you multiply a vector by a matrix, you're transforming that vector.

```
Original space    Matrix A    Transformed space
     |               ->            /
     |                           /
     +---                      ---
```

---

## Matrix Multiplication

```python
A = np.array([[1, 2], [3, 4]])  # 2x2
B = np.array([[5, 6], [7, 8]])  # 2x2

C = A @ B  # Matrix multiplication
# C[i,j] = sum of (row i of A) * (column j of B)
```

### The Shape Rule

```
(m × n) @ (n × p) = (m × p)
 -----     -----
   |         |
   +---------+  <- These must match!
```

If inner dimensions don't match, multiplication is undefined.

### Visualizing Matrix Multiplication

```
         B columns
         [b1 b2]
         [b3 b4]
            |
            v
A rows  [a1 a2] ->  [a1*b1+a2*b3  a1*b2+a2*b4]
        [a3 a4] ->  [a3*b1+a4*b3  a3*b2+a4*b4]
```

Each element is a dot product of a row from A with a column from B.

### In Neural Networks

```python
# Input: 100 samples, 784 features (e.g., 28x28 images flattened)
X = np.random.randn(100, 784)

# Weight matrix: 784 inputs -> 256 outputs
W = np.random.randn(784, 256)

# Forward pass: each sample gets transformed
output = X @ W  # Shape: (100, 256)

# This is EXACTLY what a Dense/Linear layer does!
```

---

## Special Matrices

```python
# Identity matrix: I × A = A (like multiplying by 1)
I = np.eye(3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Transpose: flip rows and columns
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
A_T = A.T  # 2x3

# Inverse: A × A^(-1) = I (like dividing)
A_inv = np.linalg.inv(A)
# Only square matrices can have inverses
# Not all square matrices have inverses (singular matrices)
```

---

## Eigenvalues and Eigenvectors

For a matrix A, an eigenvector v satisfies:
```
A × v = lambda × v
```

The matrix only **scales** v by lambda, doesn't change its direction.

```
Regular vector:           Eigenvector:
    |                         |
    v  --A-->  /              v  --A-->  |
              /                          | (same direction, just scaled)
```

### Why This Matters for ML

**Principal Component Analysis (PCA):**
- Find eigenvectors of covariance matrix
- These are directions of maximum variance
- Use them to reduce dimensions while keeping important information

```python
# PCA uses eigenvectors to find principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # 100D -> 2D
# pca.components_ are the eigenvectors!
```

---

## Broadcasting in NumPy

NumPy automatically expands dimensions to make operations work:

```python
# Add bias to all samples
X = np.random.randn(100, 784)  # 100 samples
b = np.random.randn(784)        # 1 bias vector

result = X + b  # b is "broadcast" to (100, 784)
# Each row of X gets b added to it
```

**Broadcasting rules:**
1. Align shapes from the right
2. Dimensions must be equal OR one of them must be 1

```
(100, 784) + (784,)   -> Works! (784,) becomes (1, 784), then (100, 784)
(100, 784) + (100, 1) -> Works! Broadcasts to (100, 784)
(100, 784) + (50, 784) -> ERROR! 100 != 50
```

---

## Common Pitfalls

### 1. Dimension Mismatch
```python
# Wrong: inner dimensions don't match
A = np.random.randn(10, 5)
B = np.random.randn(10, 3)
# A @ B -> Error! (10,5) @ (10,3) - 5 != 10
```

### 2. Row vs Column Vectors
```python
# NumPy 1D arrays are neither row nor column
v = np.array([1, 2, 3])  # Shape: (3,)

# Make it explicit if needed
row = v.reshape(1, -1)     # Shape: (1, 3)
col = v.reshape(-1, 1)     # Shape: (3, 1)
```

### 3. In-place vs New Array
```python
A = np.array([1, 2, 3])
B = A          # B points to same memory!
B[0] = 99      # This changes A too!

B = A.copy()   # Use copy() for independent array
```

---

## Files

- `linear_algebra.py` - Interactive examples and visualizations

## Key Takeaways

1. **Vectors** are lists of numbers (features in ML)
2. **Dot product** measures similarity between vectors
3. **Matrix multiplication** = many dot products = transformation
4. **Neural network layers** are just matrix operations
5. **Eigenvectors** find principal directions (used in PCA)
6. **NumPy broadcasting** handles dimension mismatches automatically

## What's Next?

Step 2: **Calculus for ML** - how we optimize neural networks.
