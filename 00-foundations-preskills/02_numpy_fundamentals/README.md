# Module 2: NumPy Fundamentals

NumPy is the foundation of numerical computing in Python. Every ML framework — PyTorch, TensorFlow, JAX — builds on concepts from NumPy. If you can think in NumPy arrays, you can think in tensors.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Create and manipulate NumPy arrays
- Apply broadcasting rules correctly
- Write vectorized code (and understand why it's faster)
- Perform common linear algebra operations
- Reshape and combine arrays
- Generate random numbers for ML experiments

---

## 📚 Part 1: Arrays — The Building Block

### Creating Arrays

```python
import numpy as np

# From Python lists
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array (matrix)

# Common constructors
zeros = np.zeros((3, 4))        # 3x4 matrix of zeros
ones = np.ones((2, 3))          # 2x3 matrix of ones
empty = np.empty((2, 2))        # Uninitialized (fast, random values)
full = np.full((3, 3), 7)       # 3x3 filled with 7

# Ranges
seq = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8] — like range()
lin = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1] — n evenly spaced

# Identity matrix
eye = np.eye(4)                 # 4x4 identity matrix
```

### Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3) — dimensions
arr.ndim       # 2 — number of dimensions
arr.size       # 6 — total number of elements
arr.dtype      # dtype('int64') — data type
```

### Data Types

```python
# Specify dtype explicitly
a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([1.5, 2.5], dtype=np.int32)  # Truncates to [1, 2]

# Common dtypes in ML:
# np.float32  — standard for neural networks (saves memory)
# np.float64  — higher precision (NumPy default)
# np.int64    — indices, labels
# np.bool_    — masks
```

---

## 📚 Part 2: Indexing and Slicing

### Basic Indexing

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Single element
arr[0, 0]      # 1
arr[2, 3]      # 12
arr[-1, -1]    # 12 (negative indexing)

# Row or column
arr[0]         # [1, 2, 3, 4] — first row
arr[:, 0]      # [1, 5, 9] — first column
arr[:, -1]     # [4, 8, 12] — last column

# Slicing: [start:stop:step]
arr[0:2]       # First two rows
arr[:, 1:3]    # Columns 1 and 2
arr[::2]       # Every other row
```

### Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Index with array of indices
indices = np.array([0, 2, 4])
arr[indices]   # [10, 30, 50]

# Boolean indexing (masking)
mask = arr > 25
arr[mask]      # [30, 40, 50]

# Shorthand
arr[arr > 25]  # [30, 40, 50]
```

**Common ML pattern — filtering data:**
```python
# Select samples where label == 1
data = np.random.randn(100, 10)  # 100 samples, 10 features
labels = np.random.randint(0, 2, 100)

class_1_data = data[labels == 1]
print(class_1_data.shape)  # (n, 10) where n is count of label==1
```

---

## 📚 Part 3: Broadcasting

Broadcasting is how NumPy handles operations between arrays of different shapes. **This is the most important concept to master.**

### The Rules

When operating on two arrays, NumPy compares shapes element-wise from right to left:
1. Dimensions are compatible if they're equal OR one of them is 1
2. If a dimension is 1, it's "stretched" to match the other

```python
# Example: Adding (3, 4) + (4,)
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])

# Shape comparison (right to left):
# A: 3 x 4
# b:     4
# Result: 3 x 4 (b broadcasts across each row)

result = A + b
# [[2, 3, 4, 5],
#  [2, 3, 4, 5],
#  [2, 3, 4, 5]]
```

### Broadcasting Examples

```python
# (3, 4) + (3, 1) → (3, 4)
A = np.ones((3, 4))
b = np.array([[1], [2], [3]])  # Column vector
result = A + b
# [[2, 2, 2, 2],
#  [3, 3, 3, 3],
#  [4, 4, 4, 4]]

# (1, 4) + (3, 1) → (3, 4)
row = np.array([[1, 2, 3, 4]])  # Shape (1, 4)
col = np.array([[10], [20], [30]])  # Shape (3, 1)
result = row + col
# [[11, 12, 13, 14],
#  [21, 22, 23, 24],
#  [31, 32, 33, 34]]

# Scalar broadcasts to any shape
A = np.ones((3, 4))
result = A * 5  # All elements multiplied by 5
```

### Broadcasting Failures

```python
# (3, 4) + (3,) → ERROR!
A = np.ones((3, 4))
b = np.array([1, 2, 3])

# Shape comparison:
# A: 3 x 4
# b:     3
# 4 ≠ 3 and neither is 1 → FAIL

A + b  # ValueError: operands could not be broadcast together
```

### Common ML Broadcasting Patterns

```python
# Normalize each feature (subtract mean, divide by std)
data = np.random.randn(100, 10)  # 100 samples, 10 features

mean = data.mean(axis=0)  # Shape (10,) — mean of each feature
std = data.std(axis=0)    # Shape (10,)

normalized = (data - mean) / std  # Broadcasting: (100, 10) - (10,) / (10,)

# Add bias to each sample
weights = np.random.randn(10, 5)  # 10 input features, 5 outputs
bias = np.random.randn(5)         # 5 biases

output = data @ weights + bias    # (100, 5) + (5,) → (100, 5)
```

---

## 📚 Part 4: Vectorization

Vectorization means replacing Python loops with NumPy operations. It's typically 10-100x faster.

### Why Loops Are Slow

```python
import time

# Slow: Python loop
def slow_add(a, b):
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result

# Fast: Vectorized
def fast_add(a, b):
    return a + b

# Benchmark
a = np.random.randn(1000000)
b = np.random.randn(1000000)

start = time.time()
slow_add(a, b)
print(f"Loop: {time.time() - start:.3f}s")

start = time.time()
fast_add(a, b)
print(f"Vectorized: {time.time() - start:.3f}s")

# Typical output:
# Loop: 0.450s
# Vectorized: 0.003s (150x faster!)
```

### Common Vectorization Patterns

```python
# Instead of loops, use:

# Element-wise operations
c = a + b
c = a * b
c = np.exp(a)
c = np.log(a)
c = np.maximum(a, 0)  # ReLU activation

# Aggregations
total = np.sum(a)
mean = np.mean(a)
max_val = np.max(a)
argmax = np.argmax(a)  # Index of maximum

# Along axes
row_sums = data.sum(axis=1)      # Sum each row
col_means = data.mean(axis=0)    # Mean of each column
```

### Vectorized Conditionals

```python
# Instead of:
# for i in range(len(a)):
#     if a[i] > 0:
#         result[i] = a[i]
#     else:
#         result[i] = 0

# Use np.where:
result = np.where(a > 0, a, 0)  # ReLU

# Or boolean indexing:
result = a.copy()
result[a < 0] = 0
```

---

## 📚 Part 5: Reshaping and Combining

### Reshaping

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 2D
matrix = arr.reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Use -1 to infer dimension
arr.reshape(3, -1)   # (3, 4) — infers 4
arr.reshape(-1, 2)   # (6, 2) — infers 6

# Flatten back to 1D
matrix.flatten()     # Returns copy
matrix.ravel()       # Returns view (faster, but modifications affect original)

# Add dimension
a = np.array([1, 2, 3])  # Shape (3,)
a[np.newaxis, :]         # Shape (1, 3) — row vector
a[:, np.newaxis]         # Shape (3, 1) — column vector

# Or use reshape
a.reshape(1, -1)         # Shape (1, 3)
a.reshape(-1, 1)         # Shape (3, 1)
```

### Combining Arrays

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Stack vertically (add rows)
np.vstack([a, b])
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# Stack horizontally (add columns)
np.hstack([a, b])
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]

# Concatenate along axis
np.concatenate([a, b], axis=0)  # Same as vstack
np.concatenate([a, b], axis=1)  # Same as hstack

# Stack to create new dimension
np.stack([a, b], axis=0)  # Shape (2, 2, 2)
np.stack([a, b], axis=2)  # Shape (2, 2, 2) but different arrangement
```

### Transpose

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr.T  # Transpose
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# For higher dimensions, use transpose with axis order
arr_3d = np.random.randn(2, 3, 4)
arr_3d.transpose(1, 0, 2).shape  # (3, 2, 4)
```

---

## 📚 Part 6: Linear Algebra

### Matrix Multiplication

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiplication (3 equivalent ways)
C = A @ B           # Preferred (Python 3.5+)
C = np.dot(A, B)    # Traditional
C = np.matmul(A, B) # Explicit

# Note: * is element-wise, NOT matrix multiplication!
# A * B → ERROR if shapes don't match for broadcasting
```

### Dot Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product of vectors
np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Or
a @ b  # 32
```

### Common Operations

```python
A = np.random.randn(3, 3)

# Inverse
A_inv = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (used in PCA)
U, S, Vt = np.linalg.svd(A)

# Norm
np.linalg.norm(a)         # L2 norm (Euclidean)
np.linalg.norm(a, ord=1)  # L1 norm
```

### Cosine Similarity (Common in ML)

```python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# For batches of vectors
def batch_cosine_similarity(A, B):
    """Compute cosine similarity between all pairs.
    
    A: (n, d) — n vectors of dimension d
    B: (m, d) — m vectors of dimension d
    Returns: (n, m) — similarity matrix
    """
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T
```

---

## 📚 Part 7: Random Numbers

### Basic Random Generation

```python
# Set seed for reproducibility
np.random.seed(42)

# Uniform [0, 1)
np.random.rand(3, 4)      # Shape (3, 4)
np.random.random((3, 4))  # Same thing

# Standard normal (mean=0, std=1)
np.random.randn(3, 4)

# Normal with custom mean/std
np.random.normal(loc=0, scale=1, size=(3, 4))

# Integers
np.random.randint(0, 10, size=(3, 4))  # [0, 10)

# Choice from array
arr = np.array([10, 20, 30, 40, 50])
np.random.choice(arr, size=3, replace=False)  # Sample without replacement
```

### Modern Random API (Recommended)

```python
# Create generator (preferred for reproducibility)
rng = np.random.default_rng(seed=42)

rng.random((3, 4))          # Uniform [0, 1)
rng.standard_normal((3, 4)) # Standard normal
rng.integers(0, 10, (3, 4)) # Integers
rng.choice(arr, size=3)     # Choice
rng.shuffle(arr)            # In-place shuffle
rng.permutation(arr)        # Return shuffled copy
```

### ML-Specific Random Operations

```python
rng = np.random.default_rng(42)

# Train/test split indices
n_samples = 1000
indices = rng.permutation(n_samples)
train_idx = indices[:800]
test_idx = indices[800:]

# Xavier/Glorot initialization (for neural networks)
fan_in, fan_out = 784, 256
limit = np.sqrt(6 / (fan_in + fan_out))
weights = rng.uniform(-limit, limit, (fan_in, fan_out))

# He initialization (for ReLU networks)
std = np.sqrt(2 / fan_in)
weights = rng.normal(0, std, (fan_in, fan_out))
```

---

## 🏋️ Exercises

### Exercise 1: Broadcasting Practice
```python
# Predict the output shape (or "Error") for each:
# 1. (5, 3) + (3,)
# 2. (5, 3) + (5,)
# 3. (2, 3, 4) + (3, 4)
# 4. (2, 3, 4) + (2, 1, 4)
# 5. (4, 1, 3) + (1, 5, 3)
```

### Exercise 2: Vectorized Operations
```python
# Implement these WITHOUT using any Python loops:

# 1. Given array of temperatures in Celsius, convert to Fahrenheit
celsius = np.array([0, 20, 37, 100])
# fahrenheit = ?

# 2. Normalize a matrix so each ROW sums to 1
matrix = np.random.rand(5, 4)
# row_normalized = ?

# 3. Compute softmax for each row: exp(x) / sum(exp(x))
logits = np.random.randn(5, 4)
# softmax = ?
```

### Exercise 3: Linear Algebra
```python
# 1. Compute the cosine similarity between these vectors:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 2. Given data matrix X (100 samples, 10 features), compute 
#    the covariance matrix (10x10)
X = np.random.randn(100, 10)
# Hint: cov = (X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / (n - 1)

# 3. Compute all pairwise Euclidean distances between points
#    Result should be (n, n) matrix where [i,j] = distance from point i to j
points = np.random.randn(50, 3)  # 50 points in 3D
# distances = ?
```

### Exercise 4: Data Manipulation
```python
# Given a dataset:
data = np.random.randn(1000, 5)  # 1000 samples, 5 features
labels = np.random.randint(0, 3, 1000)  # 3 classes

# 1. Compute the mean of each feature FOR EACH CLASS
#    Result: (3, 5) array — class_means[i, j] = mean of feature j for class i

# 2. Standardize the data (zero mean, unit variance per feature)
#    Then verify mean ≈ 0 and std ≈ 1

# 3. Split into train (80%) and test (20%) RANDOMLY
#    Use np.random for shuffling
```

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 1: Broadcasting Practice
```python
# 1. (5, 3) + (3,) → (5, 3) ✓
# 2. (5, 3) + (5,) → Error! (3 ≠ 5)
# 3. (2, 3, 4) + (3, 4) → (2, 3, 4) ✓
# 4. (2, 3, 4) + (2, 1, 4) → (2, 3, 4) ✓
# 5. (4, 1, 3) + (1, 5, 3) → (4, 5, 3) ✓
```

### Exercise 2: Vectorized Operations
```python
# 1. Celsius to Fahrenheit
fahrenheit = celsius * 9/5 + 32

# 2. Row normalization
row_normalized = matrix / matrix.sum(axis=1, keepdims=True)

# 3. Softmax
exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))  # Subtract max for numerical stability
softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
```

### Exercise 3: Linear Algebra
```python
# 1. Cosine similarity
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 2. Covariance matrix
X_centered = X - X.mean(axis=0)
cov_matrix = X_centered.T @ X_centered / (X.shape[0] - 1)
# Or simply: np.cov(X.T)

# 3. Pairwise distances
# Method: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
sq_norms = (points ** 2).sum(axis=1)
distances = np.sqrt(sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * points @ points.T)
```

### Exercise 4: Data Manipulation
```python
# 1. Class means
class_means = np.array([data[labels == c].mean(axis=0) for c in range(3)])

# 2. Standardize
mean = data.mean(axis=0)
std = data.std(axis=0)
standardized = (data - mean) / std
print(standardized.mean(axis=0))  # Close to 0
print(standardized.std(axis=0))   # Close to 1

# 3. Train/test split
rng = np.random.default_rng(42)
indices = rng.permutation(len(data))
split = int(0.8 * len(data))
train_data, test_data = data[indices[:split]], data[indices[split:]]
train_labels, test_labels = labels[indices[:split]], labels[indices[split:]]
```

</details>

---

## 🔗 What's Next?

You can now think in arrays. Move on to **Module 3: Matplotlib Visualization** to learn how to see your data and training progress.

---

## 📖 References

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) by Nicolas Rougier (free book)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
