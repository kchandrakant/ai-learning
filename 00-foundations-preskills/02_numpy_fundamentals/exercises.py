"""
Module 2: NumPy Fundamentals - Practice Exercises
==================================================

This module contains hands-on exercises to build fluency with NumPy.
NumPy is the foundation of all ML computation - master this and you can
understand any ML codebase.

Usage:
    python exercises.py           # Run all exercises
    python exercises.py --check   # Check your solutions
"""

import numpy as np
import time

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# PART 1: ARRAY CREATION AND BASICS
# =============================================================================

print("\n" + "="*60)
print("PART 1: ARRAY CREATION AND BASICS")
print("="*60)

# Exercise 1.1: Creating Arrays
print("\n--- Exercise 1.1: Creating Arrays ---")

# From Python list
arr1 = np.array([1, 2, 3, 4, 5])
print(f"From list: {arr1}, shape: {arr1.shape}, dtype: {arr1.dtype}")

# 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{arr2}")
print(f"Shape: {arr2.shape}, ndim: {arr2.ndim}, size: {arr2.size}")

# Common constructors
print(f"\nzeros(3, 4):\n{np.zeros((3, 4))}")
print(f"\nones(2, 3):\n{np.ones((2, 3))}")
print(f"\neye(3):\n{np.eye(3)}")
print(f"\narange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"\nlinspace(0, 1, 5): {np.linspace(0, 1, 5)}")

# Exercise 1.2: Data Types
print("\n--- Exercise 1.2: Data Types ---")

# ML commonly uses float32 for memory efficiency
arr_f32 = np.array([1, 2, 3], dtype=np.float32)
arr_f64 = np.array([1, 2, 3], dtype=np.float64)
print(f"float32: {arr_f32}, dtype: {arr_f32.dtype}")
print(f"float64: {arr_f64}, dtype: {arr_f64.dtype}")
print(f"Memory: float32 uses {arr_f32.nbytes} bytes, float64 uses {arr_f64.nbytes} bytes")


# =============================================================================
# PART 2: INDEXING AND SLICING
# =============================================================================

print("\n" + "="*60)
print("PART 2: INDEXING AND SLICING")
print("="*60)

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
print(f"Array:\n{arr}")

# Exercise 2.1: Basic Indexing
print("\n--- Exercise 2.1: Basic Indexing ---")
print(f"arr[0, 0] = {arr[0, 0]}")          # First element
print(f"arr[-1, -1] = {arr[-1, -1]}")      # Last element
print(f"arr[1] = {arr[1]}")                # Second row
print(f"arr[:, 2] = {arr[:, 2]}")          # Third column
print(f"arr[0:2, 1:3] =\n{arr[0:2, 1:3]}") # Submatrix

# Exercise 2.2: Fancy Indexing
print("\n--- Exercise 2.2: Fancy Indexing ---")

# Index with array of indices
indices = np.array([0, 2])
print(f"Rows 0 and 2:\n{arr[indices]}")

# Boolean indexing (masking)
print(f"\nElements > 5: {arr[arr > 5]}")
print(f"Mask for > 5:\n{arr > 5}")

# Exercise 2.3: Common ML Pattern - Data Filtering
print("\n--- Exercise 2.3: Data Filtering Pattern ---")

# Simulated dataset
data = np.random.randn(100, 5)   # 100 samples, 5 features
labels = np.random.randint(0, 3, 100)  # 3 classes

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Select samples for class 0
class_0_data = data[labels == 0]
print(f"Class 0 samples: {class_0_data.shape[0]}")

# Select samples where feature 0 > 0
positive_feat0 = data[data[:, 0] > 0]
print(f"Samples with feature 0 > 0: {positive_feat0.shape[0]}")


# =============================================================================
# PART 3: BROADCASTING - THE MOST IMPORTANT CONCEPT
# =============================================================================

print("\n" + "="*60)
print("PART 3: BROADCASTING")
print("="*60)

print("""
Broadcasting Rules (memorize these!):
1. Compare shapes from right to left
2. Dimensions are compatible if equal OR one is 1
3. If dimension is 1, it "stretches" to match
""")

# Exercise 3.1: Broadcasting Examples
print("\n--- Exercise 3.1: Broadcasting Examples ---")

# (3, 4) + (4,) -> (3, 4)
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
print(f"A shape: {A.shape}, b shape: {b.shape}")
print(f"A + b (broadcasts b to each row):\n{A + b}")

# (3, 4) + (3, 1) -> (3, 4)
c = np.array([[10], [20], [30]])
print(f"\nA shape: {A.shape}, c shape: {c.shape}")
print(f"A + c (broadcasts c to each column):\n{A + c}")

# (1, 4) + (3, 1) -> (3, 4) - outer product style
row = np.array([[1, 2, 3, 4]])
col = np.array([[10], [20], [30]])
print(f"\nrow shape: {row.shape}, col shape: {col.shape}")
print(f"row + col:\n{row + col}")

# Exercise 3.2: Broadcasting Quiz
print("\n--- Exercise 3.2: Broadcasting Quiz ---")

def predict_broadcast(shape1, shape2):
    """Predict the result shape or 'Error'."""
    try:
        a = np.ones(shape1)
        b = np.ones(shape2)
        result = a + b
        return result.shape
    except ValueError:
        return "Error"

test_cases = [
    ((5, 3), (3,)),      # Should work
    ((5, 3), (5,)),      # Should fail
    ((2, 3, 4), (3, 4)), # Should work
    ((2, 3, 4), (2, 1, 4)), # Should work
    ((4, 1, 3), (1, 5, 3)), # Should work
    ((3, 4), (4, 3)),    # Should fail
]

print("Shape predictions:")
for s1, s2 in test_cases:
    result = predict_broadcast(s1, s2)
    print(f"  {s1} + {s2} -> {result}")

# Exercise 3.3: Common ML Broadcasting Patterns
print("\n--- Exercise 3.3: ML Broadcasting Patterns ---")

# Pattern 1: Normalize features (subtract mean, divide by std)
data = np.random.randn(100, 10)  # 100 samples, 10 features
mean = data.mean(axis=0)  # Shape (10,)
std = data.std(axis=0)    # Shape (10,)
normalized = (data - mean) / std
print(f"Normalized data: mean={normalized.mean(axis=0).round(2)}, std={normalized.std(axis=0).round(2)}")

# Pattern 2: Add bias to linear layer output
weights = np.random.randn(10, 5)  # 10 features -> 5 outputs
bias = np.random.randn(5)         # 5 biases
output = data @ weights + bias    # (100, 5) + (5,) broadcasts correctly
print(f"Linear layer output shape: {output.shape}")

# Pattern 3: Softmax
logits = np.random.randn(100, 5)
# Subtract max for numerical stability (broadcasts correctly)
exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
print(f"Softmax sums to 1: {softmax.sum(axis=1)[:5].round(3)}")


# =============================================================================
# PART 4: VECTORIZATION - SPEED MATTERS
# =============================================================================

print("\n" + "="*60)
print("PART 4: VECTORIZATION")
print("="*60)

# Exercise 4.1: Loop vs Vectorized
print("\n--- Exercise 4.1: Speed Comparison ---")

n = 100000
a = np.random.randn(n)
b = np.random.randn(n)

# Slow: Python loop
start = time.time()
result_loop = np.empty(n)
for i in range(n):
    result_loop[i] = a[i] + b[i]
loop_time = time.time() - start

# Fast: Vectorized
start = time.time()
result_vec = a + b
vec_time = time.time() - start

print(f"Loop time: {loop_time:.4f}s")
print(f"Vectorized time: {vec_time:.6f}s")
print(f"Speedup: {loop_time / vec_time:.0f}x faster!")

# Exercise 4.2: Vectorized Operations
print("\n--- Exercise 4.2: Common Vectorized Operations ---")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Element-wise operations
print(f"x = {x}")
print(f"x + 1 = {x + 1}")
print(f"x * 2 = {x * 2}")
print(f"x ** 2 = {x ** 2}")
print(f"np.sqrt(x) = {np.sqrt(x).round(3)}")
print(f"np.exp(x) = {np.exp(x).round(3)}")
print(f"np.log(x) = {np.log(x).round(3)}")

# Aggregations
print(f"\nnp.sum(x) = {np.sum(x)}")
print(f"np.mean(x) = {np.mean(x)}")
print(f"np.std(x) = {np.std(x):.3f}")
print(f"np.max(x) = {np.max(x)}")
print(f"np.argmax(x) = {np.argmax(x)}")  # Index of max

# Exercise 4.3: Vectorized Conditionals
print("\n--- Exercise 4.3: Vectorized Conditionals ---")

x = np.array([-2, -1, 0, 1, 2])

# ReLU activation (max(0, x))
relu = np.maximum(x, 0)
print(f"x = {x}")
print(f"ReLU(x) = {relu}")

# np.where: vectorized if-else
result = np.where(x > 0, x, 0)  # Same as ReLU
print(f"np.where(x > 0, x, 0) = {result}")

# Clip values to range
clipped = np.clip(x, -1, 1)
print(f"np.clip(x, -1, 1) = {clipped}")


# =============================================================================
# PART 5: RESHAPING AND COMBINING
# =============================================================================

print("\n" + "="*60)
print("PART 5: RESHAPING AND COMBINING")
print("="*60)

# Exercise 5.1: Reshaping
print("\n--- Exercise 5.1: Reshaping ---")

arr = np.arange(12)
print(f"Original: {arr}, shape: {arr.shape}")

# Reshape to 2D
reshaped = arr.reshape(3, 4)
print(f"\nReshaped to (3, 4):\n{reshaped}")

# Use -1 to infer dimension
print(f"\nreshape(4, -1):\n{arr.reshape(4, -1)}")
print(f"reshape(-1, 2):\n{arr.reshape(-1, 2)}")

# Flatten back
print(f"\nflatten(): {reshaped.flatten()}")

# Exercise 5.2: Adding Dimensions
print("\n--- Exercise 5.2: Adding Dimensions ---")

vec = np.array([1, 2, 3])
print(f"vec shape: {vec.shape}")

# Add dimension to make row vector
row = vec[np.newaxis, :]
print(f"row = vec[np.newaxis, :], shape: {row.shape}")
print(f"row:\n{row}")

# Add dimension to make column vector
col = vec[:, np.newaxis]
print(f"col = vec[:, np.newaxis], shape: {col.shape}")
print(f"col:\n{col}")

# Exercise 5.3: Combining Arrays
print("\n--- Exercise 5.3: Combining Arrays ---")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"a:\n{a}")
print(f"b:\n{b}")

# Stack vertically (add rows)
vstacked = np.vstack([a, b])
print(f"\nvstack:\n{vstacked}")

# Stack horizontally (add columns)
hstacked = np.hstack([a, b])
print(f"\nhstack:\n{hstacked}")

# Stack along new axis
stacked = np.stack([a, b], axis=0)
print(f"\nstack axis=0, shape: {stacked.shape}")
print(stacked)


# =============================================================================
# PART 6: LINEAR ALGEBRA
# =============================================================================

print("\n" + "="*60)
print("PART 6: LINEAR ALGEBRA")
print("="*60)

# Exercise 6.1: Matrix Multiplication
print("\n--- Exercise 6.1: Matrix Multiplication ---")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A:\n{A}")
print(f"B:\n{B}")

# Matrix multiplication (NOT element-wise!)
print(f"\nA @ B (matrix multiply):\n{A @ B}")
print(f"A * B (element-wise):\n{A * B}")

# Exercise 6.2: Dot Product
print("\n--- Exercise 6.2: Dot Product ---")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"a = {a}, b = {b}")
print(f"np.dot(a, b) = {np.dot(a, b)}")
print(f"a @ b = {a @ b}")  # Same thing

# Exercise 6.3: Cosine Similarity (Common in ML)
print("\n--- Exercise 6.3: Cosine Similarity ---")

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = np.array([1, 0, 0])
vec2 = np.array([1, 1, 0])
vec3 = np.array([0, 1, 0])

print(f"vec1 = {vec1}, vec2 = {vec2}, vec3 = {vec3}")
print(f"cosine(vec1, vec2) = {cosine_similarity(vec1, vec2):.3f}")
print(f"cosine(vec1, vec3) = {cosine_similarity(vec1, vec3):.3f}")
print(f"cosine(vec2, vec3) = {cosine_similarity(vec2, vec3):.3f}")

# Exercise 6.4: Batch Cosine Similarity (Vectorized)
print("\n--- Exercise 6.4: Batch Cosine Similarity ---")

def batch_cosine_similarity(A, B):
    """
    Compute cosine similarity between all pairs of vectors.
    
    Args:
        A: (n, d) matrix - n vectors of dimension d
        B: (m, d) matrix - m vectors of dimension d
    
    Returns:
        (n, m) matrix of cosine similarities
    """
    # Normalize rows to unit length
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    # Cosine similarity is just dot product of normalized vectors
    return A_norm @ B_norm.T

# Test with embedding-like data
embeddings = np.random.randn(5, 64)  # 5 embeddings of dimension 64
queries = np.random.randn(3, 64)     # 3 queries

similarities = batch_cosine_similarity(queries, embeddings)
print(f"Queries shape: {queries.shape}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Similarity matrix shape: {similarities.shape}")
print(f"Most similar embedding for each query: {similarities.argmax(axis=1)}")


# =============================================================================
# PART 7: RANDOM NUMBERS
# =============================================================================

print("\n" + "="*60)
print("PART 7: RANDOM NUMBERS")
print("="*60)

# Exercise 7.1: Basic Random Generation
print("\n--- Exercise 7.1: Basic Random ---")

# Set seed for reproducibility
np.random.seed(42)

print(f"Uniform [0, 1): {np.random.rand(5).round(3)}")
print(f"Standard normal: {np.random.randn(5).round(3)}")
print(f"Integers [0, 10): {np.random.randint(0, 10, 5)}")

# Exercise 7.2: Modern RNG (Recommended)
print("\n--- Exercise 7.2: Modern RNG API ---")

rng = np.random.default_rng(seed=42)
print(f"rng.random(5): {rng.random(5).round(3)}")
print(f"rng.standard_normal(5): {rng.standard_normal(5).round(3)}")
print(f"rng.integers(0, 10, 5): {rng.integers(0, 10, 5)}")

# Exercise 7.3: ML-Specific Random Operations
print("\n--- Exercise 7.3: ML Random Patterns ---")

# Shuffle indices for train/test split
rng = np.random.default_rng(42)
n_samples = 100
indices = rng.permutation(n_samples)
train_idx = indices[:80]
test_idx = indices[80:]
print(f"Train indices (first 10): {train_idx[:10]}")
print(f"Test indices: {test_idx}")

# Xavier/Glorot initialization
fan_in, fan_out = 784, 256
limit = np.sqrt(6 / (fan_in + fan_out))
xavier_weights = rng.uniform(-limit, limit, (fan_in, fan_out))
print(f"\nXavier weights: shape={xavier_weights.shape}, std={xavier_weights.std():.4f}")

# He initialization (for ReLU)
he_std = np.sqrt(2 / fan_in)
he_weights = rng.normal(0, he_std, (fan_in, fan_out))
print(f"He weights: shape={he_weights.shape}, std={he_weights.std():.4f}")


# =============================================================================
# PART 8: COMPREHENSIVE EXERCISES
# =============================================================================

print("\n" + "="*60)
print("PART 8: COMPREHENSIVE EXERCISES")
print("="*60)

# Exercise 8.1: Implement Softmax
print("\n--- Exercise 8.1: Softmax Implementation ---")

def softmax(x, axis=-1):
    """
    Compute softmax along specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax probabilities (same shape as input)
    """
    # Subtract max for numerical stability
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

logits = np.array([[1, 2, 3], [1, 1, 1], [0, 0, 100]])
probs = softmax(logits)
print(f"Logits:\n{logits}")
print(f"Softmax:\n{probs.round(4)}")
print(f"Row sums: {probs.sum(axis=1)}")

# Exercise 8.2: Implement Cross-Entropy Loss
print("\n--- Exercise 8.2: Cross-Entropy Loss ---")

def cross_entropy_loss(predictions, targets):
    """
    Compute cross-entropy loss.
    
    Args:
        predictions: (N, C) softmax probabilities
        targets: (N,) integer class labels
    
    Returns:
        Scalar loss value
    """
    n_samples = predictions.shape[0]
    # Get probability of correct class for each sample
    correct_probs = predictions[np.arange(n_samples), targets]
    # Compute negative log likelihood
    return -np.log(correct_probs + 1e-8).mean()

predictions = softmax(np.array([[2.0, 1.0, 0.1],
                                [0.5, 2.5, 0.3],
                                [0.1, 0.1, 3.0]]))
targets = np.array([0, 1, 2])  # Correct classes
loss = cross_entropy_loss(predictions, targets)
print(f"Predictions:\n{predictions.round(3)}")
print(f"Targets: {targets}")
print(f"Cross-entropy loss: {loss:.4f}")

# Exercise 8.3: Pairwise Euclidean Distance
print("\n--- Exercise 8.3: Pairwise Euclidean Distance ---")

def pairwise_euclidean_distance(X):
    """
    Compute pairwise Euclidean distances between points.
    
    Args:
        X: (N, D) array of N points in D dimensions
    
    Returns:
        (N, N) distance matrix
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = (X ** 2).sum(axis=1)
    # sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] broadcasts to (N, N)
    sq_distances = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
    # Numerical stability: ensure non-negative before sqrt
    return np.sqrt(np.maximum(sq_distances, 0))

points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
distances = pairwise_euclidean_distance(points)
print(f"Points:\n{points}")
print(f"Pairwise distances:\n{distances.round(3)}")

# Exercise 8.4: Mini-Batch Gradient Descent Step
print("\n--- Exercise 8.4: Linear Regression Step ---")

def linear_regression_step(X, y, weights, bias, learning_rate):
    """
    Perform one gradient descent step for linear regression.
    
    Args:
        X: (N, D) input features
        y: (N,) targets
        weights: (D,) weight vector
        bias: scalar bias
        learning_rate: step size
    
    Returns:
        Updated weights and bias
    """
    n_samples = X.shape[0]
    
    # Forward pass
    predictions = X @ weights + bias
    
    # Compute gradients (MSE loss)
    error = predictions - y
    grad_weights = (2 / n_samples) * X.T @ error
    grad_bias = (2 / n_samples) * error.sum()
    
    # Update parameters
    weights = weights - learning_rate * grad_weights
    bias = bias - learning_rate * grad_bias
    
    # Compute loss
    loss = (error ** 2).mean()
    
    return weights, bias, loss

# Test linear regression
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1, -2, 3, -1, 0.5])
true_bias = 2.0
y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.1

# Initialize and train
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.1

print("Training linear regression:")
for epoch in range(100):
    weights, bias, loss = linear_regression_step(X, y, weights, bias, learning_rate)
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: loss = {loss:.4f}")

print(f"\nLearned weights: {weights.round(3)}")
print(f"True weights:    {true_weights}")
print(f"Learned bias: {bias:.3f}, True bias: {true_bias}")


# =============================================================================
# PRACTICE CHALLENGES
# =============================================================================

print("\n" + "="*60)
print("PRACTICE CHALLENGES")
print("="*60)

print("""
Try implementing these on your own:

1. BROADCASTING CHALLENGE:
   Given a (100, 768) embedding matrix and a (768,) query vector,
   compute the dot product of the query with each embedding
   (result should be shape (100,)).

2. NORMALIZATION CHALLENGE:
   Implement batch normalization: for each feature, subtract mean
   and divide by std, computed across the batch.
   
   Input: (N, D) -> Output: (N, D) with each column normalized.

3. ATTENTION CHALLENGE:
   Implement scaled dot-product attention:
   Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
   
   Q, K, V are all (seq_len, d_model) matrices.

4. KNN CHALLENGE:
   Implement k-nearest neighbors prediction:
   Given training data (N, D) with labels (N,) and a query point (D,),
   find the k nearest neighbors and return the majority label.

5. PCA CHALLENGE:
   Implement PCA dimensionality reduction:
   - Center the data (subtract mean)
   - Compute covariance matrix
   - Get top k eigenvectors
   - Project data onto eigenvectors

Run this file with --check to see solutions!
""")


# =============================================================================
# SOLUTIONS
# =============================================================================

def show_solutions():
    """Display solutions to practice challenges."""
    
    print("\n" + "="*60)
    print("SOLUTIONS")
    print("="*60)
    
    np.random.seed(42)
    
    # Challenge 1: Dot Products
    print("\n--- Challenge 1: Dot Products ---")
    embeddings = np.random.randn(100, 768)
    query = np.random.randn(768)
    
    # Solution: just use @ operator
    scores = embeddings @ query
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Query shape: {query.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Top 3 indices: {np.argsort(scores)[-3:][::-1]}")
    
    # Challenge 2: Batch Normalization
    print("\n--- Challenge 2: Batch Normalization ---")
    def batch_normalize(x, eps=1e-5):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return (x - mean) / (std + eps)
    
    data = np.random.randn(32, 128) * 5 + 10  # Not normalized
    normalized = batch_normalize(data)
    print(f"Before: mean={data.mean(axis=0).mean():.2f}, std={data.std(axis=0).mean():.2f}")
    print(f"After: mean={normalized.mean(axis=0).mean():.6f}, std={normalized.std(axis=0).mean():.4f}")
    
    # Challenge 3: Attention
    print("\n--- Challenge 3: Attention ---")
    def scaled_dot_product_attention(Q, K, V):
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores, axis=-1)
        return weights @ V
    
    seq_len, d_model = 10, 64
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    
    output = scaled_dot_product_attention(Q, K, V)
    print(f"Q, K, V shapes: ({seq_len}, {d_model})")
    print(f"Attention output shape: {output.shape}")
    
    # Challenge 4: KNN
    print("\n--- Challenge 4: KNN ---")
    def knn_predict(X_train, y_train, query, k=3):
        # Compute distances from query to all training points
        distances = np.sqrt(((X_train - query) ** 2).sum(axis=1))
        # Get k nearest neighbor indices
        nearest_indices = np.argsort(distances)[:k]
        # Get their labels
        nearest_labels = y_train[nearest_indices]
        # Return majority vote
        unique, counts = np.unique(nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]
    
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 3, 100)
    query = np.random.randn(5)
    
    prediction = knn_predict(X_train, y_train, query, k=5)
    print(f"Training data shape: {X_train.shape}")
    print(f"Query shape: {query.shape}")
    print(f"Predicted class: {prediction}")
    
    # Challenge 5: PCA
    print("\n--- Challenge 5: PCA ---")
    def pca(X, n_components):
        # Center the data
        X_centered = X - X.mean(axis=0)
        # Compute covariance matrix
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        # Take top n_components
        components = eigenvectors[:, :n_components]
        # Project data
        return X_centered @ components
    
    X = np.random.randn(100, 10)
    X_reduced = pca(X, n_components=3)
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Variance explained: {X_reduced.var(axis=0).sum() / X.var():.2%}")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        show_solutions()
    
    print("\n" + "="*60)
    print("Module 2 Complete!")
    print("="*60)
    print("""
Key takeaways:
1. Broadcasting: shapes align right-to-left, 1 broadcasts
2. Vectorize: avoid Python loops, use NumPy operations
3. Axis: axis=0 is rows, axis=1 is columns
4. keepdims=True: preserve shape for broadcasting

Next steps:
1. Try the practice challenges on your own
2. Run with --check to see solutions
3. Move on to Module 3: Matplotlib Visualization
""")
