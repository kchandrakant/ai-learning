# Self-Assessment: Are You Ready for Course 01?

This assessment covers the skills from Course 00. Complete all sections to verify you're ready for Course 01: ML Foundations.

**Rules:**
- Try each exercise without looking at solutions first
- Time yourself — if you're spending more than the suggested time, review that module
- It's okay to use documentation, but not to copy solutions

---

## Section 1: Python Essentials
**Time limit: 15 minutes**

### Exercise 1.1: Comprehensions
```python
# Given this data:
records = [
    {'name': 'Alice', 'score': 92, 'subject': 'math'},
    {'name': 'Bob', 'score': 78, 'subject': 'math'},
    {'name': 'Charlie', 'score': 85, 'subject': 'physics'},
    {'name': 'Diana', 'score': 91, 'subject': 'physics'},
    {'name': 'Eve', 'score': 67, 'subject': 'math'},
]

# Write ONE LINE for each:
# 1. List of all names
names = # ???

# 2. List of scores for math students only
math_scores = # ???

# 3. Dictionary mapping name -> score
name_to_score = # ???

# 4. Average score (use sum() and len())
avg_score = # ???
```

### Exercise 1.2: Generator Function
```python
# Write a generator function that yields all pairs (i, j) where i < j
# from a list. Example: pairs([1,2,3]) yields (1,2), (1,3), (2,3)

def pairs(items):
    # Your code here
    pass

# Test it:
# list(pairs([1, 2, 3, 4])) should give:
# [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
```

### Exercise 1.3: Class
```python
# Implement a simple Counter class:
# - __init__ starts count at 0
# - increment() adds 1 to count
# - decrement() subtracts 1 (but not below 0)
# - value property returns current count
# - __repr__ returns "Counter(value=N)"

class Counter:
    # Your code here
    pass

# Test:
# c = Counter()
# c.increment()
# c.increment()
# print(c.value)  # 2
# c.decrement()
# print(c)  # Counter(value=1)
```

---

## Section 2: NumPy Fundamentals
**Time limit: 20 minutes**

### Exercise 2.1: Broadcasting Predictions
```python
# Predict the OUTPUT SHAPE (or "Error") for each operation:

# a) (4, 3) + (3,)         → ???
# b) (4, 3) + (4,)         → ???
# c) (2, 3, 4) + (3, 4)    → ???
# d) (5, 1, 3) + (1, 4, 3) → ???
# e) (3, 4) * (4, 3)       → ???
```

### Exercise 2.2: Vectorized Operations
```python
import numpy as np

# Given:
data = np.random.randn(100, 5)  # 100 samples, 5 features

# WITHOUT USING LOOPS, compute:

# 1. Standardize each column (zero mean, unit variance)
standardized = # ???

# 2. Find the row index with the maximum sum
max_row_idx = # ???

# 3. Create a boolean mask where all values in a row are positive
all_positive_mask = # ???

# 4. Compute the correlation matrix (5x5) between features
# Hint: Standardize first, then (X.T @ X) / (n-1)
correlation = # ???
```

### Exercise 2.3: Matrix Operations
```python
import numpy as np

# Given two sets of vectors:
A = np.random.randn(10, 64)  # 10 vectors of dimension 64
B = np.random.randn(20, 64)  # 20 vectors of dimension 64

# Compute the cosine similarity matrix (10x20)
# where result[i,j] = cosine_similarity(A[i], B[j])
# Do this WITHOUT loops.

cosine_sim = # ???

# What should the shape of cosine_sim be? ???
```

---

## Section 3: Matplotlib
**Time limit: 15 minutes**

### Exercise 3.1: Training Curves
```python
import matplotlib.pyplot as plt
import numpy as np

# Generate fake training data
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.1 + np.random.randn(50) * 0.02
val_loss = 2.7 * np.exp(-0.06 * epochs) + 0.15 + np.random.randn(50) * 0.03

# Create a figure with:
# - Title: "Training Progress"
# - X-axis labeled "Epoch"
# - Y-axis labeled "Loss" 
# - Two lines: train (blue) and validation (orange)
# - Legend in upper right
# - Grid with alpha=0.3

# Your code here
```

### Exercise 3.2: Multi-Panel Figure
```python
# Create a 2x2 figure containing:
# - Top-left: Histogram of 1000 random normal samples
# - Top-right: Scatter plot of 50 random 2D points
# - Bottom-left: Bar chart of values [3, 7, 2, 8] for categories A, B, C, D
# - Bottom-right: Line plot of sin(x) for x from 0 to 2π

# Each subplot should have an appropriate title
# Use figsize=(12, 10) and tight_layout()

# Your code here
```

---

## Section 4: Jupyter Workflow
**Time limit: 5 minutes (conceptual)**

### Exercise 4.1: Magic Commands
Answer these questions:

1. What magic command times a single line of code?
2. What magic command enters the debugger after an error?
3. How do you auto-reload imported modules when they change?
4. How do you run a shell command from a notebook?

### Exercise 4.2: Best Practices
List three things you should do before sharing a notebook to ensure reproducibility.

---

## Section 5: Reading Papers
**Time limit: 15 minutes**

### Exercise 5.1: Paper Structure
Match each section to what it contains:

| Section | Contains |
|---------|----------|
| 1. Abstract | A. Prior approaches and how this differs |
| 2. Introduction | B. Implementation details, proofs |
| 3. Related Work | C. The algorithm/architecture/technique |
| 4. Method | D. Entire paper in one paragraph |
| 5. Experiments | E. Problem motivation and contribution summary |
| 6. Appendix | F. Datasets, baselines, results tables |

### Exercise 5.2: Math Notation
Translate these to plain English:

1. `x ∈ ℝ^d`
2. `argmax_θ p(y|x, θ)`
3. `∇_θ L(θ)`
4. `∑_{i=1}^{N} x_i`
5. `||x - y||_2`

### Exercise 5.3: Paper Summary
Read this abstract and write a 2-3 sentence summary:

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

---

## Grading

### Passing Criteria

| Section | Minimum to Pass |
|---------|----------------|
| Python | 3/3 exercises working |
| NumPy | 2.5/3 exercises correct |
| Matplotlib | Both plots render correctly |
| Jupyter | 5/6 questions correct |
| Papers | 5/8 notation + reasonable summary |

**If you pass all sections:** Proceed to Course 01!

**If you struggle with a section:** Review that module before continuing.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

### Section 1: Python

**1.1 Comprehensions:**
```python
names = [r['name'] for r in records]
math_scores = [r['score'] for r in records if r['subject'] == 'math']
name_to_score = {r['name']: r['score'] for r in records}
avg_score = sum(r['score'] for r in records) / len(records)
```

**1.2 Generator:**
```python
def pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield (items[i], items[j])
```

**1.3 Class:**
```python
class Counter:
    def __init__(self):
        self._count = 0
    
    def increment(self):
        self._count += 1
    
    def decrement(self):
        self._count = max(0, self._count - 1)
    
    @property
    def value(self):
        return self._count
    
    def __repr__(self):
        return f"Counter(value={self._count})"
```

### Section 2: NumPy

**2.1 Broadcasting:**
```
a) (4, 3) + (3,)         → (4, 3)
b) (4, 3) + (4,)         → Error! (3 ≠ 4)
c) (2, 3, 4) + (3, 4)    → (2, 3, 4)
d) (5, 1, 3) + (1, 4, 3) → (5, 4, 3)
e) (3, 4) * (4, 3)       → Error! (4 ≠ 3 and 3 ≠ 4)
```

**2.2 Vectorized Operations:**
```python
# 1. Standardize
mean = data.mean(axis=0)
std = data.std(axis=0)
standardized = (data - mean) / std

# 2. Max row index
max_row_idx = data.sum(axis=1).argmax()

# 3. All positive mask
all_positive_mask = (data > 0).all(axis=1)

# 4. Correlation matrix
X = (data - data.mean(axis=0)) / data.std(axis=0)
correlation = (X.T @ X) / (len(X) - 1)
```

**2.3 Cosine Similarity:**
```python
# Normalize rows to unit length
A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)

# Cosine similarity is dot product of normalized vectors
cosine_sim = A_norm @ B_norm.T  # Shape: (10, 20)
```

### Section 3: Matplotlib

**3.1 Training Curves:**
```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_loss, color='blue', label='Train')
ax.plot(epochs, val_loss, color='orange', label='Validation')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Progress')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.show()
```

**3.2 Multi-Panel:**
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(np.random.randn(1000), bins=30)
axes[0, 0].set_title('Normal Distribution')

# Scatter
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[0, 1].set_title('Random Points')

# Bar
axes[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 8])
axes[1, 0].set_title('Category Values')

# Line
x = np.linspace(0, 2*np.pi, 100)
axes[1, 1].plot(x, np.sin(x))
axes[1, 1].set_title('Sine Wave')

plt.tight_layout()
plt.show()
```

### Section 4: Jupyter

**4.1 Magic Commands:**
1. `%timeit`
2. `%debug`
3. `%load_ext autoreload` then `%autoreload 2`
4. `!command` (e.g., `!ls`)

**4.2 Best Practices:**
1. Restart kernel and run all cells (`Restart & Run All`)
2. Set random seeds at the beginning
3. Clear all outputs and re-run to verify
4. Document dependencies/versions
5. Ensure cells are in logical order

### Section 5: Papers

**5.1 Structure Matching:**
```
1-D, 2-E, 3-A, 4-C, 5-F, 6-B
```

**5.2 Math Notation:**
1. x is a d-dimensional real vector
2. The value of θ that maximizes the probability of y given x
3. The gradient of loss L with respect to parameters θ
4. The sum of x_i for i from 1 to N
5. The L2 (Euclidean) distance between x and y

**5.3 Paper Summary:**
> BERT is a language model that pre-trains bidirectional representations by looking at both left and right context simultaneously, unlike previous models that only looked left-to-right. The pre-trained model can be fine-tuned for various NLP tasks by adding a simple output layer, achieving state-of-the-art results without needing task-specific architectures.

</details>

---

## What's Next?

✅ **Passed everything?** 
→ Proceed to **Course 01: ML Foundations**

❌ **Struggled with Python?** 
→ Review Module 01, practice more comprehensions and classes

❌ **Struggled with NumPy?** 
→ Review Module 02, focus on broadcasting rules

❌ **Struggled with Papers?** 
→ Review Module 05, try reading one more paper with the three-pass method

---

## Congratulations!

If you've made it here, you have the foundational skills to begin your ML journey. The tools you've learned — Python patterns, NumPy thinking, visualization, paper reading — will serve you throughout the entire learning path.

**Next stop: Course 01 - ML Foundations**
