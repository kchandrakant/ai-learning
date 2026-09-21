# Module 1: Python Essentials for ML

This module covers the Python patterns you'll encounter repeatedly in machine learning codebases. It's not a Python tutorial from scratch — it focuses on the specific patterns that ML code uses heavily.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Write and understand list comprehensions and generator expressions
- Use `*args`, `**kwargs`, and default arguments effectively
- Build classes following the patterns used in PyTorch and other ML frameworks
- Use context managers and understand the `with` statement
- Handle errors gracefully and debug Python code
- Read and write type hints

---

## 📚 Part 1: Functions — The ML Patterns

### Default Arguments and Keyword Arguments

ML functions often have many optional parameters:

```python
def train_model(
    model,
    data,
    epochs=10,           # Default value
    learning_rate=0.001,
    batch_size=32,
    verbose=True,
    checkpoint_path=None
):
    """Train a model with configurable hyperparameters."""
    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
        # ... training logic
```

**Calling with keyword arguments:**
```python
# Only override what you need
train_model(my_model, my_data, epochs=50, learning_rate=0.0001)
```

### *args and **kwargs

Used extensively in wrapper functions and class inheritance:

```python
def log_function_call(func):
    """Decorator that logs function calls."""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"Returned: {result}")
        return result
    return wrapper

@log_function_call
def add(a, b):
    return a + b

add(2, 3)       # Calling add with args=(2, 3), kwargs={}
add(a=2, b=3)   # Calling add with args=(), kwargs={'a': 2, 'b': 3}
```

**In class inheritance (PyTorch pattern):**
```python
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass through to parent
        self.layer = nn.Linear(input_dim, hidden_dim)
```

---

## 📚 Part 2: Comprehensions and Generators

### List Comprehensions

Compact way to transform and filter data:

```python
# Basic transformation
texts = ["Hello", "World", "ML"]
lengths = [len(t) for t in texts]  # [5, 5, 2]

# With condition (filtering)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [n for n in numbers if n % 2 == 0]  # [2, 4, 6, 8, 10]

# Transformation + filtering
long_texts = [t.lower() for t in texts if len(t) > 2]  # ['hello', 'world']

# Nested comprehension (flattening)
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]  # [1, 2, 3, 4, 5, 6]
```

**Common ML use case — processing data:**
```python
# Tokenize all documents
tokenized = [tokenizer.encode(doc) for doc in documents]

# Filter valid samples
valid_samples = [s for s in samples if s['label'] is not None]

# Extract specific field
labels = [sample['label'] for sample in dataset]
```

### Dictionary Comprehensions

```python
# Create lookup dictionary
words = ["apple", "banana", "cherry"]
word_to_idx = {word: idx for idx, word in enumerate(words)}
# {'apple': 0, 'banana': 1, 'cherry': 2}

# Reverse a dictionary
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Filter dictionary
scores = {'alice': 85, 'bob': 72, 'charlie': 91}
passed = {name: score for name, score in scores.items() if score >= 80}
# {'alice': 85, 'charlie': 91}
```

### Generator Expressions (Memory Efficient)

For large datasets, generators don't load everything into memory:

```python
# List comprehension — creates full list in memory
squares_list = [x**2 for x in range(1000000)]  # Uses ~8MB

# Generator expression — computes on demand
squares_gen = (x**2 for x in range(1000000))   # Uses ~100 bytes

# Iterate over generator
for square in squares_gen:
    # Computed one at a time
    pass
```

**Generator functions with `yield`:**
```python
def batch_generator(data, batch_size):
    """Yield batches from data without loading all into memory."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Usage
for batch in batch_generator(large_dataset, batch_size=32):
    process(batch)
```

---

## 📚 Part 3: Classes — The PyTorch Pattern

### Basic Class Structure

```python
class DataProcessor:
    """Process data for ML training."""
    
    def __init__(self, vocab_size, max_length=512):
        """Initialize the processor.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self._cache = {}  # Private attribute (convention: underscore prefix)
    
    def process(self, text):
        """Process a single text."""
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        result = self._tokenize(text)
        self._cache[text] = result
        return result
    
    def _tokenize(self, text):
        """Internal tokenization method."""
        # Implementation
        return text.split()[:self.max_length]
    
    def __repr__(self):
        """String representation for debugging."""
        return f"DataProcessor(vocab_size={self.vocab_size}, max_length={self.max_length})"
    
    def __len__(self):
        """Support len() function."""
        return len(self._cache)
```

### Inheritance — The nn.Module Pattern

This is the pattern used in PyTorch for all neural network models:

```python
import torch.nn as nn

class MyModel(nn.Module):
    """Custom neural network model."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # MUST call parent __init__
        
        # Define layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass — called when you do model(x)."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Usage
model = MyModel(input_dim=784, hidden_dim=256, output_dim=10)
output = model(input_tensor)  # Calls forward() automatically
```

### Properties and Setters

```python
class Config:
    def __init__(self):
        self._learning_rate = 0.001
    
    @property
    def learning_rate(self):
        """Get learning rate."""
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        """Set learning rate with validation."""
        if value <= 0:
            raise ValueError("Learning rate must be positive")
        self._learning_rate = value

config = Config()
print(config.learning_rate)  # 0.001
config.learning_rate = 0.01  # Uses setter
config.learning_rate = -1    # Raises ValueError
```

---

## 📚 Part 4: Context Managers

### The `with` Statement

Context managers handle setup and cleanup automatically:

```python
# File handling — file is automatically closed
with open('data.txt', 'r') as f:
    content = f.read()
# File is closed here, even if an error occurred

# Multiple context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    outfile.write(infile.read())
```

**Common ML context managers:**
```python
import torch

# Disable gradient computation (for inference)
with torch.no_grad():
    predictions = model(inputs)

# Mixed precision training
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Timer context manager
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"Elapsed: {self.elapsed:.2f}s")

with Timer():
    # Code to time
    result = expensive_operation()
```

---

## 📚 Part 5: Error Handling

### Try/Except Pattern

```python
def load_model(path):
    """Load model with error handling."""
    try:
        model = torch.load(path)
        return model
    except FileNotFoundError:
        print(f"Model file not found: {path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        raise  # Re-raise the exception

# More specific handling
def process_batch(batch):
    try:
        result = model(batch)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU OOM — try smaller batch size")
            torch.cuda.empty_cache()
            raise
        else:
            raise
```

### Assertions for Debugging

```python
def train_step(inputs, labels):
    # Validate inputs during development
    assert inputs.shape[0] == labels.shape[0], \
        f"Batch size mismatch: {inputs.shape[0]} vs {labels.shape[0]}"
    
    assert not torch.isnan(inputs).any(), "NaN in inputs!"
    
    # Training logic...
```

---

## 📚 Part 6: Type Hints

Type hints make code more readable and enable IDE autocompletion:

```python
from typing import List, Dict, Optional, Tuple, Union, Callable

def process_texts(
    texts: List[str],
    max_length: int = 512,
    tokenizer: Optional[Callable] = None
) -> List[List[int]]:
    """Process texts into token IDs.
    
    Args:
        texts: List of input strings
        max_length: Maximum sequence length
        tokenizer: Optional tokenizer function
    
    Returns:
        List of token ID sequences
    """
    if tokenizer is None:
        tokenizer = default_tokenizer
    
    return [tokenizer(t)[:max_length] for t in texts]

# Complex types
ModelConfig = Dict[str, Union[int, float, str]]

def create_model(config: ModelConfig) -> 'MyModel':
    """Create model from config dictionary."""
    return MyModel(**config)

# Type hints in classes
class Dataset:
    def __init__(self, data: List[Dict[str, any]]):
        self.data = data
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        return self.data[idx]
    
    def __len__(self) -> int:
        return len(self.data)
```

---

## 🏋️ Exercises

### Exercise 1: Comprehensions
```python
# Given this data:
students = [
    {'name': 'Alice', 'score': 85, 'passed': True},
    {'name': 'Bob', 'score': 72, 'passed': True},
    {'name': 'Charlie', 'score': 58, 'passed': False},
    {'name': 'Diana', 'score': 91, 'passed': True},
]

# 1. Get list of all names
# 2. Get list of scores for students who passed
# 3. Create dict mapping name -> score
# 4. Create dict of only passing students (name -> score)
```

### Exercise 2: Generator Function
```python
# Write a generator that yields sliding windows over a sequence
# sliding_window([1,2,3,4,5], window_size=3) should yield:
# [1,2,3], [2,3,4], [3,4,5]

def sliding_window(sequence, window_size):
    # Your code here
    pass
```

### Exercise 3: Class Implementation
```python
# Implement a simple Vocabulary class:
# - __init__ takes a list of words
# - word_to_idx: property returning word -> index dict
# - idx_to_word: property returning index -> word dict
# - encode(word): returns index (or -1 if unknown)
# - decode(idx): returns word (or "<UNK>" if invalid)
# - __len__: returns vocabulary size
# - __contains__: supports "word in vocab" syntax

class Vocabulary:
    # Your code here
    pass
```

### Exercise 4: Error Handling
```python
# Write a function that safely loads JSON from a file
# - Returns the parsed data if successful
# - Returns None and prints error if file not found
# - Returns None and prints error if JSON is invalid
# - Re-raises any other exceptions

def safe_load_json(filepath):
    # Your code here
    pass
```

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 1: Comprehensions
```python
# 1. Get list of all names
names = [s['name'] for s in students]

# 2. Get list of scores for students who passed
passing_scores = [s['score'] for s in students if s['passed']]

# 3. Create dict mapping name -> score
name_to_score = {s['name']: s['score'] for s in students}

# 4. Create dict of only passing students
passing_dict = {s['name']: s['score'] for s in students if s['passed']}
```

### Exercise 2: Generator Function
```python
def sliding_window(sequence, window_size):
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i:i + window_size]
```

### Exercise 3: Class Implementation
```python
class Vocabulary:
    def __init__(self, words):
        self._words = list(words)
        self._word_to_idx = {w: i for i, w in enumerate(self._words)}
    
    @property
    def word_to_idx(self):
        return self._word_to_idx.copy()
    
    @property
    def idx_to_word(self):
        return {i: w for w, i in self._word_to_idx.items()}
    
    def encode(self, word):
        return self._word_to_idx.get(word, -1)
    
    def decode(self, idx):
        if 0 <= idx < len(self._words):
            return self._words[idx]
        return "<UNK>"
    
    def __len__(self):
        return len(self._words)
    
    def __contains__(self, word):
        return word in self._word_to_idx
```

### Exercise 4: Error Handling
```python
import json

def safe_load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
        return None
```

</details>

---

## 🔗 What's Next?

Now that you're comfortable with Python patterns, move on to **Module 2: NumPy Fundamentals** — where you'll learn to think in arrays and matrices.

---

## 📖 References

- [Python Official Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- "Fluent Python" by Luciano Ramalho — for deep Python mastery
