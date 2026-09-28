# Module 1: Python Essentials for ML

This module covers the Python patterns you'll encounter repeatedly in machine learning codebases. It's not a Python tutorial from scratch — it focuses on the specific patterns that ML code uses heavily.

---

## 🎯 Why This Matters

Before diving into neural networks and loss functions, you need to be *fluent* in the language ML is written in. Not just "know Python" — but recognize the specific patterns that appear over and over in PyTorch, TensorFlow, scikit-learn, and Hugging Face code.

When you see this in a codebase:

```python
def forward(self, x, mask=None, **kwargs):
    return self.transformer(x, attention_mask=mask, **kwargs)
```

You should instantly recognize:
- `self` → this is a method on a class
- `mask=None` → optional parameter with default
- `**kwargs` → pass-through arguments to another function

**The patterns in this module appear in virtually every ML repository.** Master them once, read ML code forever.

---

## 📚 Part 1: Functions — The ML Patterns

### Why Functions Matter in ML

ML functions often have **many configurable parameters**. A training function might have 20+ arguments. Understanding how Python handles default values, keyword arguments, and argument forwarding is essential.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ML Function Signature                           │
│                                                                     │
│  def train(model, data, epochs=10, lr=0.001, batch_size=32, ...)   │
│            ─────  ────  ─────────  ────────  ─────────────         │
│              │     │        │          │           │                │
│              │     │        │          │           └─ Default       │
│              │     │        │          └─ Default                   │
│              │     │        └─ Default                              │
│              │     └─ Required positional                           │
│              └─ Required positional                                 │
└─────────────────────────────────────────────────────────────────────┘
```

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

> **Intuition:** Default arguments let you provide sensible defaults while allowing customization. This is why you can call `model.fit(X, y)` with just data, but also `model.fit(X, y, epochs=100, validation_split=0.2)` when you need control.

---

### *args and **kwargs

These appear everywhere in ML code, especially in:
- Wrapper functions (decorators, callbacks)
- Class inheritance (passing arguments to parent classes)
- Flexible APIs

```
┌─────────────────────────────────────────────────────────────────────┐
│                    *args and **kwargs Flow                          │
│                                                                     │
│    Your function          Another function                          │
│   ┌──────────────┐       ┌──────────────┐                          │
│   │ def wrapper( │       │              │                          │
│   │   *args,     │──────▶│  func(       │                          │
│   │   **kwargs   │       │    *args,    │                          │
│   │ ):           │       │    **kwargs  │                          │
│   │   func(...)  │       │  )           │                          │
│   └──────────────┘       └──────────────┘                          │
│                                                                     │
│   *args   = tuple of positional arguments: (1, 2, 3)               │
│   **kwargs = dict of keyword arguments: {'lr': 0.01, 'epochs': 10} │
└─────────────────────────────────────────────────────────────────────┘
```

**Example: Logging decorator**
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

> **Why it works:** `*args` collects extra positional arguments into a tuple. `**kwargs` collects extra keyword arguments into a dictionary. When you call `func(*args, **kwargs)`, you're "unpacking" them back into individual arguments.

---

### Comparison: Argument Styles

| Style | Use Case | Example |
|-------|----------|---------|
| `def f(x, y)` | Fixed, required args | `def add(a, b)` |
| `def f(x, y=10)` | Optional with default | `def train(data, epochs=10)` |
| `def f(*args)` | Variable positional | `def concat(*tensors)` |
| `def f(**kwargs)` | Variable keyword | `def config(**options)` |
| `def f(x, *args, **kwargs)` | Flexible forwarding | Decorators, wrappers |

---

## 📚 Part 2: Comprehensions and Generators

### Why Comprehensions Matter in ML

Data processing in ML is all about transformations:
- Convert text to tokens
- Filter valid samples
- Extract features from records

Comprehensions express these transformations concisely and Pythonically.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Comprehension Mental Model                       │
│                                                                     │
│   [expression for item in iterable if condition]                    │
│    ──────────     ────     ────────    ─────────                   │
│        │           │          │            │                        │
│        │           │          │            └─ Optional filter       │
│        │           │          └─ Source data                        │
│        │           └─ Loop variable                                 │
│        └─ What to compute for each item                             │
│                                                                     │
│   Equivalent loop:                                                  │
│   result = []                                                       │
│   for item in iterable:                                             │
│       if condition:                                                 │
│           result.append(expression)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### List Comprehensions

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

---

### Generator Expressions (Memory Efficient)

For large datasets, generators don't load everything into memory:

```
┌─────────────────────────────────────────────────────────────────────┐
│              List vs Generator Memory Usage                         │
│                                                                     │
│   List: [x**2 for x in range(1_000_000)]                           │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ 0 │ 1 │ 4 │ 9 │ 16 │ ... │ 999998000001 │  ← All in memory  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│   Memory: ~8 MB                                                     │
│                                                                     │
│   Generator: (x**2 for x in range(1_000_000))                      │
│   ┌─────────┐                                                       │
│   │ next()  │ → computes one value at a time                       │
│   └─────────┘                                                       │
│   Memory: ~100 bytes (just the generator object)                    │
└─────────────────────────────────────────────────────────────────────┘
```

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

> **Why it works:** `yield` pauses the function and returns a value. When you call `next()` or iterate, it resumes from where it left off. This is how PyTorch DataLoaders work under the hood.

---

## 📚 Part 3: Classes — The PyTorch Pattern

### Why Classes Matter in ML

Every PyTorch model is a class. Every Keras layer is a class. Understanding the class pattern unlocks:
- Reading any ML framework's source code
- Building custom models
- Understanding inheritance and composition

```
┌─────────────────────────────────────────────────────────────────────┐
│                    The nn.Module Pattern                            │
│                                                                     │
│   class MyModel(nn.Module):      ← Inherit from nn.Module           │
│       def __init__(self, ...):                                      │
│           super().__init__()     ← MUST call parent's __init__      │
│           self.layer = nn.Linear(...)  ← Define layers as attrs     │
│                                                                     │
│       def forward(self, x):      ← Define forward pass              │
│           return self.layer(x)                                      │
│                                                                     │
│   model = MyModel(...)                                              │
│   output = model(input)          ← Calls forward() automatically    │
└─────────────────────────────────────────────────────────────────────┘
```

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

> **Why `super().__init__()`?** The parent class (`nn.Module`) sets up essential bookkeeping: tracking parameters, handling device placement, enabling `.train()` / `.eval()` modes. Without it, your model won't work correctly.

---

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

### Common Dunder Methods in ML

| Method | Purpose | Example Use |
|--------|---------|-------------|
| `__init__` | Initialize object | Set up layers, load config |
| `__repr__` | Debug representation | `print(model)` |
| `__len__` | Support `len()` | Dataset size |
| `__getitem__` | Support `obj[i]` | Dataset indexing |
| `__call__` | Support `obj()` | PyTorch's forward pass |
| `__iter__` | Support `for x in obj` | DataLoader iteration |

---

## 📚 Part 4: Context Managers

### Why Context Managers Matter in ML

Context managers handle setup and cleanup automatically. In ML, they're used for:
- Disabling gradients during inference
- Mixed precision training
- Timing code blocks
- Managing GPU memory

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Context Manager Flow                             │
│                                                                     │
│   with context_manager as cm:                                       │
│       │                                                             │
│       ├── __enter__() called  ← Setup                               │
│       │                                                             │
│       │   # Your code runs here                                     │
│       │                                                             │
│       └── __exit__() called   ← Cleanup (even if error!)            │
│                                                                     │
│   # After 'with' block, cleanup is guaranteed                       │
└─────────────────────────────────────────────────────────────────────┘
```

### The `with` Statement

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
```

### Custom Context Manager

```python
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

> **Why it works:** `__enter__` runs at the start of the `with` block, `__exit__` runs at the end — even if an exception occurs. This guarantees cleanup.

---

## 📚 Part 5: Error Handling

### Why Error Handling Matters in ML

ML training is fragile. Common issues:
- Out of memory errors
- NaN values appearing
- File not found
- Shape mismatches

Good error handling helps you recover gracefully and debug faster.

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

### Common Pitfalls in Error Handling

| Pitfall | Problem | Better Approach |
|---------|---------|-----------------|
| Bare `except:` | Catches everything, hides bugs | Catch specific exceptions |
| Silent failures | `except: pass` | At least log the error |
| Not re-raising | Swallowing important errors | Use `raise` to propagate |
| Assertions in production | Disabled with `-O` flag | Use explicit `if` + `raise` |

---

## 📚 Part 6: Type Hints

### Why Type Hints Matter in ML

Type hints make code more readable and enable IDE autocompletion. When you see:

```python
def process_texts(texts: List[str]) -> List[List[int]]:
```

You immediately know: input is a list of strings, output is a list of token ID lists.

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
```

### Common Type Patterns in ML

```python
# Type aliases for clarity
ModelConfig = Dict[str, Union[int, float, str]]
Tensor = 'torch.Tensor'  # Forward reference

# Function signatures
def create_model(config: ModelConfig) -> 'MyModel':
    """Create model from config dictionary."""
    return MyModel(**config)

# Class with typed attributes
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

## 🎯 Key Takeaways

1. **`*args` and `**kwargs`** enable flexible function signatures — essential for decorators, wrappers, and class inheritance in ML frameworks.

2. **Comprehensions** are the Pythonic way to transform data. Use list comprehensions for transformations, dict comprehensions for lookups, generators for large data.

3. **The nn.Module pattern** (`__init__` + `super()` + `forward`) is how every PyTorch model works. Master it once, understand all PyTorch code.

4. **Context managers** (`with` statements) guarantee cleanup. Use them for `torch.no_grad()`, file handling, and timing.

5. **Error handling** helps you debug faster. Catch specific exceptions, provide helpful messages, and re-raise when appropriate.

6. **Type hints** document your code's contract. They help you and your IDE understand what goes in and what comes out.

7. **Python patterns compound** — when you recognize `*args`, comprehensions, and classes instantly, you can focus on the ML concepts instead of deciphering syntax.

---

## 🔗 What's Next?

Now that you're comfortable with Python patterns, move on to **Module 2: NumPy Fundamentals** — where you'll learn to think in arrays and matrices.

---

## 📖 References

- [Python Official Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- "Fluent Python" by Luciano Ramalho — for deep Python mastery
