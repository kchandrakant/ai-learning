"""
Module 1: Python Essentials for ML - Practice Exercises
========================================================

This module contains hands-on exercises to build fluency with Python patterns
commonly used in ML codebases. Run this file to practice interactively.

Usage:
    python exercises.py           # Run all exercises
    python exercises.py --check   # Check your solutions
"""

import json
from typing import List, Dict, Optional, Callable, Any, Generator, Tuple

# =============================================================================
# PART 1: COMPREHENSIONS
# =============================================================================

print("\n" + "="*60)
print("PART 1: COMPREHENSIONS")
print("="*60)

# Sample data for exercises
students = [
    {'name': 'Alice', 'score': 85, 'passed': True, 'subject': 'math'},
    {'name': 'Bob', 'score': 72, 'passed': True, 'subject': 'physics'},
    {'name': 'Charlie', 'score': 58, 'passed': False, 'subject': 'math'},
    {'name': 'Diana', 'score': 91, 'passed': True, 'subject': 'physics'},
    {'name': 'Eve', 'score': 67, 'passed': False, 'subject': 'math'},
]

print("\nGiven this data:")
print("students =", json.dumps(students, indent=2))

# Exercise 1.1: List Comprehensions
print("\n--- Exercise 1.1: List Comprehensions ---")
print("Complete these one-liners:")

# TODO: Get list of all student names
# names = ???
names = [s['name'] for s in students]
print(f"1. All names: {names}")

# TODO: Get scores of students who passed
# passing_scores = ???
passing_scores = [s['score'] for s in students if s['passed']]
print(f"2. Passing scores: {passing_scores}")

# TODO: Get names of math students with scores > 60
# good_math_students = ???
good_math_students = [s['name'] for s in students if s['subject'] == 'math' and s['score'] > 60]
print(f"3. Good math students: {good_math_students}")


# Exercise 1.2: Dictionary Comprehensions
print("\n--- Exercise 1.2: Dictionary Comprehensions ---")

# TODO: Create dict mapping name -> score
# name_to_score = ???
name_to_score = {s['name']: s['score'] for s in students}
print(f"1. Name to score: {name_to_score}")

# TODO: Create dict of passing students only (name -> score)
# passing_dict = ???
passing_dict = {s['name']: s['score'] for s in students if s['passed']}
print(f"2. Passing students: {passing_dict}")

# TODO: Invert a dictionary
original = {'a': 1, 'b': 2, 'c': 3}
# inverted = ???
inverted = {v: k for k, v in original.items()}
print(f"3. Inverted {original}: {inverted}")


# Exercise 1.3: Nested Comprehensions
print("\n--- Exercise 1.3: Nested Comprehensions ---")

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Given matrix: {matrix}")

# TODO: Flatten the matrix to a single list
# flat = ???
flat = [item for row in matrix for item in row]
print(f"1. Flattened: {flat}")

# TODO: Get all even numbers from the matrix
# evens = ???
evens = [item for row in matrix for item in row if item % 2 == 0]
print(f"2. Even numbers: {evens}")

# TODO: Transpose the matrix (swap rows and columns)
# transposed = ???
transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
print(f"3. Transposed: {transposed}")


# =============================================================================
# PART 2: GENERATORS
# =============================================================================

print("\n" + "="*60)
print("PART 2: GENERATORS")
print("="*60)

# Exercise 2.1: Basic Generator
print("\n--- Exercise 2.1: Basic Generator ---")

def count_up_to(n: int) -> Generator[int, None, None]:
    """Yield numbers from 1 to n."""
    for i in range(1, n + 1):
        yield i

print("count_up_to(5):", list(count_up_to(5)))


# Exercise 2.2: Batch Generator (Common ML Pattern)
print("\n--- Exercise 2.2: Batch Generator ---")

def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Yield batches of data.
    
    This is THE most common pattern in ML - splitting data into batches
    for training.
    
    Args:
        data: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Lists of items, each of length batch_size (last may be smaller)
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

data = list(range(10))
print(f"Data: {data}")
print(f"Batches of 3: {list(batch_generator(data, 3))}")


# Exercise 2.3: Sliding Window Generator
print("\n--- Exercise 2.3: Sliding Window Generator ---")

def sliding_window(sequence: List[Any], window_size: int) -> Generator[List[Any], None, None]:
    """
    Yield sliding windows over a sequence.
    
    Used in time series, NLP (n-grams), and other sequential data.
    
    Args:
        sequence: Input sequence
        window_size: Size of each window
        
    Yields:
        Windows of the specified size
    """
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i:i + window_size]

seq = [1, 2, 3, 4, 5]
print(f"Sequence: {seq}")
print(f"Windows of size 3: {list(sliding_window(seq, 3))}")


# Exercise 2.4: Generator Expression vs List Comprehension
print("\n--- Exercise 2.4: Memory Efficiency ---")

import sys

# List comprehension - stores all values in memory
list_comp = [x**2 for x in range(1000)]
print(f"List size: {sys.getsizeof(list_comp)} bytes")

# Generator expression - computes on demand
gen_exp = (x**2 for x in range(1000))
print(f"Generator size: {sys.getsizeof(gen_exp)} bytes")

# Both produce the same values
print(f"Sum from list: {sum(list_comp)}")
gen_exp = (x**2 for x in range(1000))  # Recreate (generators exhaust)
print(f"Sum from generator: {sum(gen_exp)}")


# =============================================================================
# PART 3: CLASSES - THE PYTORCH PATTERN
# =============================================================================

print("\n" + "="*60)
print("PART 3: CLASSES")
print("="*60)

# Exercise 3.1: Basic Class with Properties
print("\n--- Exercise 3.1: Vocabulary Class ---")

class Vocabulary:
    """
    A vocabulary for mapping between words and indices.
    
    This pattern appears in every NLP codebase.
    """
    
    def __init__(self, words: List[str], add_special_tokens: bool = True):
        """
        Initialize vocabulary.
        
        Args:
            words: List of words to include
            add_special_tokens: Whether to add <PAD>, <UNK> tokens
        """
        self._words = []
        self._word_to_idx = {}
        
        if add_special_tokens:
            self._words = ['<PAD>', '<UNK>']
        
        for word in words:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = len(self._words)
                self._words.append(word)
        
        # Update word_to_idx for special tokens
        self._word_to_idx = {w: i for i, w in enumerate(self._words)}
    
    @property
    def word_to_idx(self) -> Dict[str, int]:
        """Return word to index mapping."""
        return self._word_to_idx.copy()
    
    @property
    def idx_to_word(self) -> Dict[int, str]:
        """Return index to word mapping."""
        return {i: w for w, i in self._word_to_idx.items()}
    
    def encode(self, word: str) -> int:
        """Convert word to index."""
        return self._word_to_idx.get(word, self._word_to_idx.get('<UNK>', -1))
    
    def decode(self, idx: int) -> str:
        """Convert index to word."""
        if 0 <= idx < len(self._words):
            return self._words[idx]
        return '<UNK>'
    
    def encode_sequence(self, words: List[str]) -> List[int]:
        """Encode a sequence of words."""
        return [self.encode(w) for w in words]
    
    def decode_sequence(self, indices: List[int]) -> List[str]:
        """Decode a sequence of indices."""
        return [self.decode(i) for i in indices]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._words)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self._word_to_idx
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Vocabulary(size={len(self)}, words={self._words[:5]}...)"


# Test the Vocabulary class
vocab = Vocabulary(['the', 'cat', 'sat', 'on', 'mat'])
print(f"Vocabulary: {vocab}")
print(f"Size: {len(vocab)}")
print(f"'cat' in vocab: {'cat' in vocab}")
print(f"'dog' in vocab: {'dog' in vocab}")
print(f"encode('cat'): {vocab.encode('cat')}")
print(f"encode('dog'): {vocab.encode('dog')} (unknown word)")
print(f"decode(3): {vocab.decode(3)}")
print(f"encode_sequence: {vocab.encode_sequence(['the', 'cat', 'sat'])}")


# Exercise 3.2: Model-like Class (PyTorch Pattern)
print("\n--- Exercise 3.2: Model Pattern ---")

class SimpleLayer:
    """
    A simple layer demonstrating the forward() pattern.
    
    This is how PyTorch models work - you define layers in __init__
    and computation in forward.
    """
    
    def __init__(self, input_size: int, output_size: int):
        """Initialize layer parameters."""
        self.input_size = input_size
        self.output_size = output_size
        # In real PyTorch, these would be nn.Parameter
        self.weight = [[0.1] * output_size for _ in range(input_size)]
        self.bias = [0.0] * output_size
    
    def forward(self, x: List[float]) -> List[float]:
        """
        Forward pass - compute output from input.
        
        Args:
            x: Input vector of size input_size
            
        Returns:
            Output vector of size output_size
        """
        assert len(x) == self.input_size, f"Expected {self.input_size} inputs, got {len(x)}"
        
        output = []
        for j in range(self.output_size):
            val = self.bias[j]
            for i in range(self.input_size):
                val += x[i] * self.weight[i][j]
            output.append(val)
        return output
    
    def __call__(self, x: List[float]) -> List[float]:
        """Make the layer callable - layer(x) calls forward(x)."""
        return self.forward(x)
    
    def __repr__(self) -> str:
        return f"SimpleLayer({self.input_size}, {self.output_size})"


# Test the layer
layer = SimpleLayer(3, 2)
print(f"Layer: {layer}")
x = [1.0, 2.0, 3.0]
print(f"Input: {x}")
print(f"Output: {layer(x)}")  # Uses __call__ -> forward


# =============================================================================
# PART 4: CONTEXT MANAGERS
# =============================================================================

print("\n" + "="*60)
print("PART 4: CONTEXT MANAGERS")
print("="*60)

# Exercise 4.1: Timer Context Manager
print("\n--- Exercise 4.1: Timer Context Manager ---")

import time

class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer() as t:
            # code to time
        print(t.elapsed)
    """
    
    def __init__(self, name: str = "Block"):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        """Called when entering the 'with' block."""
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block."""
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions


# Test the timer
with Timer("Sum computation"):
    total = sum(range(1000000))
print(f"Result: {total}")


# Exercise 4.2: Temporary Value Context Manager
print("\n--- Exercise 4.2: Temporary Value ---")

class TemporaryValue:
    """
    Temporarily modify a dictionary value, then restore it.
    
    Useful for temporarily changing configuration during a block.
    """
    
    def __init__(self, d: dict, key: str, value: Any):
        self.d = d
        self.key = key
        self.new_value = value
        self.old_value = None
        self.had_key = False
    
    def __enter__(self):
        self.had_key = self.key in self.d
        if self.had_key:
            self.old_value = self.d[self.key]
        self.d[self.key] = self.new_value
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.had_key:
            self.d[self.key] = self.old_value
        else:
            del self.d[self.key]
        return False


# Test temporary value
config = {'learning_rate': 0.001, 'batch_size': 32}
print(f"Before: {config}")

with TemporaryValue(config, 'learning_rate', 0.01):
    print(f"Inside: {config}")

print(f"After: {config}")


# =============================================================================
# PART 5: ERROR HANDLING
# =============================================================================

print("\n" + "="*60)
print("PART 5: ERROR HANDLING")
print("="*60)

# Exercise 5.1: Safe File Loading
print("\n--- Exercise 5.1: Safe JSON Loading ---")

def safe_load_json(filepath: str) -> Optional[Dict]:
    """
    Safely load JSON from a file.
    
    Returns:
        Parsed JSON data, or None if loading fails
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error loading {filepath}: {e}")
        raise


# Test with non-existent file
result = safe_load_json("nonexistent.json")
print(f"Result for missing file: {result}")


# Exercise 5.2: Retry Decorator
print("\n--- Exercise 5.2: Retry Decorator ---")

def retry(max_attempts: int = 3, delay: float = 0.1):
    """
    Decorator that retries a function on failure.
    
    Common pattern for network calls, API requests, etc.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# Test retry decorator
attempt_count = 0

@retry(max_attempts=3, delay=0.1)
def flaky_function():
    """Simulates a function that fails sometimes."""
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError("Random failure!")
    return "Success!"

print(f"Result: {flaky_function()}")


# =============================================================================
# PART 6: TYPE HINTS
# =============================================================================

print("\n" + "="*60)
print("PART 6: TYPE HINTS")
print("="*60)

print("\n--- Exercise 6.1: Typed Functions ---")

def process_batch(
    data: List[Dict[str, Any]],
    batch_size: int = 32,
    transform: Optional[Callable[[Dict], Dict]] = None
) -> List[List[Dict[str, Any]]]:
    """
    Process data into batches with optional transformation.
    
    This demonstrates:
    - List and Dict type hints
    - Optional parameters
    - Callable type hints
    - Docstrings with typed parameters
    
    Args:
        data: List of data items (dictionaries)
        batch_size: Number of items per batch
        transform: Optional function to apply to each item
        
    Returns:
        List of batches, where each batch is a list of items
    """
    if transform:
        data = [transform(item) for item in data]
    
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


# Test the function
sample_data = [{'id': i, 'value': i * 10} for i in range(5)]
print(f"Original data: {sample_data}")

# With transform
def double_value(item: Dict[str, Any]) -> Dict[str, Any]:
    return {**item, 'value': item['value'] * 2}

batched = process_batch(sample_data, batch_size=2, transform=double_value)
print(f"Batched with transform: {batched}")


# =============================================================================
# SUMMARY AND PRACTICE CHALLENGES
# =============================================================================

print("\n" + "="*60)
print("PRACTICE CHALLENGES")
print("="*60)

print("""
Try implementing these on your own:

1. COMPREHENSION CHALLENGE:
   Given a list of sentences, create a dictionary mapping each unique
   word to its frequency across all sentences.
   
   sentences = ["the cat sat", "the dog ran", "the cat ran"]
   # Expected: {'the': 3, 'cat': 2, 'sat': 1, 'dog': 1, 'ran': 2}

2. GENERATOR CHALLENGE:
   Write a generator that yields prime numbers up to n.
   
   list(primes_up_to(20))
   # Expected: [2, 3, 5, 7, 11, 13, 17, 19]

3. CLASS CHALLENGE:
   Implement a DataLoader class that:
   - Takes a dataset and batch_size
   - Has __iter__ that yields batches
   - Has __len__ that returns number of batches
   - Supports shuffle parameter

4. DECORATOR CHALLENGE:
   Write a @log_calls decorator that logs function name, arguments,
   and return value each time a function is called.

5. CONTEXT MANAGER CHALLENGE:
   Write a context manager that temporarily changes the working directory
   and restores it when done.

Run this file with --check to see solutions!
""")


# =============================================================================
# SOLUTIONS (shown with --check flag)
# =============================================================================

def show_solutions():
    """Display solutions to practice challenges."""
    
    print("\n" + "="*60)
    print("SOLUTIONS")
    print("="*60)
    
    # Challenge 1: Word Frequency
    print("\n--- Challenge 1: Word Frequency ---")
    sentences = ["the cat sat", "the dog ran", "the cat ran"]
    word_freq = {}
    for sentence in sentences:
        for word in sentence.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    # Or as a one-liner with Counter:
    from collections import Counter
    word_freq_v2 = Counter(word for sentence in sentences for word in sentence.split())
    print(f"Word frequencies: {dict(word_freq_v2)}")
    
    # Challenge 2: Prime Generator
    print("\n--- Challenge 2: Prime Generator ---")
    def primes_up_to(n: int) -> Generator[int, None, None]:
        for num in range(2, n + 1):
            is_prime = True
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                yield num
    print(f"Primes up to 20: {list(primes_up_to(20))}")
    
    # Challenge 3: DataLoader
    print("\n--- Challenge 3: DataLoader ---")
    import random
    
    class DataLoader:
        def __init__(self, dataset: List[Any], batch_size: int, shuffle: bool = False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
        
        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield [self.dataset[j] for j in batch_indices]
        
        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    loader = DataLoader(list(range(10)), batch_size=3, shuffle=True)
    print(f"DataLoader batches: {list(loader)}")
    
    # Challenge 4: Log Calls Decorator
    print("\n--- Challenge 4: Log Calls Decorator ---")
    def log_calls(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}(args={args}, kwargs={kwargs})")
            result = func(*args, **kwargs)
            print(f"  -> returned {result}")
            return result
        return wrapper
    
    @log_calls
    def add(a, b):
        return a + b
    
    add(2, 3)
    
    # Challenge 5: Change Directory Context Manager
    print("\n--- Challenge 5: Change Directory ---")
    import os
    
    class ChangeDirectory:
        def __init__(self, path: str):
            self.path = path
            self.original_path = None
        
        def __enter__(self):
            self.original_path = os.getcwd()
            os.chdir(self.path)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            os.chdir(self.original_path)
            return False
    
    print(f"Current: {os.getcwd()}")
    # Uncomment to test:
    # with ChangeDirectory('..'):
    #     print(f"Inside: {os.getcwd()}")
    # print(f"After: {os.getcwd()}")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        show_solutions()
    
    print("\n" + "="*60)
    print("Module 1 Complete!")
    print("="*60)
    print("""
Next steps:
1. Try the practice challenges on your own
2. Run with --check to see solutions
3. Move on to Module 2: NumPy Fundamentals
""")
