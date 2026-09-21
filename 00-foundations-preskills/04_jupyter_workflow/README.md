# Module 4: Jupyter Workflow

Jupyter notebooks are the standard environment for ML experimentation. They let you write code, see outputs, visualize results, and document your work — all in one place. This module covers the workflow and best practices that make notebooks effective.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Navigate Jupyter Notebook and JupyterLab efficiently
- Use magic commands for timing, debugging, and system interaction
- Follow best practices for reproducible notebooks
- Debug effectively in notebooks
- Know when to move code from notebooks to Python modules

---

## 📚 Part 1: Getting Started

### Jupyter Options

| Tool | Best For |
|------|----------|
| **Jupyter Notebook** | Simple, classic interface |
| **JupyterLab** | Modern IDE-like environment, multiple panels |
| **Google Colab** | Free GPUs, no setup, collaboration |
| **VS Code Notebooks** | Integration with IDE features |

### Starting Jupyter

```bash
# Jupyter Notebook
jupyter notebook

# JupyterLab (recommended)
jupyter lab

# In a specific directory
jupyter lab --notebook-dir=/path/to/projects
```

### Google Colab

- Go to [colab.research.google.com](https://colab.research.google.com)
- Free GPU/TPU access (limited)
- Saves to Google Drive
- Great for learning and small experiments

---

## 📚 Part 2: Notebook Basics

### Cell Types

**Code cells** — Run Python code:
```python
x = 5
print(x * 2)
```

**Markdown cells** — Documentation:
```markdown
# Heading
This is **bold** and *italic*.

- Bullet point
- Another point

```python
# Code block in markdown
print("hello")
```
```

### Essential Keyboard Shortcuts

**Command mode** (press `Esc` first):

| Shortcut | Action |
|----------|--------|
| `A` | Insert cell above |
| `B` | Insert cell below |
| `D D` | Delete cell |
| `M` | Change to Markdown |
| `Y` | Change to Code |
| `Shift + Enter` | Run and move to next |
| `Ctrl + Enter` | Run and stay |
| `Z` | Undo cell delete |
| `L` | Toggle line numbers |

**Edit mode** (press `Enter` to enter):

| Shortcut | Action |
|----------|--------|
| `Tab` | Autocomplete |
| `Shift + Tab` | Show docstring |
| `Ctrl + /` | Comment/uncomment |
| `Ctrl + Shift + -` | Split cell at cursor |

### Cell Output

```python
# Last expression is displayed automatically
x = 5
x  # This shows: 5

# Multiple outputs with display()
from IPython.display import display
display(x)
display(x * 2)

# Suppress output with semicolon
plt.plot([1, 2, 3]);  # No "Out[]: [<matplotlib.lines.Line2D ...>]"
```

---

## 📚 Part 3: Magic Commands

Magic commands are special Jupyter commands that start with `%` (line magic) or `%%` (cell magic).

### Timing Code

```python
# Time a single line
%timeit np.dot(a, b)
# Output: 1.23 µs ± 45.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# Time a cell
%%timeit
result = 0
for i in range(1000):
    result += i

# Simple wall-clock time
%time result = expensive_function()
# Output: CPU times: user 1.23 s, sys: 456 ms, total: 1.69 s
#         Wall time: 1.72 s
```

### Running Shell Commands

```python
# Single command
!pip install numpy
!ls -la
!nvidia-smi  # Check GPU

# Capture output
files = !ls *.py
print(files)  # Python list of filenames

# In Windows
!dir
!where python
```

### Useful Magics

```python
# Show all available magics
%lsmagic

# Matplotlib inline (usually automatic now)
%matplotlib inline

# Auto-reload modules when they change (for development)
%load_ext autoreload
%autoreload 2

# Now changes to my_module.py are automatically reloaded
import my_module

# Show variables
%who      # List variable names
%whos     # List with details
%who_ls   # Return as list

# History
%history -n 1-10  # Show commands 1-10

# Reset namespace
%reset -f  # Clear all variables (careful!)

# Run Python file
%run script.py

# Load file content into cell
%load script.py

# Write cell content to file
%%writefile script.py
def hello():
    print("Hello!")
```

### Debugging

```python
# Drop into debugger after error
%debug

# Or use pdb
%pdb on  # Auto-enter debugger on exception

# Interactive debugger commands:
# n - next line
# s - step into function
# c - continue
# q - quit
# p variable - print variable
# l - show code context
```

---

## 📚 Part 4: Best Practices

### Notebook Structure

A well-organized notebook:

```markdown
# Project Title

Brief description of what this notebook does.

## Setup

Import libraries, set random seeds, configure settings.

## Data Loading

Load and inspect data.

## Exploration / Analysis

Explore data, try experiments.

## Results

Final results, visualizations.

## Conclusions

What did we learn?
```

### Reproducibility

```python
# ALWAYS set random seeds at the start
import numpy as np
import random
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Document versions
import sys
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
```

### Restart and Run All

**Critical habit:** Before sharing or trusting results, always:
1. `Kernel > Restart & Run All`
2. Verify all cells execute in order without error

Hidden state bugs occur when you:
- Run cells out of order
- Delete cells that defined variables
- Modify variables interactively

### Keep Cells Small

```python
# Bad: One giant cell
# ... 100 lines of code ...

# Good: Small, focused cells
# Cell 1: Load data
data = load_data('file.csv')

# Cell 2: Preprocess
data = preprocess(data)

# Cell 3: Split
train, test = split(data)

# Cell 4: Train
model = train_model(train)
```

### Use Markdown Liberally

```markdown
## Data Preprocessing

We apply the following transformations:
1. Remove null values (found 23 rows with nulls)
2. Normalize features to [0, 1] range
3. Encode categorical variables

**Note:** The `category` column has 15 unique values.
```

---

## 📚 Part 5: Colab-Specific Tips

### Check GPU

```python
# Check if GPU is available
!nvidia-smi

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Access files
!ls /content/drive/MyDrive/
```

### Install Packages

```python
# Packages are lost when runtime disconnects
!pip install transformers datasets
```

### Upload/Download Files

```python
from google.colab import files

# Upload
uploaded = files.upload()

# Download
files.download('results.csv')
```

### Prevent Disconnection

Colab disconnects after ~90 minutes of inactivity. For long training:
- Keep the browser tab active
- Consider saving checkpoints frequently

---

## 📚 Part 6: When to Use .py Files

Notebooks are great for experimentation, but some code belongs in Python modules.

### Move to .py files when:

- **Reusable code** — Functions/classes used across notebooks
- **Complex logic** — Hard to read/debug in cells
- **Production code** — Anything that will be deployed
- **Version control** — Notebooks have messy diffs

### Typical Project Structure

```
project/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py        # Data loading/processing
│   ├── model.py       # Model definitions
│   ├── train.py       # Training logic
│   └── utils.py       # Utilities
├── requirements.txt
└── README.md
```

### Using Local Modules in Notebooks

```python
# Add project root to path
import sys
sys.path.append('..')

# Now import your modules
from src.model import MyModel
from src.data import load_dataset

# With autoreload, changes to .py files are picked up
%load_ext autoreload
%autoreload 2

# Changes to src/model.py now auto-reload
```

---

## 📚 Part 7: Common Pitfalls

### Hidden State

```python
# Cell 1
x = 10

# Cell 2 (run this)
y = x * 2  # y = 20

# Cell 1 (run again, modified)
x = 5

# Cell 3
print(y)  # Still 20! Not updated to 10.
```

**Solution:** Restart & Run All regularly.

### Memory Issues

```python
# Bad: Keeping large objects
huge_data = load_huge_dataset()
processed = process(huge_data)
# huge_data still in memory!

# Good: Delete when done
huge_data = load_huge_dataset()
processed = process(huge_data)
del huge_data  # Free memory

# Or use gc
import gc
gc.collect()

# Check memory usage
%whos
```

### Output Overload

```python
# Bad: Printing 10000 items
for item in large_list:
    print(item)  # Notebook becomes unresponsive

# Good: Sample or summarize
print(large_list[:10])  # First 10
print(f"Total: {len(large_list)}")
```

---

## 🏋️ Exercises

### Exercise 1: Magic Commands
```python
# Use magic commands to:
# 1. Time how long it takes to create a 1000x1000 random matrix
# 2. List all variables currently defined
# 3. Show the last 5 commands you ran
# 4. Check your Python version

# Your code here
```

### Exercise 2: Notebook Organization
```markdown
Create a well-structured notebook that:
1. Has a title and description
2. Imports libraries in a setup section
3. Sets a random seed
4. Loads some sample data (use np.random to generate)
5. Creates a simple visualization
6. Has markdown documentation explaining each section

Save it and verify it works with "Restart & Run All"
```

### Exercise 3: Debugging Practice
```python
# This code has a bug. Use %debug to find it.
def process_data(data):
    result = []
    for item in data:
        result.append(item['value'] * 2)
    return result

data = [{'value': 1}, {'value': 2}, {'valu': 3}]  # Note the typo
process_data(data)

# After the error, run %debug and use 'p item' to inspect
```

### Exercise 4: Module Integration
```python
# 1. Create a file called 'my_utils.py' with:
#    - A function `normalize(data)` that normalizes to [0, 1]
#    - A function `summarize(data)` that prints mean, std, min, max

# 2. In your notebook, use autoreload to import and test it
# 3. Modify the .py file and verify the changes are picked up

# Your code here
```

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 1: Magic Commands
```python
# 1. Time matrix creation
%timeit np.random.randn(1000, 1000)

# 2. List variables
%whos

# 3. Show last 5 commands
%history -n -5:

# 4. Python version
import sys
print(sys.version)
# Or: !python --version
```

### Exercise 2: Notebook Organization
```python
# Cell 1 (Markdown)
"""
# Sample Data Analysis

This notebook demonstrates proper notebook structure.
"""

# Cell 2 (Code - Setup)
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

print("Setup complete!")

# Cell 3 (Markdown)
"""
## Data Loading

Generate sample data for demonstration.
"""

# Cell 4 (Code)
# Generate sample data
n_samples = 1000
data = np.random.randn(n_samples) * 2 + 5  # Mean=5, std=2

print(f"Generated {n_samples} samples")
print(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")

# Cell 5 (Markdown)
"""
## Visualization

Plot the distribution of our sample data.
"""

# Cell 6 (Code)
plt.figure(figsize=(10, 5))
plt.hist(data, bins=30, edgecolor='black')
plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Sample Data Distribution')
plt.legend()
plt.show()

# Cell 7 (Markdown)
"""
## Conclusions

The generated data follows the expected normal distribution 
with mean ≈ 5 and standard deviation ≈ 2.
"""
```

### Exercise 3: Debugging Practice
```python
# After running the buggy code and getting KeyError:
%debug

# In debugger:
# > p item
# {'valu': 3}
# > p item.keys()
# dict_keys(['valu'])
# > q

# The bug is a typo: 'valu' instead of 'value' in the third dict
```

### Exercise 4: Module Integration
```python
# my_utils.py content:
%%writefile my_utils.py
import numpy as np

def normalize(data):
    """Normalize data to [0, 1] range."""
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min())

def summarize(data):
    """Print summary statistics."""
    data = np.array(data)
    print(f"Mean: {data.mean():.4f}")
    print(f"Std:  {data.std():.4f}")
    print(f"Min:  {data.min():.4f}")
    print(f"Max:  {data.max():.4f}")

# In notebook:
%load_ext autoreload
%autoreload 2

import my_utils

data = np.random.randn(100)
my_utils.summarize(data)
normalized = my_utils.normalize(data)
my_utils.summarize(normalized)
```

</details>

---

## 🔗 What's Next?

You now have a solid workflow for ML experimentation. Move on to **Module 5: Reading ML Papers** — the skill that unlocks deeper understanding of everything you'll learn.

---

## 📖 References

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [IPython Documentation](https://ipython.readthedocs.io/)
- "Ten Simple Rules for Reproducible Research in Jupyter Notebooks"
