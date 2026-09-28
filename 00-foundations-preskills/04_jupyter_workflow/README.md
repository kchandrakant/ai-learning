# Module 4: Jupyter Workflow

Jupyter notebooks are the standard environment for ML experimentation. They let you write code, see outputs, visualize results, and document your work — all in one place. This module covers the workflow and best practices that make notebooks effective.

---

## 🎯 Why This Matters

Notebooks aren't just a place to run code — they're where ideas become experiments. But notebooks can also become messy, unreproducible disasters if used carelessly.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Notebook: Power and Danger                          │
│                                                                     │
│   Power:                          Danger:                           │
│   ─────                           ──────                            │
│   ✓ Rapid iteration               ✗ Hidden state                    │
│   ✓ Inline visualization          ✗ Out-of-order execution          │
│   ✓ Interactive exploration       ✗ Unreproducible results          │
│   ✓ Documentation + code          ✗ Hard to version control         │
│   ✓ Share complete analyses       ✗ "Works on my machine"           │
│                                                                     │
│   This module teaches you to get the power while avoiding danger.   │
└─────────────────────────────────────────────────────────────────────┘
```

The goal: **reproducible, readable notebooks** that work when you "Restart & Run All."

---

## 📚 Part 1: Getting Started

### Jupyter Options

| Tool | Best For | Key Feature |
|------|----------|-------------|
| **Jupyter Notebook** | Simple, classic interface | Lightweight, familiar |
| **JupyterLab** | Modern IDE-like environment | Multiple panels, file browser |
| **Google Colab** | Free GPUs, no setup | Cloud-based, collaboration |
| **VS Code Notebooks** | Integration with IDE | Debugging, git integration |
| **Kaggle Notebooks** | Competitions, free GPUs | Pre-installed packages, datasets |

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

> **Tip:** Colab's free tier is excellent for learning. You get GPUs without any setup.

---

## 📚 Part 2: Notebook Basics

### Cell Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Cell Types                                      │
│                                                                     │
│   Code Cell:                      Markdown Cell:                    │
│   ┌────────────────────┐          ┌────────────────────┐           │
│   │ x = 5              │          │ # Heading          │           │
│   │ print(x * 2)       │          │ This is **bold**   │           │
│   ├────────────────────┤          │ - Bullet point     │           │
│   │ Out: 10            │          └────────────────────┘           │
│   └────────────────────┘                                            │
│   Press Y to convert              Press M to convert                │
│                                                                     │
│   Raw Cell (rarely used):                                           │
│   ┌────────────────────┐                                            │
│   │ Not executed,      │                                            │
│   │ not rendered       │                                            │
│   └────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Essential Keyboard Shortcuts

Master these shortcuts and you'll be 3x faster:

**Command mode** (press `Esc` first):

| Shortcut | Action | When to Use |
|----------|--------|-------------|
| `A` | Insert cell above | Adding setup code |
| `B` | Insert cell below | Continuing work |
| `D D` | Delete cell | Cleaning up |
| `M` | Change to Markdown | Documentation |
| `Y` | Change to Code | Back to code |
| `Shift + Enter` | Run and move to next | Normal execution |
| `Ctrl + Enter` | Run and stay | Iterating on same cell |
| `Z` | Undo cell delete | Accidents happen |
| `L` | Toggle line numbers | Debugging |
| `O` | Toggle output | Hide long outputs |
| `Shift + M` | Merge selected cells | Combining cells |

**Edit mode** (press `Enter` to enter):

| Shortcut | Action | When to Use |
|----------|--------|-------------|
| `Tab` | Autocomplete | Finding methods |
| `Shift + Tab` | Show docstring | Reading documentation |
| `Ctrl + /` | Comment/uncomment | Quick commenting |
| `Ctrl + Shift + -` | Split cell at cursor | Breaking up long cells |
| `Ctrl + ]` | Indent | Formatting |
| `Ctrl + [` | Dedent | Formatting |

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

# Rich output
from IPython.display import HTML, Markdown
display(Markdown("**Bold text** in output"))
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

# Simple wall-clock time (single run)
%time result = expensive_function()
# Output: CPU times: user 1.23 s, sys: 456 ms, total: 1.69 s
#         Wall time: 1.72 s
```

| Magic | Purpose | When to Use |
|-------|---------|-------------|
| `%timeit` | Benchmark (many runs) | Comparing implementations |
| `%time` | Single timing | Long-running code |
| `%%timeit` | Benchmark whole cell | Multi-line benchmarks |

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

### The Most Useful Magics

```python
# Show all available magics
%lsmagic

# Matplotlib inline (usually automatic now)
%matplotlib inline

# Auto-reload modules when they change (ESSENTIAL for development)
%load_ext autoreload
%autoreload 2
# Now changes to my_module.py are automatically reloaded
import my_module

# Show variables
%who      # List variable names
%whos     # List with details (type, size)
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

### Magic Commands Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Magic Command Quick Reference                   │
│                                                                     │
│   Timing:           %time, %timeit, %%timeit                        │
│   Shell:            !command, !!command (capture output)            │
│   Variables:        %who, %whos, %reset                             │
│   Files:            %run, %load, %%writefile                        │
│   Debugging:        %debug, %pdb                                    │
│   Development:      %autoreload                                     │
│   Environment:      %env, %pwd, %cd                                 │
│   Display:          %matplotlib inline                              │
│                                                                     │
│   Line magic:  %magic (affects one line)                            │
│   Cell magic:  %%magic (affects whole cell)                         │
└─────────────────────────────────────────────────────────────────────┘
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
# h - help
```

> **Tip:** After an error, type `%debug` in a new cell to drop into the debugger at the point of failure. This is incredibly useful for understanding what went wrong.

---

## 📚 Part 4: Best Practices

### Notebook Structure

A well-organized notebook tells a story:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Ideal Notebook Structure                        │
│                                                                     │
│   # Project Title                                                   │
│   Brief description of what this notebook does.                     │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 1. Setup                                                       │
│   - Imports                                                         │
│   - Configuration                                                   │
│   - Random seeds                                                    │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 2. Data Loading                                                │
│   - Load data                                                       │
│   - Initial inspection                                              │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 3. Exploration / Analysis                                      │
│   - EDA                                                             │
│   - Preprocessing                                                   │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 4. Modeling                                                    │
│   - Model definition                                                │
│   - Training                                                        │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 5. Results                                                     │
│   - Evaluation                                                      │
│   - Visualizations                                                  │
│   ─────────────────────────────────────────────                     │
│                                                                     │
│   ## 6. Conclusions                                                 │
│   What did we learn?                                                │
└─────────────────────────────────────────────────────────────────────┘
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

### The Golden Rule: Restart & Run All

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE GOLDEN RULE                                 │
│                                                                     │
│   Before sharing or trusting results:                               │
│                                                                     │
│        Kernel → Restart & Run All                                   │
│                                                                     │
│   If it doesn't work, your notebook has hidden state.               │
│                                                                     │
│   Common causes:                                                    │
│   - Ran cells out of order                                          │
│   - Deleted cells that defined variables                            │
│   - Modified variables interactively                                │
│   - Imported then modified modules                                  │
│                                                                     │
│   A notebook that fails "Restart & Run All" is unreproducible.      │
└─────────────────────────────────────────────────────────────────────┘
```

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

**Benefits of small cells:**
- Easier to debug (find which cell failed)
- Faster iteration (re-run just what changed)
- Better documentation (each cell has clear purpose)
- Partial execution (run expensive cells once, iterate on later cells)

### Use Markdown Liberally

```markdown
## Data Preprocessing

We apply the following transformations:
1. Remove null values (found 23 rows with nulls)
2. Normalize features to [0, 1] range
3. Encode categorical variables

**Note:** The `category` column has 15 unique values.

### Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Baseline | 0.72 | 0.68 |
| MLP | 0.85 | 0.82 |
```

---

## 📚 Part 5: Colab-Specific Tips

### Check GPU

```python
# Check if GPU is available
!nvidia-smi

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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

# Install specific version
!pip install torch==2.0.0

# Install from git
!pip install git+https://github.com/user/repo.git
```

### Upload/Download Files

```python
from google.colab import files

# Upload
uploaded = files.upload()

# Download
files.download('results.csv')
```

### Colab Tips

| Tip | How |
|-----|-----|
| Keep session alive | Keep browser tab active |
| Save checkpoints | Save to Drive frequently |
| Use GPU runtime | Runtime → Change runtime type |
| Check GPU memory | `!nvidia-smi` |
| Install deps fast | Add `!pip install` to first cell |

---

## 📚 Part 6: When to Use .py Files

Notebooks are great for experimentation, but some code belongs in Python modules.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Notebooks vs .py Files                              │
│                                                                     │
│   Use Notebooks for:              Use .py Files for:                │
│   ──────────────────              ──────────────────                │
│   • Exploration & EDA             • Reusable functions              │
│   • Prototyping                   • Classes & modules               │
│   • Visualization                 • Production code                 │
│   • Teaching & sharing            • Tested code                     │
│   • Quick experiments             • Code used across notebooks      │
│   • One-off analyses              • Version-controlled logic        │
│                                                                     │
│   Rule of thumb: If you copy-paste code between notebooks,          │
│   put it in a .py file.                                             │
└─────────────────────────────────────────────────────────────────────┘
```

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

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hidden State Problem                            │
│                                                                     │
│   Cell 1 (run first):     x = 10                                    │
│                                  ↓                                  │
│   Cell 2 (run second):    y = x * 2    # y = 20                     │
│                                  ↓                                  │
│   Cell 1 (run again!):    x = 5        # Changed x                  │
│                                  ↓                                  │
│   Cell 3 (run):           print(y)     # Still 20! Not 10!          │
│                                                                     │
│   The notebook "remembers" the old value of y even though           │
│   x changed. This is hidden state.                                  │
│                                                                     │
│   Solution: Restart & Run All to catch these issues.                │
└─────────────────────────────────────────────────────────────────────┘
```

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

# In Colab/Jupyter
!free -h  # Linux
```

### Output Overload

```python
# Bad: Printing 10000 items
for item in large_list:
    print(item)  # Notebook becomes unresponsive

# Good: Sample or summarize
print(large_list[:10])  # First 10
print(f"Total: {len(large_list)}")

# Or use head/tail patterns
import pandas as pd
df.head(10)  # First 10 rows
```

### The Cell Execution Trap

```python
# Cell 1
model = train_model(data)  # Takes 10 minutes
# Out: Model trained!

# Cell 2 - you run this many times
evaluate(model)

# Later you change something in Cell 1 and forget to re-run it
# Cell 2 is using the OLD model!
```

> **Tip:** After modifying an early cell, re-run all cells below it. Or better: Restart & Run All.

---

## 📚 Part 8: Productivity Tips

### Quick Navigation

```python
# Use Ctrl+Shift+P (or Cmd+Shift+P) for command palette
# Search for any command by name

# Use Table of Contents
# JupyterLab: Left sidebar → Table of Contents
# VS Code: Outline view

# Use search
# Ctrl+F in current cell
# Ctrl+Shift+F to search all cells (JupyterLab)
```

### Snippets for Common Tasks

```python
# At the top of every notebook:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn-whitegrid')

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
```

### Progress Bars

```python
from tqdm.notebook import tqdm

# In a loop
for i in tqdm(range(1000)):
    # work
    pass

# With description
for epoch in tqdm(range(10), desc='Training'):
    for batch in tqdm(dataloader, desc='Batches', leave=False):
        # training step
        pass
```

### Interactive Widgets

```python
from ipywidgets import interact, widgets

@interact(x=(0, 10, 0.1))
def plot_sine(x=1):
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(t, np.sin(x * t))
    plt.show()

# Dropdown
@interact(model=['Linear', 'MLP', 'CNN'])
def show_results(model):
    print(f"Results for {model}")
```

### Profiling

```python
# Line profiler (install: pip install line_profiler)
%load_ext line_profiler
%lprun -f my_function my_function(args)

# Memory profiler (install: pip install memory_profiler)
%load_ext memory_profiler
%memit my_function(args)

# Built-in profiler
import cProfile
cProfile.run('my_function()')
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
import numpy as np

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

## 🎯 Key Takeaways

1. **"Restart & Run All" is the test.** If your notebook doesn't work after restart, it has hidden state and isn't reproducible.

2. **Set random seeds at the top.** Always. Every notebook. `np.random.seed(42)`, `torch.manual_seed(42)`, etc.

3. **Keep cells small and focused.** One logical step per cell. Easier to debug, iterate, and understand.

4. **Use `%autoreload` for development.** When working on .py modules alongside notebooks, autoreload picks up changes automatically.

5. **Magic commands save time.** `%timeit` for benchmarking, `%debug` for debugging, `%whos` for inspecting state.

6. **Document with Markdown.** Your future self will thank you. Explain *why*, not just *what*.

7. **Move reusable code to .py files.** If you copy-paste between notebooks, refactor into a module.

---

## 🔗 What's Next?

You now have a solid workflow for ML experimentation. Move on to **Module 5: Reading ML Papers** — the skill that unlocks deeper understanding of everything you'll learn.

---

## 📖 References

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [IPython Documentation](https://ipython.readthedocs.io/)
- "Ten Simple Rules for Reproducible Research in Jupyter Notebooks"
