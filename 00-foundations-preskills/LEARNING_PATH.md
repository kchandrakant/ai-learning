# Foundations & Pre-Skills: A Step-by-Step Learning Journey

This prerequisite course prepares you for the AI/ML learning path. It covers Python programming patterns used in ML code, NumPy for numerical computing, visualization basics, and — critically — how to read ML research papers.

---

## 🎯 Why This Course?

Every course in this learning path assumes you can:
- **Write Python fluently** — Not just syntax, but the patterns (list comprehensions, generators, classes) that appear in ML codebases
- **Think in arrays** — ML is matrix math. NumPy fluency is non-negotiable
- **Visualize data** — Plotting distributions, loss curves, and results
- **Read papers** — Every course references research papers. Reading them efficiently is a skill

If you can already do these things, take the self-assessment at the end. If you pass, skip to Course 01.

---

## 🎯 Prerequisites

- Basic programming experience in any language
- High school algebra
- Willingness to practice

---

## 📚 Part 1: Python for ML

The Python patterns you'll see repeatedly in ML code.

### Module 1: Python Essentials
**Why it matters:** ML code uses specific Python patterns heavily. Knowing them makes reading and writing ML code much easier.

**What we'll cover:**
- Functions: args, kwargs, default values
- List comprehensions and generator expressions
- Classes and object-oriented patterns
- Context managers (`with` statements)
- Error handling and debugging
- Type hints (increasingly common in ML libraries)

**Key patterns you'll see everywhere:**
```python
# List comprehension
embeddings = [model.encode(text) for text in documents]

# Generator for memory efficiency
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Class-based models (PyTorch pattern)
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer(x)
```

---

### Module 2: NumPy Fundamentals
**Why it matters:** Neural networks are matrix operations. NumPy is how you work with matrices in Python. PyTorch and TensorFlow are built on the same concepts.

**What we'll cover:**
- Array creation and data types
- Indexing and slicing (including fancy indexing)
- Broadcasting rules
- Vectorization (why loops are slow)
- Reshaping, stacking, splitting
- Linear algebra operations
- Random number generation

**Key insight:**
```python
# Slow (Python loop)
result = []
for i in range(len(a)):
    result.append(a[i] * b[i])

# Fast (vectorized)
result = a * b  # NumPy does this in C, 100x faster
```

**The broadcasting rules you must know:**
```python
# These work:
(3, 4) + (4,)     → (3, 4)  # (4,) broadcasts to each row
(3, 4) + (3, 1)   → (3, 4)  # (3, 1) broadcasts across columns
(1, 4) + (3, 1)   → (3, 4)  # Both broadcast

# This fails:
(3, 4) + (3,)     → Error!  # Dimensions don't align
```

---

### Module 3: Matplotlib Visualization
**Why it matters:** You need to see your data and training progress. Visualization is how you debug ML.

**What we'll cover:**
- Figure and axes basics
- Line plots, scatter plots, histograms
- Subplots and layouts
- Customization (labels, legends, colors)
- Saving figures
- Quick plots for debugging

**Patterns you'll use constantly:**
```python
# Training loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()

# Data distribution
plt.hist(data, bins=50)
plt.show()

# Multiple subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(train_loss)
axes[1].plot(val_loss)
axes[2].scatter(predictions, actuals)
plt.tight_layout()
plt.show()
```

---

### Module 4: Jupyter Workflow
**Why it matters:** Jupyter notebooks are the standard environment for ML experimentation. Knowing the workflow saves time.

**What we'll cover:**
- Jupyter vs JupyterLab vs Google Colab
- Notebook best practices
- Magic commands (`%timeit`, `%matplotlib inline`, etc.)
- Debugging in notebooks
- Reproducibility tips
- When to move code to `.py` files

**Key tips:**
```python
# Time your code
%timeit np.dot(a, b)

# Autoreload modules during development
%load_ext autoreload
%autoreload 2

# Show plots inline
%matplotlib inline

# Check GPU (in Colab)
!nvidia-smi
```

---

## 📚 Part 2: Reading & Following ML Research

Skills for engaging with the research that underpins every course.

### Module 5: Reading ML Papers
**Why it matters:** Every course references papers. "Attention Is All You Need," "BERT," "GPT-3" — you'll see these names constantly. Reading papers efficiently unlocks deeper understanding.

**What we'll cover:**
- Anatomy of an ML paper
- Three-pass reading strategy
- Extracting key ideas without getting lost in math
- Reading math notation (the essentials)
- Guided reading of foundational papers
- Building a paper reading habit

**Paper structure (most ML papers):**
```
Abstract        → The elevator pitch (read first)
Introduction    → The problem and why it matters
Related Work    → Context (skim or skip initially)
Method          → The actual contribution (read carefully)
Experiments     → Does it work? (focus on tables/figures)
Conclusion      → Summary and limitations
```

**Three-pass reading strategy:**

| Pass | Time | Focus | Goal |
|------|------|-------|------|
| First | 10 min | Abstract → Figures → Conclusion | What problem? What solution? Does it work? |
| Second | 30 min | Introduction → Method (skim math) | What's the key idea? What's novel? |
| Third | 1-2 hrs | Deep dive into math, code, reproduce | Full understanding (only for important papers) |

**Math notation essentials:**
```
∈        means "element of" (x ∈ ℝ means x is a real number)
∀        means "for all"
∑        summation
∏        product
argmax   the input that maximizes a function
||x||    norm (length) of vector x
x̂        estimate/prediction of x
θ        model parameters (convention)
∇        gradient
```

---

### Module 6: Staying Current
**Why it matters:** ML moves fast. Knowing how to follow the field prevents you from learning outdated techniques.

**What we'll cover:**
- Where papers are published (arXiv, conferences)
- Tools for finding papers (Semantic Scholar, Connected Papers)
- Newsletters and digests
- Key people to follow
- Filtering signal from noise
- Building a sustainable reading habit

**The landscape:**
```
arXiv cs.LG, cs.CL    → Daily preprints (unreviewed)
NeurIPS, ICML, ICLR   → Top ML conferences
ACL, EMNLP, NAACL     → NLP conferences
CVPR, ICCV, ECCV      → Computer vision conferences
```

**Recommended resources:**
```
Newsletters:
- The Batch (Andrew Ng)
- ImportAI
- Papers With Code newsletter

Tools:
- Semantic Scholar → Find papers, see citations
- Connected Papers → Visualize paper relationships
- Papers With Code → Papers + implementations

Habit suggestion:
- 1 paper per week (deep read)
- 5 abstracts per week (skim)
```

---

## 🧪 Self-Assessment

Complete these exercises to confirm you're ready for Course 01.

### Python Assessment
1. Write a function that takes a list of dictionaries and returns only those where a specific key's value exceeds a threshold
2. Implement a simple class with `__init__`, `__repr__`, and one method
3. Use a generator to yield batches from a large list

### NumPy Assessment
1. Create a 5x5 matrix of random numbers and normalize each row to sum to 1
2. Given two arrays of shape (100, 768), compute the cosine similarity between all pairs
3. Explain why this fails: `np.array([[1,2,3]]) + np.array([1,2])`

### Visualization Assessment
1. Plot a sine wave and cosine wave on the same axes with a legend
2. Create a 2x2 grid of subplots showing different views of a dataset

### Paper Reading Assessment
1. Read the abstract of "Attention Is All You Need"
2. In 3 sentences, explain: What problem does it solve? What's the key idea? Why does it matter?

**If you can complete these comfortably, proceed to Course 01: ML Foundations.**

---

## 🗂️ Project Structure

```
00-foundations-preskills/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Dependencies
├── verify_setup.py           # Environment verification
│
├── 01_python_essentials/
│   └── README.md
│
├── 02_numpy_fundamentals/
│   └── README.md
│
├── 03_matplotlib_visualization/
│   └── README.md
│
├── 04_jupyter_workflow/
│   └── README.md
│
├── 05_reading_ml_papers/
│   └── README.md
│
├── 06_staying_current/
│   └── README.md
│
├── self_assessment/
│   └── README.md             # Assessment exercises with solutions
│
└── beyond/
    └── README.md             # Additional resources
```

---

## 📅 Recommended Timeline

```
Day 1-2: Python Essentials
├── Review patterns
└── Practice exercises

Day 3-4: NumPy Fundamentals
├── Array operations
├── Broadcasting
└── Vectorization practice

Day 5: Visualization & Jupyter
├── Matplotlib basics
└── Notebook workflow

Day 6-7: Reading Papers
├── Learn the strategy
├── Guided reading of one paper
└── Practice with a second paper

Day 8: Self-Assessment
└── Complete all exercises
```

**Total: ~1 week at 2-3 hours/day**

---

## 🚀 Let's Begin!

Start with **Module 1: Python Essentials** — the patterns you'll see in every ML codebase.

---

## 📖 References

### Python
- "Fluent Python" by Luciano Ramalho (advanced patterns)
- Python official documentation

### NumPy
- NumPy User Guide (official)
- "From Python to NumPy" by Nicolas Rougier (free online)

### Visualization
- Matplotlib documentation
- "Fundamentals of Data Visualization" by Claus Wilke (free online)

### Paper Reading
- "How to Read a Paper" by S. Keshav
- Yannic Kilcher's YouTube (paper explanations)
- Two Minute Papers (summaries)
