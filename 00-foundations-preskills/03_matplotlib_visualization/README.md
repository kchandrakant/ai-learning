# Module 3: Matplotlib Visualization

Visualization is how you understand your data and debug your models. If you can't see what's happening — data distributions, loss curves, prediction patterns — you're flying blind. This module covers the Matplotlib patterns you'll use constantly in ML.

---

## 🎯 Why This Matters

When your model isn't training correctly, you need to **see** what's happening:

- Is the loss actually decreasing?
- Are the predictions making sense?
- Is the data distributed as expected?
- Are there outliers causing problems?

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Visualization as Debugging                          │
│                                                                     │
│   Problem: "My model accuracy is stuck at 50%"                      │
│                                                                     │
│   Without visualization:                                            │
│   → Guess: Maybe learning rate is wrong?                            │
│   → Guess: Maybe data is bad?                                       │
│   → Guess: Maybe model is too small?                                │
│                                                                     │
│   With visualization:                                               │
│   → Plot loss: "Loss is flat — gradients might be zero"             │
│   → Plot predictions: "All predictions are 0.5 — sigmoid saturation"│
│   → Plot data: "Classes are perfectly balanced, data looks fine"    │
│   → Diagnosis: Vanishing gradients, try different initialization    │
│                                                                     │
│   Visualization turns guessing into debugging.                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Part 1: The Basics

### Two Ways to Plot

Matplotlib has two interfaces. Understanding the difference saves confusion:

```
┌─────────────────────────────────────────────────────────────────────┐
│              pyplot (simple) vs Object-Oriented (flexible)          │
│                                                                     │
│   pyplot style:                    OO style:                        │
│   ─────────────                    ─────────                        │
│   plt.plot(x, y)                   fig, ax = plt.subplots()         │
│   plt.xlabel('x')                  ax.plot(x, y)                    │
│   plt.title('Title')               ax.set_xlabel('x')               │
│   plt.show()                       ax.set_title('Title')            │
│                                    plt.show()                       │
│                                                                     │
│   Use pyplot for:                  Use OO for:                      │
│   - Quick exploration              - Multiple subplots              │
│   - Single plots                   - Complex figures                │
│   - Interactive work               - Reproducible scripts           │
│                                                                     │
│   Recommendation: Learn OO style — it's what you'll see in ML code  │
└─────────────────────────────────────────────────────────────────────┘
```

### Your First Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

### The Figure/Axes Pattern (Recommended)

For more control, use the explicit Figure and Axes objects:

```python
# Create figure and axes
fig, ax = plt.subplots()

# Plot on the axes
ax.plot(x, y)

# Customize
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.set_title('Sine Wave')

plt.show()
```

> **Why this pattern?** It's explicit — you know exactly what you're modifying. It's required for multiple subplots. And it's what you'll see in ML codebases everywhere.

---

## 📚 Part 2: Common Plot Types

### When to Use Each Plot Type

| Plot Type | Use When | ML Example |
|-----------|----------|------------|
| **Line plot** | Showing trends over time/steps | Loss curves, learning rate schedules |
| **Scatter plot** | Showing relationships between variables | 2D embeddings, predictions vs actuals |
| **Histogram** | Showing distributions | Weight distributions, prediction confidence |
| **Bar chart** | Comparing categories | Model comparison, class frequencies |
| **Heatmap** | Showing 2D patterns | Confusion matrices, attention weights |
| **Box plot** | Comparing distributions across groups | Feature distributions by class |

---

### Line Plots (Training Curves)

The most common plot in ML — tracking loss and metrics over training.

```python
# Simulated training history
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-0.1 * epochs) + 0.1 + np.random.randn(50) * 0.05
val_loss = 2.2 * np.exp(-0.08 * epochs) + 0.15 + np.random.randn(50) * 0.08

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_loss, label='Train Loss', color='blue')
ax.plot(epochs, val_loss, label='Validation Loss', color='orange')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Progress')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

**Reading Training Curves — What to Look For:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Training Curve Diagnostics                          │
│                                                                     │
│   Healthy Training:           Overfitting:                          │
│   Loss                        Loss                                  │
│   │\                          │\                                    │
│   │ \  train                  │ \  train                            │
│   │  \_____                   │  \________                          │
│   │   \____  val              │    val                              │
│   │        ‾‾‾                │      /‾‾‾‾ (val goes UP)            │
│   └─────────── Epoch          └─────────── Epoch                    │
│   Both decrease together      Gap widens, val increases             │
│                                                                     │
│   Underfitting:               Unstable Training:                    │
│   Loss                        Loss                                  │
│   │                           │ /\/\/\                              │
│   │ _______________           │/      \/\/\                         │
│   │   (both flat)             │          \/\/                       │
│   │                           │                                     │
│   └─────────────── Epoch      └─────────────── Epoch                │
│   Neither improving           Wild oscillations                     │
│   → Need more capacity        → Learning rate too high              │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Scatter Plots (Data Visualization)

Essential for understanding data structure and model predictions.

```python
# 2D data with two classes
np.random.seed(42)
class_0 = np.random.randn(50, 2) + np.array([0, 0])
class_1 = np.random.randn(50, 2) + np.array([3, 3])

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.7)
ax.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.7)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Two-Class Dataset')
ax.legend()
ax.axis('equal')  # Equal scaling

plt.show()
```

**Predictions vs Actuals Plot — The Regression Diagnostic:**

```python
def plot_predictions(y_true, y_pred, title='Predictions vs Actuals'):
    """The most important plot for regression models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    
    plt.show()
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              Reading Predictions vs Actuals                         │
│                                                                     │
│   Good model:              Systematic bias:       High variance:    │
│   Pred                     Pred                   Pred              │
│   │      /                 │      /               │    · ·  /       │
│   │    ·/·                 │  ···/                │  ·   · /        │
│   │  ·/·                   │ ·· /                 │ ·  ·  /  ·      │
│   │ /···                   │·  /                  │/·    ·   ·      │
│   │/                       │  /  (below line)     │  ·  ·           │
│   └────── Actual           └────── Actual         └────── Actual    │
│   Points on line           Under-predicting       Wide scatter      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Histograms (Distributions)

Critical for understanding what your data and model outputs look like.

```python
# Model predictions vs actual
predictions = np.random.randn(1000) * 0.5 + 2.5
actuals = np.random.randn(1000) * 0.3 + 2.5

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(predictions, bins=30, alpha=0.7, label='Predictions')
ax.hist(actuals, bins=30, alpha=0.7, label='Actuals')

ax.set_xlabel('Value')
ax.set_ylabel('Count')
ax.set_title('Prediction Distribution')
ax.legend()

plt.show()
```

**Weight Distribution — Debugging Neural Networks:**

```python
def plot_weight_distributions(model):
    """Check for vanishing/exploding gradients by looking at weights."""
    fig, axes = plt.subplots(1, len(list(model.parameters())), figsize=(15, 4))
    
    for ax, (name, param) in zip(axes, model.named_parameters()):
        weights = param.data.cpu().numpy().flatten()
        ax.hist(weights, bins=50)
        ax.set_title(f'{name}\nmean={weights.mean():.3f}, std={weights.std():.3f}')
    
    plt.tight_layout()
    plt.show()
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              Weight Distribution Diagnostics                        │
│                                                                     │
│   Healthy:                 Too small (vanishing):  Too large:       │
│   Count                    Count                   Count            │
│   │    ┌──┐                │   ┌┐                  │               │
│   │   ┌┘  └┐               │   ││                  │┌┐          ┌┐│ │
│   │  ┌┘    └┐              │  ┌┘└┐                 ││└┐        ┌┘││ │
│   │ ┌┘      └┐             │ ┌┘  └┐                ││ └┐      ┌┘ ││ │
│   └─┴────────┴─ 0          └─┴────┴─ 0             └┴──┴──────┴──┴┘ │
│   Spread around 0          All near 0              Bimodal/extreme  │
│   → Good initialization    → Gradients vanishing   → Exploding      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Bar Charts (Comparisons)

```python
models = ['Baseline', 'MLP', 'CNN', 'Transformer']
accuracies = [0.72, 0.85, 0.91, 0.94]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(models, accuracies, color=['gray', 'blue', 'green', 'orange'])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.2f}', ha='center', va='bottom')

ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
ax.set_ylim(0, 1.0)

plt.show()
```

---

### Heatmaps (Confusion Matrices, Attention)

```python
# Confusion matrix
confusion = np.array([
    [45, 3, 2],
    [5, 42, 3],
    [2, 4, 44]
])
classes = ['Cat', 'Dog', 'Bird']

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(confusion, cmap='Blues')

# Add labels
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Add numbers in cells
for i in range(len(classes)):
    for j in range(len(classes)):
        color = 'white' if confusion[i, j] > confusion.max() / 2 else 'black'
        ax.text(j, i, confusion[i, j], ha='center', va='center', color=color)

fig.colorbar(im)
plt.show()
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              Reading Confusion Matrices                             │
│                                                                     │
│   Good classifier:                Poor classifier:                  │
│              Predicted                      Predicted               │
│            Cat Dog Bird                   Cat Dog Bird              │
│   Actual  ┌───┬───┬───┐         Actual   ┌───┬───┬───┐             │
│   Cat     │ 45│  3│  2│         Cat      │ 25│ 15│ 10│             │
│   Dog     │  5│ 42│  3│         Dog      │ 12│ 20│ 18│             │
│   Bird    │  2│  4│ 44│         Bird     │  8│ 17│ 25│             │
│           └───┴───┴───┘                  └───┴───┴───┘             │
│   Strong diagonal                Many off-diagonal errors           │
│   → Classes well-separated       → Classes confused                 │
│                                                                     │
│   Look for patterns: Which classes get confused with each other?    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Part 3: Subplots

### Multiple Panels

```python
# 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot on each axes
axes[0].plot(x, np.sin(x))
axes[0].set_title('Sine')

axes[1].plot(x, np.cos(x))
axes[1].set_title('Cosine')

axes[2].plot(x, np.tan(x))
axes[2].set_ylim(-5, 5)
axes[2].set_title('Tangent')

plt.tight_layout()  # Prevent overlap
plt.show()
```

### 2D Grid of Subplots

```python
# 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Access with [row, col] indexing
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Top Left')

axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Top Right')

axes[1, 0].hist(np.random.randn(1000), bins=30)
axes[1, 0].set_title('Bottom Left')

axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 5])
axes[1, 1].set_title('Bottom Right')

plt.tight_layout()
plt.show()
```

### Common ML Pattern: Train/Val Curves Side by Side

```python
def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    """Standard training history visualization."""
    epochs = np.arange(1, len(train_loss) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_loss, label='Train')
    axes[0].plot(epochs, val_loss, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, label='Train')
    axes[1].plot(epochs, val_acc, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

---

## 📚 Part 4: Customization

### Colors and Styles

```python
# Named colors
ax.plot(x, y, color='red')
ax.plot(x, y, color='steelblue')

# Hex colors
ax.plot(x, y, color='#FF5733')

# RGB tuple (0-1 range)
ax.plot(x, y, color=(0.2, 0.4, 0.8))

# Line styles
ax.plot(x, y, linestyle='-')   # Solid
ax.plot(x, y, linestyle='--')  # Dashed
ax.plot(x, y, linestyle=':')   # Dotted
ax.plot(x, y, linestyle='-.')  # Dash-dot

# Markers
ax.plot(x, y, marker='o')      # Circle
ax.plot(x, y, marker='s')      # Square
ax.plot(x, y, marker='^')      # Triangle

# Combined format string
ax.plot(x, y, 'r--o')  # Red, dashed, circle markers

# Line width and marker size
ax.plot(x, y, linewidth=2, markersize=8)
```

### Common Style Combinations

| Purpose | Style | Code |
|---------|-------|------|
| Training curve | Solid line | `ax.plot(x, y, '-')` |
| Validation curve | Dashed line | `ax.plot(x, y, '--')` |
| Data points | Markers only | `ax.scatter(x, y)` |
| Threshold/reference | Thin dashed gray | `ax.axhline(y, color='gray', linestyle='--', alpha=0.5)` |
| Best result | Highlighted marker | `ax.scatter([x_best], [y_best], s=100, color='red', zorder=5)` |

### Legends

```python
fig, ax = plt.subplots()

ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# Basic legend
ax.legend()

# Legend with location
ax.legend(loc='upper right')
ax.legend(loc='lower left')
ax.legend(loc='best')  # Auto-choose

# Legend outside plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Multiple columns
ax.legend(ncol=2)
```

### Axis Limits and Scales

```python
# Set limits
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# Log scale (common for loss curves)
ax.set_yscale('log')

# Equal aspect ratio
ax.set_aspect('equal')
ax.axis('equal')

# Remove axis
ax.axis('off')
```

### Annotations

```python
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))

# Add text
ax.text(5, 0.5, 'Peak region', fontsize=12)

# Add arrow annotation
ax.annotate('Maximum', xy=(np.pi/2, 1), xytext=(3, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10)

# Vertical/horizontal lines
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)

plt.show()
```

---

## 📚 Part 5: Saving Figures

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x))
ax.set_title('Sine Wave')

# Save as PNG (raster)
fig.savefig('figure.png', dpi=150, bbox_inches='tight')

# Save as PDF (vector — good for papers)
fig.savefig('figure.pdf', bbox_inches='tight')

# Save as SVG (vector — good for web)
fig.savefig('figure.svg', bbox_inches='tight')

# With transparent background
fig.savefig('figure.png', transparent=True, dpi=150, bbox_inches='tight')
```

| Format | Type | Best For | Typical DPI |
|--------|------|----------|-------------|
| PNG | Raster | Screenshots, web | 150-300 |
| PDF | Vector | Papers, presentations | N/A |
| SVG | Vector | Web, editing | N/A |
| JPEG | Raster | Photos (not plots!) | 150-300 |

> **Tip:** Always use `bbox_inches='tight'` to crop whitespace. Use `dpi=300` for print-quality figures.

---

## 📚 Part 6: Quick Debugging Plots

When training models, you need fast visualization. Here are patterns for quick debugging:

### Quick Distribution Check

```python
def quick_hist(data, title='Distribution'):
    """Quick histogram for debugging."""
    plt.figure(figsize=(8, 4))
    plt.hist(data.flatten(), bins=50)
    plt.title(f'{title}\nmean={data.mean():.3f}, std={data.std():.3f}')
    plt.show()

# Usage
weights = np.random.randn(1000, 100)
quick_hist(weights, 'Weight Distribution')
```

### Quick Loss Plot

```python
def plot_loss(losses, title='Training Loss'):
    """Quick loss curve."""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# Usage during training
losses = []
for step in range(1000):
    loss = train_step()  # Your training logic
    losses.append(loss)
    
    if step % 100 == 0:
        plot_loss(losses)
```

### Quick Image Grid

```python
def show_images(images, titles=None, cols=4):
    """Display a grid of images."""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    
    # Hide empty subplots
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
images = [np.random.rand(28, 28) for _ in range(8)]
show_images(images)
```

### The Four-Panel Debug Dashboard

```python
def plot_training_debug(losses, accuracies, predictions, targets):
    """Complete training step debug visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_title(f'Loss (current: {losses[-1]:.4f})')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[0, 1].plot(accuracies)
    axes[0, 1].set_title(f'Accuracy (current: {accuracies[-1]:.4f})')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction histogram
    axes[1, 0].hist(predictions.flatten(), bins=30, alpha=0.7)
    axes[1, 0].set_title(f'Predictions (mean: {predictions.mean():.3f})')
    axes[1, 0].set_xlabel('Prediction')
    axes[1, 0].set_ylabel('Count')
    
    # Predictions vs targets
    axes[1, 1].scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
    lims = [min(targets.min(), predictions.min()), 
            max(targets.max(), predictions.max())]
    axes[1, 1].plot(lims, lims, 'r--')
    axes[1, 1].set_title('Predictions vs Targets')
    axes[1, 1].set_xlabel('Target')
    axes[1, 1].set_ylabel('Prediction')
    
    plt.tight_layout()
    plt.show()
```

---

## 📚 Part 7: Seaborn for Statistical Plots

Seaborn builds on Matplotlib for statistical visualization:

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plot
sns.histplot(data, kde=True)

# Box plot
sns.boxplot(x='category', y='value', data=df)

# Heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Pair plot (scatter matrix)
sns.pairplot(df, hue='class')
```

### When to Use Seaborn vs Matplotlib

| Task | Seaborn | Matplotlib |
|------|---------|------------|
| Quick EDA | ✓ | |
| Statistical plots | ✓ | |
| Pair plots | ✓ | |
| Full control | | ✓ |
| Custom layouts | | ✓ |
| Publication figures | | ✓ |
| Subplots | | ✓ |

> **Tip:** Seaborn is great for exploration, but Matplotlib gives you more control for publication figures.

---

## 📚 Part 8: Common Pitfalls and Fixes

### Pitfall 1: Overlapping Labels

```python
# Problem
fig, ax = plt.subplots()
ax.bar(range(10), range(10))
ax.set_xticklabels(['Very Long Label ' + str(i) for i in range(10)])
# Labels overlap!

# Fix: Rotate labels
ax.set_xticklabels(['Very Long Label ' + str(i) for i in range(10)], 
                   rotation=45, ha='right')
plt.tight_layout()
```

### Pitfall 2: Color Scale Issues

```python
# Problem: Linear scale hides patterns in skewed data
ax.imshow(data_with_outliers)  # Outliers dominate color scale

# Fix 1: Log scale
ax.imshow(np.log1p(data_with_outliers))

# Fix 2: Clip outliers
vmin, vmax = np.percentile(data_with_outliers, [5, 95])
ax.imshow(data_with_outliers, vmin=vmin, vmax=vmax)
```

### Pitfall 3: Misleading Axis

```python
# Problem: Y-axis doesn't start at 0 (exaggerates differences)
values = [98, 99, 100, 101, 102]
ax.plot(values)  # Looks like huge variation!

# Fix: Either start at 0 or be explicit
ax.set_ylim(0, 110)  # Start at 0
# Or add a break indicator and note in caption
```

### Pitfall 4: Too Many Colors

```python
# Problem: 10 lines with default colors — hard to distinguish
for i in range(10):
    ax.plot(data[i])

# Fix 1: Use line styles
styles = ['-', '--', ':', '-.']
for i in range(10):
    ax.plot(data[i], linestyle=styles[i % 4])

# Fix 2: Use a colormap
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for i, c in enumerate(colors):
    ax.plot(data[i], color=c)
```

---

## 🏋️ Exercises

### Exercise 1: Training Curves
```python
# Create a figure with loss and accuracy curves
# - Left panel: Train and validation loss (log scale y-axis)
# - Right panel: Train and validation accuracy
# - Add grid, legend, titles, and labels

epochs = np.arange(1, 101)
train_loss = 2.5 * np.exp(-0.05 * epochs) + 0.1
val_loss = 2.8 * np.exp(-0.04 * epochs) + 0.15
train_acc = 1 - 0.5 * np.exp(-0.05 * epochs)
val_acc = 1 - 0.6 * np.exp(-0.04 * epochs)

# Your code here
```

### Exercise 2: Confusion Matrix
```python
# Create a proper confusion matrix visualization
# - Use imshow with a good colormap
# - Add class labels on axes
# - Add numbers in each cell
# - Add colorbar

confusion = np.array([
    [85, 10, 5, 0],
    [8, 78, 12, 2],
    [3, 15, 75, 7],
    [1, 2, 8, 89]
])
classes = ['Cat', 'Dog', 'Bird', 'Fish']

# Your code here
```

### Exercise 3: Multi-panel Figure
```python
# Create a 2x2 figure showing:
# - Top left: Histogram of normally distributed data
# - Top right: Scatter plot of 2D data with two classes
# - Bottom left: Line plot of sin and cos
# - Bottom right: Bar chart comparing 4 values

# Your code here
```

### Exercise 4: Debugging Helper
```python
# Write a function `plot_training_step` that takes:
# - losses: list of loss values so far
# - accuracies: list of accuracy values so far
# - predictions: current batch predictions (1D array)
# - targets: current batch targets (1D array)
#
# And creates a 2x2 figure showing:
# - Loss curve
# - Accuracy curve  
# - Histogram of predictions
# - Scatter of predictions vs targets

def plot_training_step(losses, accuracies, predictions, targets):
    # Your code here
    pass
```

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 1: Training Curves
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss (log scale)
axes[0].plot(epochs, train_loss, label='Train')
axes[0].plot(epochs, val_loss, label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, train_acc, label='Train')
axes[1].plot(epochs, val_acc, label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Exercise 2: Confusion Matrix
```python
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(confusion, cmap='Blues')

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Add numbers
for i in range(len(classes)):
    for j in range(len(classes)):
        color = 'white' if confusion[i, j] > confusion.max() / 2 else 'black'
        ax.text(j, i, confusion[i, j], ha='center', va='center', color=color)

fig.colorbar(im)
plt.tight_layout()
plt.show()
```

### Exercise 3: Multi-panel Figure
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: Histogram
data = np.random.randn(1000)
axes[0, 0].hist(data, bins=30, edgecolor='black')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Count')

# Top right: Scatter with classes
class_0 = np.random.randn(50, 2)
class_1 = np.random.randn(50, 2) + 2
axes[0, 1].scatter(class_0[:, 0], class_0[:, 1], label='Class 0')
axes[0, 1].scatter(class_1[:, 0], class_1[:, 1], label='Class 1')
axes[0, 1].set_title('2D Classification Data')
axes[0, 1].legend()

# Bottom left: Sin and cos
x = np.linspace(0, 2*np.pi, 100)
axes[1, 0].plot(x, np.sin(x), label='sin')
axes[1, 0].plot(x, np.cos(x), label='cos')
axes[1, 0].set_title('Trigonometric Functions')
axes[1, 0].legend()

# Bottom right: Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values)
axes[1, 1].set_title('Category Comparison')

plt.tight_layout()
plt.show()
```

### Exercise 4: Debugging Helper
```python
def plot_training_step(losses, accuracies, predictions, targets):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_title(f'Loss (current: {losses[-1]:.4f})')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[0, 1].plot(accuracies)
    axes[0, 1].set_title(f'Accuracy (current: {accuracies[-1]:.4f})')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction histogram
    axes[1, 0].hist(predictions, bins=30, alpha=0.7)
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].set_xlabel('Prediction')
    axes[1, 0].set_ylabel('Count')
    
    # Predictions vs targets
    axes[1, 1].scatter(targets, predictions, alpha=0.5)
    lims = [min(targets.min(), predictions.min()), 
            max(targets.max(), predictions.max())]
    axes[1, 1].plot(lims, lims, 'r--')
    axes[1, 1].set_title('Predictions vs Targets')
    axes[1, 1].set_xlabel('Target')
    axes[1, 1].set_ylabel('Prediction')
    
    plt.tight_layout()
    plt.show()
```

</details>

---

## 🎯 Key Takeaways

1. **Visualization is debugging.** When training goes wrong, the first step is always "plot it" — loss curves, predictions, weight distributions.

2. **Use the OO interface** (`fig, ax = plt.subplots()`) — it's more explicit, works for complex figures, and matches what you'll see in ML codebases.

3. **Training curves tell a story.** Learn to read overfitting (val loss going up), underfitting (both flat), and instability (oscillations).

4. **Predictions vs actuals** is the most important regression diagnostic. Points should cluster around the y=x line.

5. **Confusion matrices** reveal which classes are confused. Look at off-diagonal patterns.

6. **Build a library of quick-debug functions** — `quick_hist()`, `plot_loss()`, `show_images()`. You'll use them constantly.

7. **Save figures with `bbox_inches='tight'`** and appropriate DPI. Use PDF for papers, PNG for sharing.

---

## 🔗 What's Next?

You can now visualize data and training progress. Move on to **Module 4: Jupyter Workflow** to learn the environment where you'll do most of your ML experimentation.

---

## 📖 References

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Cheatsheets](https://matplotlib.org/cheatsheets/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- "Fundamentals of Data Visualization" by Claus Wilke (free online)
