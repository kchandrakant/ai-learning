# Module 3: Matplotlib Visualization

Visualization is how you understand your data and debug your models. If you can't see what's happening — data distributions, loss curves, prediction patterns — you're flying blind. This module covers the Matplotlib patterns you'll use constantly in ML.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Create line plots, scatter plots, and histograms
- Build multi-panel figures with subplots
- Customize plots (labels, legends, colors, styles)
- Save publication-quality figures
- Create quick diagnostic plots for debugging

---

## 📚 Part 1: The Basics

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

**Why this pattern?**
- More explicit — you know exactly what you're modifying
- Required for multiple subplots
- Better for complex figures
- Consistent with ML plotting code you'll read

---

## 📚 Part 2: Common Plot Types

### Line Plots (Training Curves)

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

### Scatter Plots (Data Visualization)

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

### Histograms (Distributions)

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
        text = ax.text(j, i, confusion[i, j], ha='center', va='center')

fig.colorbar(im)
plt.show()
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
epochs = np.arange(1, 101)
train_loss = 2.0 * np.exp(-0.05 * epochs) + np.random.randn(100) * 0.03
val_loss = 2.2 * np.exp(-0.04 * epochs) + np.random.randn(100) * 0.05
train_acc = 1 - train_loss / 3
val_acc = 1 - val_loss / 3

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

**Key parameters:**
- `dpi`: Resolution (150-300 for print)
- `bbox_inches='tight'`: Crop whitespace
- `transparent=True`: Transparent background

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

### Prediction vs Actual

```python
def plot_predictions(y_true, y_pred, title='Predictions'):
    """Scatter plot of predictions vs actuals."""
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

# Usage
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100) * 0.2
plot_predictions(y_true, y_pred)
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

Seaborn is great for exploration, but Matplotlib gives you more control for publication figures.

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

## 🔗 What's Next?

You can now visualize data and training progress. Move on to **Module 4: Jupyter Workflow** to learn the environment where you'll do most of your ML experimentation.

---

## 📖 References

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Cheatsheets](https://matplotlib.org/cheatsheets/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- "Fundamentals of Data Visualization" by Claus Wilke (free online)
