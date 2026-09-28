"""
Module 3: Matplotlib Visualization - Practice Exercises
========================================================

Visualization is how you understand data and debug ML models.
This module builds fluency with the plots you'll use constantly:
- Training curves
- Data distributions
- Confusion matrices
- Debugging visualizations

Usage:
    python exercises.py           # Run all exercises (saves images)
    python exercises.py --show    # Display plots interactively
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Create output directory for saved figures
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Check if we should show plots interactively
SHOW_PLOTS = '--show' in sys.argv

def save_or_show(fig, filename):
    """Save figure to file or show interactively."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# PART 1: BASIC PLOTS
# =============================================================================

print("\n" + "="*60)
print("PART 1: BASIC PLOTS")
print("="*60)

# Exercise 1.1: Simple Line Plot
print("\n--- Exercise 1.1: Simple Line Plot ---")

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color='blue', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.set_title('Sine Wave')
ax.grid(True, alpha=0.3)

save_or_show(fig, '01_line_plot.png')

# Exercise 1.2: Multiple Lines with Legend
print("\n--- Exercise 1.2: Multiple Lines ---")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), label='sin(x)', color='blue')
ax.plot(x, np.cos(x), label='cos(x)', color='orange')
ax.plot(x, np.sin(x) + np.cos(x), label='sin(x) + cos(x)', color='green', linestyle='--')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trigonometric Functions')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

save_or_show(fig, '02_multiple_lines.png')

# Exercise 1.3: Line Styles and Markers
print("\n--- Exercise 1.3: Line Styles ---")

x = np.linspace(0, 5, 20)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, x, 'b-', label='solid', linewidth=2)
ax.plot(x, x + 1, 'g--', label='dashed', linewidth=2)
ax.plot(x, x + 2, 'r:', label='dotted', linewidth=2)
ax.plot(x, x + 3, 'ko-', label='markers', markersize=5)
ax.plot(x, x + 4, 'ms--', label='square markers', markersize=5)

ax.legend()
ax.set_title('Line Styles and Markers')
ax.grid(True, alpha=0.3)

save_or_show(fig, '03_line_styles.png')


# =============================================================================
# PART 2: ML-ESSENTIAL PLOTS
# =============================================================================

print("\n" + "="*60)
print("PART 2: ML-ESSENTIAL PLOTS")
print("="*60)

# Exercise 2.1: Training Curves
print("\n--- Exercise 2.1: Training Curves ---")

# Simulate training history
epochs = np.arange(1, 101)
train_loss = 2.5 * np.exp(-0.05 * epochs) + 0.1 + np.random.randn(100) * 0.02
val_loss = 2.7 * np.exp(-0.04 * epochs) + 0.15 + np.random.randn(100) * 0.03
train_acc = 1 - 0.5 * np.exp(-0.05 * epochs) + np.random.randn(100) * 0.01
val_acc = 1 - 0.6 * np.exp(-0.04 * epochs) + np.random.randn(100) * 0.015

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(epochs, train_loss, label='Train Loss', color='blue')
axes[0].plot(epochs, val_loss, label='Val Loss', color='orange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(epochs, train_acc, label='Train Accuracy', color='blue')
axes[1].plot(epochs, val_acc, label='Val Accuracy', color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_or_show(fig, '04_training_curves.png')

# Exercise 2.2: Loss with Log Scale
print("\n--- Exercise 2.2: Log Scale Loss ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axes[0].plot(epochs, train_loss, label='Train')
axes[0].plot(epochs, val_loss, label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss (Linear Scale)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].plot(epochs, train_loss, label='Train')
axes[1].plot(epochs, val_loss, label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss (Log Scale)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_or_show(fig, '05_log_scale_loss.png')

# Exercise 2.3: Histograms - Data Distributions
print("\n--- Exercise 2.3: Data Distributions ---")

# Simulated model predictions
correct_preds = np.random.randn(500) * 0.2 + 0.8
incorrect_preds = np.random.randn(500) * 0.3 + 0.4

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Single histogram
all_preds = np.concatenate([correct_preds, incorrect_preds])
axes[0].hist(all_preds, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
axes[0].set_xlabel('Prediction Confidence')
axes[0].set_ylabel('Count')
axes[0].set_title('All Predictions')
axes[0].legend()

# Overlapping histograms
axes[1].hist(correct_preds, bins=30, alpha=0.7, label='Correct', color='green')
axes[1].hist(incorrect_preds, bins=30, alpha=0.7, label='Incorrect', color='red')
axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[1].set_xlabel('Prediction Confidence')
axes[1].set_ylabel('Count')
axes[1].set_title('Predictions by Correctness')
axes[1].legend()

plt.tight_layout()
save_or_show(fig, '06_histograms.png')

# Exercise 2.4: Scatter Plot - 2D Data
print("\n--- Exercise 2.4: Scatter Plots ---")

# Generate 2-class data
class_0 = np.random.randn(100, 2) + np.array([0, 0])
class_1 = np.random.randn(100, 2) + np.array([3, 3])

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.7, s=50)
ax.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.7, s=50)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Two-Class Dataset')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)

save_or_show(fig, '07_scatter_plot.png')

# Exercise 2.5: Confusion Matrix
print("\n--- Exercise 2.5: Confusion Matrix ---")

def plot_confusion_matrix(cm, class_names, ax=None, cmap='Blues'):
    """Plot a confusion matrix with annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    im = ax.imshow(cm, cmap=cmap)
    
    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Add numbers in cells
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=12)
    
    fig.colorbar(im, ax=ax)
    return fig, ax

# Create confusion matrix
confusion = np.array([
    [85, 10, 5],
    [8, 78, 14],
    [3, 12, 85]
])
classes = ['Cat', 'Dog', 'Bird']

fig, ax = plt.subplots(figsize=(8, 6))
plot_confusion_matrix(confusion, classes, ax=ax)
ax.set_title('Classification Confusion Matrix')

plt.tight_layout()
save_or_show(fig, '08_confusion_matrix.png')

# Exercise 2.6: Heatmap - Attention Weights
print("\n--- Exercise 2.6: Attention Heatmap ---")

# Simulate attention weights
seq_len = 8
attention = np.random.rand(seq_len, seq_len)
# Make it more realistic (attend more to nearby tokens)
for i in range(seq_len):
    for j in range(seq_len):
        attention[i, j] *= np.exp(-0.3 * abs(i - j))
# Normalize rows
attention = attention / attention.sum(axis=1, keepdims=True)

tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'dog']

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(attention, cmap='viridis')

ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(tokens, rotation=45, ha='right')
ax.set_yticklabels(tokens)
ax.set_xlabel('Key')
ax.set_ylabel('Query')
ax.set_title('Attention Weights')

fig.colorbar(im, ax=ax, label='Attention Weight')

plt.tight_layout()
save_or_show(fig, '09_attention_heatmap.png')


# =============================================================================
# PART 3: SUBPLOTS AND LAYOUTS
# =============================================================================

print("\n" + "="*60)
print("PART 3: SUBPLOTS AND LAYOUTS")
print("="*60)

# Exercise 3.1: Grid of Subplots
print("\n--- Exercise 3.1: 2x2 Grid ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', label='sin')
axes[0, 0].plot(x, np.cos(x), 'r-', label='cos')
axes[0, 0].set_title('Trigonometric Functions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Top-right: Scatter
data1 = np.random.randn(50, 2)
data2 = np.random.randn(50, 2) + 2
axes[0, 1].scatter(data1[:, 0], data1[:, 1], label='Group A')
axes[0, 1].scatter(data2[:, 0], data2[:, 1], label='Group B')
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].legend()

# Bottom-left: Histogram
data = np.random.randn(1000)
axes[1, 0].hist(data, bins=30, edgecolor='black')
axes[1, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
axes[1, 0].set_title('Distribution')
axes[1, 0].legend()

# Bottom-right: Bar chart
categories = ['Model A', 'Model B', 'Model C', 'Model D']
accuracies = [0.82, 0.87, 0.91, 0.89]
colors = ['steelblue', 'steelblue', 'green', 'steelblue']
axes[1, 1].bar(categories, accuracies, color=colors)
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Model Comparison')
axes[1, 1].set_ylim(0.75, 0.95)

plt.tight_layout()
save_or_show(fig, '10_subplot_grid.png')

# Exercise 3.2: Different Subplot Sizes
print("\n--- Exercise 3.2: Mixed Subplot Sizes ---")

fig = plt.figure(figsize=(14, 8))

# Large plot on left
ax1 = fig.add_subplot(1, 2, 1)
epochs = np.arange(1, 51)
loss = 2 * np.exp(-0.1 * epochs) + np.random.randn(50) * 0.02
ax1.plot(epochs, loss, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Two small plots stacked on right
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(np.random.randn(500), bins=20, color='green', alpha=0.7)
ax2.set_title('Weight Distribution (Layer 1)')

ax3 = fig.add_subplot(2, 2, 4)
ax3.hist(np.random.randn(500), bins=20, color='orange', alpha=0.7)
ax3.set_title('Weight Distribution (Layer 2)')

plt.tight_layout()
save_or_show(fig, '11_mixed_subplots.png')


# =============================================================================
# PART 4: DEBUGGING VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("PART 4: DEBUGGING VISUALIZATIONS")
print("="*60)

# Exercise 4.1: Quick Distribution Check
print("\n--- Exercise 4.1: Quick Distribution Helper ---")

def quick_hist(data, title='Distribution', ax=None):
    """Quick histogram for debugging - shows basic stats."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    
    data = np.array(data).flatten()
    ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{title}\nmean={data.mean():.4f}, std={data.std():.4f}, min={data.min():.4f}, max={data.max():.4f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    
    return fig, ax

# Demo: Check weight distributions
weights = np.random.randn(1000, 256) * 0.02
fig, ax = quick_hist(weights, 'Neural Network Weights')
save_or_show(fig, '12_quick_hist.png')

# Exercise 4.2: Gradient Flow Visualization
print("\n--- Exercise 4.2: Gradient Flow ---")

def plot_gradient_flow(gradients_by_layer, layer_names=None):
    """
    Visualize gradient magnitudes across layers.
    Useful for debugging vanishing/exploding gradients.
    """
    n_layers = len(gradients_by_layer)
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(n_layers)]
    
    means = [np.abs(g).mean() for g in gradients_by_layer]
    stds = [np.abs(g).std() for g in gradients_by_layer]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(n_layers)
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow (mean ± std)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax

# Simulate gradient flow (healthy)
gradients = [np.random.randn(100) * (0.1 * 0.9**i) for i in range(10)]
layer_names = [f'Conv{i+1}' for i in range(5)] + [f'FC{i+1}' for i in range(5)]

fig, ax = plot_gradient_flow(gradients, layer_names)
ax.set_title('Gradient Flow (Healthy - slight decay is normal)')
save_or_show(fig, '13_gradient_flow.png')

# Exercise 4.3: Predictions vs Actuals
print("\n--- Exercise 4.3: Predictions vs Actuals ---")

def plot_predictions(y_true, y_pred, title='Predictions vs Actuals'):
    """Scatter plot comparing predictions to actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    
    # Compute R² score
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{title}\nR² = {r2:.4f}')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Demo
y_true = np.random.randn(200) * 2 + 5
y_pred = y_true + np.random.randn(200) * 0.5  # Add some noise

fig, ax = plot_predictions(y_true, y_pred, 'Regression Model Performance')
save_or_show(fig, '14_predictions_vs_actuals.png')

# Exercise 4.4: Training Dashboard
print("\n--- Exercise 4.4: Training Dashboard ---")

def plot_training_dashboard(history, predictions=None, targets=None):
    """
    Comprehensive training visualization dashboard.
    
    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        predictions: optional array of current predictions
        targets: optional array of targets
    """
    n_plots = 4 if predictions is not None else 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f"Loss (current: {history['train_loss'][-1]:.4f})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', color='blue')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f"Accuracy (current: {history['train_acc'][-1]:.4f})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if predictions is not None and targets is not None:
        # Prediction distribution
        axes[1, 0].hist(predictions.flatten(), bins=30, alpha=0.7, label='Predictions')
        axes[1, 0].hist(targets.flatten(), bins=30, alpha=0.7, label='Targets')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        
        # Residuals
        residuals = predictions - targets
        axes[1, 1].hist(residuals.flatten(), bins=30, edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Residual (Pred - Actual)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Residuals (mean: {residuals.mean():.4f})')
    else:
        # Learning rate decay visualization
        lr = 0.01 * np.exp(-0.02 * epochs)
        axes[1, 0].plot(epochs, lr, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gap between train and val (overfitting indicator)
        gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(epochs, gap, 'r-', linewidth=2)
        axes[1, 1].axhline(0, color='gray', linestyle='--')
        axes[1, 1].fill_between(epochs, 0, gap, alpha=0.3, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Gap')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# Demo training dashboard
history = {
    'train_loss': list(2.5 * np.exp(-0.05 * np.arange(50)) + np.random.randn(50) * 0.02),
    'val_loss': list(2.7 * np.exp(-0.04 * np.arange(50)) + np.random.randn(50) * 0.03),
    'train_acc': list(1 - 0.5 * np.exp(-0.05 * np.arange(50)) + np.random.randn(50) * 0.01),
    'val_acc': list(1 - 0.6 * np.exp(-0.04 * np.arange(50)) + np.random.randn(50) * 0.015),
}

fig, axes = plot_training_dashboard(history)
fig.suptitle('Training Dashboard', fontsize=14, fontweight='bold', y=1.02)
save_or_show(fig, '15_training_dashboard.png')


# =============================================================================
# PART 5: ADVANCED VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("PART 5: ADVANCED VISUALIZATIONS")
print("="*60)

# Exercise 5.1: Embedding Visualization (2D Projection)
print("\n--- Exercise 5.1: Embedding Visualization ---")

def plot_embeddings_2d(embeddings, labels, class_names=None, title='Embeddings'):
    """
    Plot 2D embeddings colored by class.
    
    In practice, you'd use t-SNE or UMAP to project high-dimensional
    embeddings to 2D first.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else f'Class {label}'
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                  c=[colors[i]], label=name, alpha=0.7, s=30)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Simulate clustered embeddings
n_per_class = 100
embeddings = np.vstack([
    np.random.randn(n_per_class, 2) * 0.5 + np.array([0, 0]),
    np.random.randn(n_per_class, 2) * 0.5 + np.array([3, 0]),
    np.random.randn(n_per_class, 2) * 0.5 + np.array([1.5, 2.5]),
])
labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

fig, ax = plot_embeddings_2d(embeddings, labels, ['Cat', 'Dog', 'Bird'])
save_or_show(fig, '16_embeddings.png')

# Exercise 5.2: Image Grid
print("\n--- Exercise 5.2: Image Grid ---")

def show_image_grid(images, titles=None, cols=4, figsize=None):
    """Display a grid of images."""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    if figsize is None:
        figsize = (3 * cols, 3 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            cmap = 'gray' if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
            if titles:
                ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes

# Generate random "images"
images = [np.random.rand(28, 28) for _ in range(8)]
titles = [f'Sample {i+1}' for i in range(8)]

fig, axes = show_image_grid(images, titles, cols=4)
save_or_show(fig, '17_image_grid.png')

# Exercise 5.3: Learning Rate Finder Visualization
print("\n--- Exercise 5.3: Learning Rate Finder ---")

def plot_lr_finder(lrs, losses, suggested_lr=None):
    """
    Plot learning rate finder results.
    
    The optimal learning rate is typically where the loss
    decreases fastest (steepest negative slope).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lrs, losses, 'b-', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Finder')
    ax.grid(True, alpha=0.3)
    
    if suggested_lr:
        ax.axvline(suggested_lr, color='red', linestyle='--', 
                  label=f'Suggested LR: {suggested_lr:.1e}')
        ax.legend()
    
    return fig, ax

# Simulate LR finder results
lrs = np.logspace(-6, 0, 100)
losses = 2.5 - 0.5 * np.log10(lrs + 1e-7) + np.exp(3 * (np.log10(lrs) + 2))
losses += np.random.randn(100) * 0.05
losses = np.clip(losses, 0.1, 10)

fig, ax = plot_lr_finder(lrs, losses, suggested_lr=1e-3)
save_or_show(fig, '18_lr_finder.png')


# =============================================================================
# PART 6: SAVING AND CUSTOMIZATION
# =============================================================================

print("\n" + "="*60)
print("PART 6: PUBLICATION-QUALITY FIGURES")
print("="*60)

# Exercise 6.1: Publication-Ready Figure
print("\n--- Exercise 6.1: Publication Figure ---")

# Use a cleaner style
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(8, 6))

# Data
models = ['Baseline', 'CNN', 'ResNet', 'Transformer']
accuracy = [0.72, 0.85, 0.91, 0.94]
std = [0.03, 0.02, 0.015, 0.01]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
bars = ax.bar(models, accuracy, yerr=std, capsize=5, 
              color=colors, edgecolor='black', linewidth=1.5)

# Styling
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Comparison on ImageNet', fontsize=14, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.tick_params(axis='both', labelsize=11)

# Add value labels
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

# Save in multiple formats
fig.savefig(os.path.join(OUTPUT_DIR, '19_publication_figure.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, '19_publication_figure.pdf'), bbox_inches='tight')
print(f"  Saved: PNG and PDF versions")

if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# Reset style
plt.style.use('default')


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("EXERCISES COMPLETE!")
print("="*60)

print(f"""
All figures saved to: {OUTPUT_DIR}

Key functions created:
  - quick_hist(data, title)        : Fast distribution check
  - plot_gradient_flow(grads)      : Debug vanishing/exploding gradients
  - plot_predictions(y_true, y_pred): Compare predictions to actuals
  - plot_training_dashboard(history): Comprehensive training monitor
  - plot_confusion_matrix(cm)      : Classification evaluation
  - plot_embeddings_2d(emb, labels): Visualize learned representations
  - show_image_grid(images)        : Display image samples
  - plot_lr_finder(lrs, losses)    : Learning rate selection

Usage tips:
  1. Use fig, ax = plt.subplots() for control
  2. Always label axes and add titles
  3. Use tight_layout() to prevent overlap
  4. Save with bbox_inches='tight' to crop whitespace
  5. Use dpi=300 for publication, dpi=150 for web

Run with --show flag to display plots interactively:
  python exercises.py --show
""")

if not SHOW_PLOTS:
    print("(Run with --show to see plots interactively)")
