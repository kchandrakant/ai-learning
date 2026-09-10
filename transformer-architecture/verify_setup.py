"""Verify the lab setup is working correctly."""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 50)
print("TRANSFORMER LEARNING LAB - SETUP VERIFICATION")
print("=" * 50)

# Check versions
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Matplotlib version: {matplotlib.__version__}")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device: {device}")

# Quick tensor test
x = torch.randn(2, 3)
print(f"✓ Tensor creation works: shape {x.shape}")

# Quick matrix multiplication
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.matmul(a, b)
print(f"✓ Matrix multiplication works: ({a.shape}) @ ({b.shape}) = {c.shape}")

# Quick softmax test (we'll use this a lot)
logits = torch.randn(1, 5)
probs = torch.softmax(logits, dim=-1)
print(f"✓ Softmax works: sum = {probs.sum().item():.4f} (should be 1.0)")

print("\n" + "=" * 50)
print("🚀 LAB SETUP COMPLETE - READY TO LEARN!")
print("=" * 50)
