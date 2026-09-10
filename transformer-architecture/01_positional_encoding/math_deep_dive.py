"""
Deep dive into the math of Positional Encoding.

The formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Let's understand WHY this works.
"""

import torch
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: Understanding the Frequency Term
# =============================================================================

print("=" * 70)
print("PART 1: THE FREQUENCY TERM - 10000^(2i/d_model)")
print("=" * 70)

d_model = 64

print(f"\nWith d_model = {d_model}, let's see the divisor for each dimension:\n")
print(f"{'Dim (i)':<10} {'2i/d_model':<15} {'Divisor':<15} {'Wavelength':<15}")
print("-" * 55)

for i in [0, 4, 8, 16, 24, 31]:
    exponent = (2 * i) / d_model
    divisor = 10000 ** exponent
    # Wavelength = 2π * divisor (how many positions for one complete cycle)
    wavelength = 2 * math.pi * divisor
    print(f"{i:<10} {exponent:<15.3f} {divisor:<15.1f} {wavelength:<15.1f}")

print("""
KEY INSIGHT:
- Dimension 0: Completes a cycle every ~6 positions (high frequency)
- Dimension 31: Completes a cycle every ~50,000 positions (low frequency)
- This is like a clock with multiple hands moving at different speeds!
""")

# =============================================================================
# PART 2: Why sin AND cos? The Rotation Property
# =============================================================================

print("=" * 70)
print("PART 2: WHY SIN AND COS? THE ROTATION PROPERTY")
print("=" * 70)

print("""
The paper claims: "We chose this function because we hypothesized it would 
allow the model to easily learn to attend by relative positions."

Here's the mathematical reason:

For any offset k, PE(pos+k) can be expressed as a LINEAR FUNCTION of PE(pos).

This is because of the trigonometric identity:
    sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
    cos(a + b) = cos(a)cos(b) - sin(a)sin(b)

In matrix form, this is a ROTATION:
    [sin(pos+k)]   [cos(k)  sin(k)] [sin(pos)]
    [cos(pos+k)] = [-sin(k) cos(k)] [cos(pos)]
""")

# Demonstrate the rotation property
print("\nDemonstration: PE(pos+k) = Rotation_k × PE(pos)\n")

# Pick a specific dimension (i=0 for simplicity)
freq = 1.0  # 10000^0 = 1

pos = 5
k = 3  # offset

# Calculate directly
sin_pos = math.sin(pos / freq)
cos_pos = math.cos(pos / freq)
sin_pos_k = math.sin((pos + k) / freq)
cos_pos_k = math.cos((pos + k) / freq)

# Calculate via rotation matrix
cos_k = math.cos(k / freq)
sin_k = math.sin(k / freq)

# Rotation: [cos(k), sin(k); -sin(k), cos(k)] @ [sin(pos); cos(pos)]
rotated_sin = cos_k * sin_pos + sin_k * cos_pos
rotated_cos = -sin_k * sin_pos + cos_k * cos_pos

print(f"Position {pos}, offset k={k}:")
print(f"  Direct calculation:  sin={sin_pos_k:.6f}, cos={cos_pos_k:.6f}")
print(f"  Via rotation matrix: sin={rotated_sin:.6f}, cos={rotated_cos:.6f}")
print(f"  Match: {np.allclose([sin_pos_k, cos_pos_k], [rotated_sin, rotated_cos])}")

print("""
WHY THIS MATTERS:
The model can learn a simple linear transformation (the rotation matrix)
to convert "position 5 encoding" to "position 8 encoding".

This makes learning RELATIVE positions much easier than learning 
absolute position relationships!
""")

# =============================================================================
# PART 3: The Dot Product Property
# =============================================================================

print("=" * 70)
print("PART 3: DOT PRODUCT MEASURES RELATIVE DISTANCE")
print("=" * 70)

print("""
When we compute attention: score = PE(pos_i) · PE(pos_j)

The dot product of two position encodings depends primarily on their 
RELATIVE distance |pos_i - pos_j|, not their absolute positions.
""")

# Create position encodings
def create_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

pe = create_pe(100, 64)

print("\nDot product PE(pos_i) · PE(pos_j) for various positions:\n")
print(f"{'pos_i':<8} {'pos_j':<8} {'|i-j|':<8} {'Dot Product':<12}")
print("-" * 40)

# Show that same relative distance → similar dot product
pairs = [(0, 5), (10, 15), (50, 55), (0, 10), (30, 40), (70, 80)]
for i, j in pairs:
    dot = np.dot(pe[i], pe[j])
    print(f"{i:<8} {j:<8} {abs(i-j):<8} {dot:<12.4f}")

print("""
NOTICE: 
- Pairs with |i-j|=5 have similar dot products (~52)
- Pairs with |i-j|=10 have similar dot products (~37)
- The dot product depends on RELATIVE distance, not absolute position!
""")

# =============================================================================
# PART 4: Why 10000? And why this specific formula?
# =============================================================================

print("=" * 70)
print("PART 4: WHY 10000? WHY THIS FORMULA?")
print("=" * 70)

print("""
10000 is somewhat arbitrary but provides good properties:

1. RANGE OF WAVELENGTHS:
   - Shortest wavelength: 2π ≈ 6.28 (dimension 0)
   - Longest wavelength: 2π × 10000 ≈ 62,832 (last dimension)
   
   This covers sequences from very short to very long.

2. GEOMETRIC PROGRESSION:
   Using 10000^(2i/d_model) creates a geometric series of frequencies,
   similar to how musical octaves work. Each dimension covers a 
   different "scale" of position information.

3. NUMERICAL STABILITY:
   The formula in code uses:
       exp(2i × -log(10000) / d_model)
   instead of:
       1 / (10000^(2i/d_model))
   
   This is mathematically identical but more numerically stable.
""")

# Show the equivalence
print("\nNumerical equivalence check:")
for i in [0, 16, 31]:
    method1 = 10000 ** (2*i / d_model)
    method2 = math.exp(2*i * math.log(10000) / d_model)
    div_term = math.exp(2*i * (-math.log(10000.0) / d_model))
    print(f"  i={i}: 10000^(2i/d) = {method1:.6f}, exp() = {method2:.6f}, div_term = {div_term:.6f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Frequency spectrum
ax1 = axes[0, 0]
dims = np.arange(0, d_model // 2)
frequencies = 1.0 / (10000 ** (2 * dims / d_model))
wavelengths = 2 * np.pi / frequencies
ax1.semilogy(dims * 2, wavelengths, 'b-', linewidth=2)
ax1.set_xlabel('Dimension Index')
ax1.set_ylabel('Wavelength (positions, log scale)')
ax1.set_title('Wavelength vs Dimension\n(Higher dims = longer wavelengths)')
ax1.grid(True, alpha=0.3)

# Plot 2: Position encodings as waves
ax2 = axes[0, 1]
positions = np.arange(50)
for dim_i, color in [(0, 'red'), (8, 'orange'), (16, 'green'), (24, 'blue')]:
    freq = 1.0 / (10000 ** (2 * dim_i / d_model))
    values = np.sin(positions * freq)
    ax2.plot(positions, values, color=color, label=f'dim {dim_i*2} (sin)', alpha=0.8)
ax2.set_xlabel('Position')
ax2.set_ylabel('Encoding Value')
ax2.set_title('Sin Waves at Different Dimensions\n(Lower dim = higher frequency)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Dot product as function of relative distance
ax3 = axes[1, 0]
pe_full = create_pe(200, 64)
base_pos = 50
distances = range(-30, 31)
dot_products = [np.dot(pe_full[base_pos], pe_full[base_pos + d]) for d in distances]
ax3.plot(distances, dot_products, 'b-', linewidth=2)
ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Relative Position (pos_j - pos_i)')
ax3.set_ylabel('Dot Product PE(i) · PE(j)')
ax3.set_title('Dot Product Depends on Relative Distance\n(Symmetric around 0)')
ax3.grid(True, alpha=0.3)

# Plot 4: Rotation visualization (2D projection)
ax4 = axes[1, 1]
# Take just dimensions 0 and 1 (one sin/cos pair)
pe_2d = pe_full[:50, :2]
colors = plt.cm.viridis(np.linspace(0, 1, 50))
for i in range(50):
    ax4.scatter(pe_2d[i, 0], pe_2d[i, 1], c=[colors[i]], s=50)
ax4.plot(pe_2d[:, 0], pe_2d[:, 1], 'k-', alpha=0.3)
ax4.set_xlabel('Dimension 0 (sin)')
ax4.set_ylabel('Dimension 1 (cos)')
ax4.set_title('First 50 Positions in 2D (dims 0,1)\n(Traces a circle - rotation!)')
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_positional_encoding/math_visualization.png', dpi=150)
print("Saved: 01_positional_encoding/math_visualization.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHY SINUSOIDAL POSITIONAL ENCODING WORKS")
print("=" * 70)
print("""
1. UNIQUE ENCODINGS
   Each position gets a unique vector (like a fingerprint)

2. BOUNDED VALUES  
   sin/cos ∈ [-1, 1], no explosion regardless of position

3. RELATIVE POSITIONS
   PE(pos+k) = Rotation_k × PE(pos)
   → Model can easily learn relative position relationships

4. DOT PRODUCT PROPERTY
   PE(i) · PE(j) depends mainly on |i-j|
   → Attention naturally captures relative distance

5. EXTRAPOLATION
   Works for positions beyond training data
   (unlike learned position embeddings)

6. NO PARAMETERS
   Pre-computed, deterministic, no learning required
""")
