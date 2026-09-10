# Step 6: Convolutions

## The Problem with Fully Connected

Images have spatial structure. FC layers ignore it.

```
28×28 image = 784 inputs
FC to 256 neurons = 784 × 256 = 200,704 parameters!
```

## The Convolution Operation

Slide a small filter (kernel) across the image.

```
Image (5×5):          Filter (3×3):         Output:
1 0 1 0 1             1 0 1
0 1 0 1 0      *      0 1 0        =        ?
1 0 1 0 1             1 0 1
0 1 0 1 0
1 0 1 0 1

At each position: element-wise multiply, then sum
```

## Implementation

```python
def conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    
    return output
```

## Key Concepts

### Filters/Kernels
```python
# Edge detection filter
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# In CNNs, filters are LEARNED!
```

### Stride
How many pixels to move the filter.
```
Stride 1: move 1 pixel → larger output
Stride 2: move 2 pixels → smaller output (downsampling)
```

### Padding
Add zeros around the image to control output size.
```
Same padding: output size = input size
Valid padding: no padding, output shrinks
```

## Parameter Sharing

Same filter applied everywhere → far fewer parameters.

```
3×3 filter on 28×28 image:
- Only 9 parameters (not 784×784)
- Same edge detector works anywhere in image
```

## Feature Maps

One filter → one feature map.
Multiple filters → multiple feature maps (channels).

```python
# Input: (batch, 3, 224, 224)  - 3 color channels
# Conv with 64 filters: (batch, 64, 222, 222)  - 64 feature maps
```

## PyTorch Convolution

```python
import torch.nn as nn

# 2D convolution
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3×3 filters
    stride=1,
    padding=1           # Same padding
)

# Input shape: (batch, channels, height, width)
x = torch.randn(16, 3, 224, 224)
output = conv(x)  # (16, 64, 224, 224)
```

## 1D Convolutions (for sequences)

```python
# For text, audio, time series
conv1d = nn.Conv1d(
    in_channels=256,    # Embedding dim
    out_channels=128,   # Output channels
    kernel_size=3       # Context window
)

# Input: (batch, channels, sequence_length)
x = torch.randn(16, 256, 100)
output = conv1d(x)  # (16, 128, 98)
```

## Files

- `convolutions.py` - Convolution visualization

## Key Takeaways

1. Convolution = sliding window dot product
2. Filters detect local patterns
3. Parameter sharing reduces complexity
4. Stride controls downsampling
5. Padding controls output size

## What's Next?

Step 7: **CNN Architectures** — building vision models.
