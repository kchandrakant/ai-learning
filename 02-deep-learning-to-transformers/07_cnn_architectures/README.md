# Step 7: CNN Architectures

## Pooling Layers

Reduce spatial dimensions, add translation invariance.

```python
# Max pooling: take maximum in each window
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# (batch, 64, 224, 224) → (batch, 64, 112, 112)

# Average pooling
pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling (for classification head)
gap = nn.AdaptiveAvgPool2d(1)
# (batch, 512, 7, 7) → (batch, 512, 1, 1)
```

## Classic Pattern

```
[Conv → ReLU → Pool] × N → Flatten → FC → Output

Spatial dims ↓, channels ↑
```

## LeNet-5 (1998)

The original CNN for digit recognition.

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

## VGG (2014)

Deeper is better. Use 3×3 filters throughout.

```python
# VGG block: stack of 3×3 convs
def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)
```

## ResNet (2015)

Residual connections enable very deep networks.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection!
        return F.relu(x)
```

**Key insight:** Easier to learn `F(x) = 0` than `F(x) = x`.

## 1×1 Convolutions

Change number of channels without changing spatial size.

```python
# Reduce 512 channels to 64
bottleneck = nn.Conv2d(512, 64, kernel_size=1)
```

Used in:
- Bottleneck blocks (reduce computation)
- Channel mixing
- Network-in-Network architectures

## Modern Architecture Pattern

```python
class ModernBlock(nn.Module):
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        return x + residual
```

## Using Pretrained Models

```python
from torchvision import models

# Load pretrained ResNet
resnet = models.resnet50(pretrained=True)

# Replace final layer for your task
resnet.fc = nn.Linear(2048, num_classes)

# Freeze early layers (optional)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad = True
```

## Files

- `cnn_architectures.py` - CNN implementations

## Key Takeaways

1. Pooling reduces spatial dimensions
2. Deeper networks learn better features
3. Residual connections enable depth
4. 1×1 convs for channel mixing
5. Pretrained models are powerful starting points

## What's Next?

Step 8: **Recurrent Neural Networks** — processing sequences.
