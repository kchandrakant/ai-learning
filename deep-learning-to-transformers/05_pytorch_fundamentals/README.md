# Step 5: PyTorch Fundamentals

## Tensors: The Foundation

```python
import torch

# Create tensors
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)  # Random normal

# From NumPy
x = torch.from_numpy(np_array)
np_array = x.numpy()

# Device
x = torch.randn(3, 4, device='cuda')  # GPU
x = x.to('cuda')
x = x.to('cpu')
```

## Tensor Operations

```python
# Basic math
y = x + 2
y = x * 3
y = x @ w  # Matrix multiplication

# Reshaping
x = x.view(2, 6)
x = x.reshape(2, 6)
x = x.squeeze()   # Remove size-1 dims
x = x.unsqueeze(0)  # Add dim

# Indexing
x[:, 0]  # First column
x[x > 0]  # Boolean indexing
```

## Autograd: Automatic Differentiation

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x

y.backward()  # Compute gradients
print(x.grad)  # dy/dx = 2*2 + 3 = 7

# Disable gradient tracking
with torch.no_grad():
    y = model(x)
```

## Building Models with nn.Module

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = MyModel(784, 256, 10)
```

## Sequential Models

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
```

## Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()  # Includes softmax
loss = criterion(logits, targets)

# Binary classification
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid

# Regression
criterion = nn.MSELoss()
criterion = nn.L1Loss()
```

## Optimizers

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # Training step
    pass
```

## Complete Training Loop

```python
model = MyModel(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(device))
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.to(device)).sum().item()
        
        print(f'Epoch {epoch}, Val Acc: {100*correct/total:.2f}%')
```

## Saving and Loading

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

## Files

- `pytorch_fundamentals.py` - PyTorch examples

## Key Takeaways

1. Tensors are like NumPy arrays with GPU support
2. Autograd handles gradient computation
3. nn.Module is the base class for models
4. DataLoader handles batching
5. Training loop: forward, loss, backward, step

## What's Next?

Step 6: **Convolutions** — learning from spatial structure.
