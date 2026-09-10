# Step 4: Training Deep Networks

## Why Deep Networks Are Hard

- Vanishing/exploding gradients
- Poor initialization
- Internal covariate shift
- Overfitting

## Weight Initialization

Bad initialization → network doesn't learn.

```python
# Xavier/Glorot (for tanh/sigmoid)
# Variance = 1 / n_in
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

# He (for ReLU)
# Variance = 2 / n_in
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

**PyTorch:**
```python
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

## Batch Normalization

Normalize activations within each mini-batch.

```python
class BatchNorm:
    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            x_norm = (x - mean) / np.sqrt(var + 1e-5)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + 1e-5)
        
        return self.gamma * x_norm + self.beta  # Learnable scale and shift
```

**Benefits:**
- Stabilizes training
- Allows higher learning rates
- Slight regularization effect

**PyTorch:**
```python
self.bn = nn.BatchNorm1d(hidden_dim)
```

## Layer Normalization

Normalize across features (not batch). Used in transformers.

```python
self.ln = nn.LayerNorm(hidden_dim)
```

## Dropout

Randomly zero out neurons during training.

```python
class Dropout:
    def forward(self, x, p=0.5, training=True):
        if training:
            mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
            return x * mask
        return x
```

**Why it works:** Prevents co-adaptation, acts like ensemble.

**PyTorch:**
```python
self.dropout = nn.Dropout(p=0.5)
```

## Gradient Clipping

Prevent exploding gradients by capping magnitude.

```python
# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Clip by norm (more common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Learning Rate Schedules

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Warmup + decay (for transformers)
# Start low, increase, then decrease
```

## Residual Connections

Skip connections allow gradients to flow directly.

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        return x + self.layers(x)  # Add input to output
```

**Why:** Enables training of very deep networks (100+ layers).

## Complete Training Setup

```python
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
```

## Files

- `training_deep_networks.py` - Training utilities

## Key Takeaways

1. He initialization for ReLU
2. BatchNorm stabilizes training
3. Dropout prevents overfitting
4. Gradient clipping prevents explosions
5. Residual connections enable depth

## What's Next?

Step 5: **PyTorch Fundamentals** — the framework in depth.
