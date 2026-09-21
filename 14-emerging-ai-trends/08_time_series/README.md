# Module 8: Time Series Transformers

## The Time Series Landscape

Traditional approaches:
- ARIMA, ETS (statistical)
- Prophet (Facebook)
- LSTM, GRU (neural)

**Question**: Can transformers do better?

## Challenges for Transformers

Time series ≠ text:
- Continuous values, not discrete tokens
- Strong temporal patterns (seasonality, trends)
- Variable lengths, irregular sampling
- Need for probabilistic forecasts

## Key Approaches

### Temporal Fusion Transformer (TFT, 2021)

Combines attention with time series-specific components:

```
Static features ─────────────────────┐
                                      ↓
Time-varying known future ──→ Variable selection → LSTM → Self-attention → Output
                                      ↑
Time-varying observed ───────────────┘
```

**Key innovations**:
- Variable selection networks
- Gating mechanisms
- Multi-horizon forecasting
- Interpretable attention

### PatchTST (2023)

**Key idea**: Treat time series like images — use patches!

```
Time series: [v1, v2, v3, v4, v5, v6, v7, v8, ...]
                 ↓ Patching
Patches:     [[v1,v2,v3], [v4,v5,v6], [v7,v8,v9], ...]
                 ↓
Each patch → Embedding → Transformer
```

**Why patches?**:
- Reduce sequence length (efficiency)
- Capture local patterns
- Similar to ViT for images

### Informer (2021)

Efficient attention for long sequences:
- ProbSparse attention (attend to important positions)
- Distilling (reduce sequence length through layers)
- Generative-style decoder

### Foundation Models (TimesFM, Chronos)

**Big idea**: Pre-train on many time series, transfer to new domains.

```python
from chronos import ChronosPipeline

model = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
forecast = model.predict(historical_data, horizon=24)
```

**Zero-shot forecasting** — no training on your specific data!

## When Transformers Help

**Good scenarios**:
- Multiple related time series
- Rich covariates (weather, events, etc.)
- Long-range dependencies
- Transfer learning available

**Challenging scenarios**:
- Single short time series
- Need extreme efficiency
- Simple seasonal patterns (ARIMA might suffice)

## Practical Implementation

### Using PatchTST

```python
from transformers import PatchTSTForPrediction

model = PatchTSTForPrediction.from_pretrained(
    "ibm/patchtst-etth1-forecast"
)

# Prepare data
past_values = torch.tensor(historical_data)
outputs = model(past_values=past_values)
predictions = outputs.prediction_outputs
```

### Using pytorch-forecasting

```python
from pytorch_forecasting import TemporalFusionTransformer

model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
)
```

## Comparison

| Method | Strengths | When to Use |
|--------|-----------|-------------|
| ARIMA | Simple, interpretable | Single series, short term |
| Prophet | Handles seasonality, holidays | Business metrics |
| LSTM | Sequences, nonlinear | Medium complexity |
| TFT | Multi-horizon, interpretable | Multiple series, covariates |
| PatchTST | Efficient, good scaling | Long sequences |
| Chronos | Zero-shot | Quick baseline, new domains |

## Exercises

1. Compare TFT vs ARIMA on a forecasting benchmark
2. Try Chronos zero-shot on your own time series
3. Experiment with patch sizes in PatchTST

## Resources

- "Temporal Fusion Transformers" (Lim et al., 2021)
- "PatchTST" paper (2023)
- pytorch-forecasting library
- Chronos from Amazon

## What's Next?

Module 9 covers **Graph Transformers** — attention on graph-structured data.
