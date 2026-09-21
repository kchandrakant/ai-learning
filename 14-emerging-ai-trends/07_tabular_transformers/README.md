# Module 7: Tabular Transformers

## The Tabular Challenge

Deep learning dominates images and text. But for tabular data, XGBoost often still wins.

**Why?**
```
Text/Images: Homogeneous data, clear structure (sequences, grids)
Tabular:     Heterogeneous columns, no natural order, mixed types
```

## The Landscape

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| XGBoost/LightGBM | Fast, robust, handles mixed types | Limited representation learning |
| Neural networks | Representation learning | Struggle with tabular data |
| Tabular transformers | Best of both? | More complex, mixed results |

## Key Approaches

### TabTransformer (2020)

**Idea**: Apply attention to categorical features only

```
Categorical columns → Embeddings → Transformer → 
                                                   → Concat → MLP → Output
Numerical columns ──────────────────────────────→
```

**Why**: Categoricals benefit from contextual embeddings; numericals are already informative.

### FT-Transformer (2021)

**Idea**: Tokenize ALL features (numerical and categorical)

```python
# Feature tokenization
for each column:
    if categorical:
        token = embedding(value)
    else:  # numerical
        token = linear_projection(value) + column_embedding
        
# Apply transformer
tokens = transformer(all_tokens)
output = classifier(tokens[CLS])
```

**Key insight**: Treat each feature as a "token" with its own embedding.

### TabPFN (2022)

**Idea**: In-context learning for tabular data

```
Training: Pre-train on millions of synthetic datasets
Inference: New dataset → Feed as context → Predict in one forward pass
```

**Revolutionary**: No training on your data — just inference!

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier()
# No fit needed — just predict!
predictions = clf.predict(X_test, X_train, y_train)
```

**Limitation**: Works best for small datasets (<10K rows, <100 features)

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Small dataset (<1K) | TabPFN |
| Medium dataset | FT-Transformer or XGBoost |
| Large dataset | XGBoost/LightGBM |
| Need interpretability | XGBoost |
| Mixed types, many categoricals | TabTransformer |
| Representation learning | FT-Transformer |

## Practical Implementation

### FT-Transformer with pytorch-tabular

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig

config = FTTransformerConfig(
    task="classification",
    num_attn_blocks=3,
    num_heads=8,
    ffn_dropout=0.1,
)

model = TabularModel(
    data_config=data_config,
    model_config=config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

model.fit(train=train_df, validation=val_df)
```

## The Honest Assessment

**When transformers help**:
- Many categorical features
- Complex feature interactions
- Transfer learning scenarios
- Feature extraction for downstream tasks

**When they don't**:
- Clean numerical data
- Small datasets (without TabPFN)
- Need for interpretability
- Computational constraints

**The reality**: XGBoost is still a strong baseline. Beat it before claiming victory.

## Exercises

1. Compare XGBoost vs FT-Transformer on a Kaggle dataset
2. Try TabPFN on small classification problems
3. Experiment with different tokenization strategies

## Resources

- TabPFN: https://github.com/automl/TabPFN
- FT-Transformer paper: https://arxiv.org/abs/2106.11959
- pytorch-tabular: https://github.com/manujosephv/pytorch_tabular

## What's Next?

Module 8 covers **Time Series Transformers** — attention for temporal data.
