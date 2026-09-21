# Module 3: Interpretability & Explainability

## Why Interpretability?

Models make decisions that affect people. We need to understand why:
- **Trust**: Should we rely on this prediction?
- **Debugging**: Why did the model fail?
- **Compliance**: Regulations may require explanations
- **Improvement**: What's the model actually learning?

## Key Distinction

**Interpretability**: Model is inherently understandable
- Linear regression: coefficients tell us feature importance
- Decision trees: follow the path

**Explainability**: We can explain outputs post-hoc
- Deep neural networks: need external methods

## Post-Hoc Explanation Methods

### SHAP (SHapley Additive exPlanations)

Based on game theory: how much does each feature contribute?

```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X)

# Visualize
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

**Pros**: Theoretically grounded, consistent
**Cons**: Expensive for large models, assumes feature independence

### LIME (Local Interpretable Model-agnostic Explanations)

Fits a simple model locally around each prediction:

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train)
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

**Pros**: Model-agnostic, intuitive
**Cons**: Explanations can be unstable, local only

## Attention as Explanation?

Attention weights show what the model "looks at" — but are they explanations?

**The problem**: "Attention is not Explanation" (Jain & Wallace, 2019)
- Attention can be manipulated without changing outputs
- Multiple attention patterns can produce same result
- Correlation ≠ causation

**Use with caution**: Attention is descriptive, not necessarily causal.

## Mechanistic Interpretability

A deeper approach: understand the algorithms learned by neural networks.

### Key Concepts
- **Circuits**: Subgraphs of the network that perform specific functions
- **Features**: What individual neurons/directions represent
- **Superposition**: Multiple features encoded in same neurons

### Sparse Autoencoders
Decompose activations into interpretable features:
```
activation → sparse code → interpretable features
```

This is cutting-edge research (Anthropic, 2023-2024).

## When Is Explainability Required?

### Legal Requirements
- **GDPR Article 22**: Right to explanation for automated decisions
- **US Equal Credit Opportunity Act**: Must explain credit denials
- **Healthcare**: Often requires interpretable models

### Practical Requirements
- High-stakes decisions (medical, legal, financial)
- Model debugging and improvement
- Building user trust

## The Interpretability-Performance Tradeoff

```
Linear models:     High interpretability, limited performance
Tree ensembles:    Moderate interpretability, good performance
Neural networks:   Low interpretability, best performance
```

This tradeoff is real but narrowing. Research aims to make powerful models interpretable.

## Exercises

1. Apply SHAP to explain a classification model's predictions
2. Compare LIME explanations across similar inputs — are they stable?
3. Visualize attention patterns and discuss what they do/don't tell us

## Resources

- SHAP documentation: https://shap.readthedocs.io
- LIME paper: https://arxiv.org/abs/1602.04938
- Anthropic's interpretability research: https://transformer-circuits.pub

## What's Next?

Module 4 covers **Constitutional AI** — training models to follow principles through self-improvement.
