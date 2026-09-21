# Module 5: Privacy & Data Rights

## The Privacy Challenge

Large language models are trained on massive datasets scraped from the internet. This raises questions:
- Whose data was used?
- Did they consent?
- Can models leak private information?

## Memorization in LLMs

Models can memorize and reproduce training data:

```
Prompt: "My social security number is..."
Risk: Model completes with actual SSN from training data
```

### Factors that increase memorization:
- Larger models
- More training epochs
- Duplicated data
- Distinctive/unique content

### Extraction attacks:
Adversaries can craft prompts to extract memorized data:
- Specific formatting prompts
- Completion attacks
- Membership inference

## Privacy Regulations

### GDPR (EU)

**Key rights:**
- **Right to access**: Know what data is held
- **Right to erasure**: Request deletion ("right to be forgotten")
- **Right to explanation**: Understand automated decisions
- **Data minimization**: Only collect necessary data

**Challenge for LLMs**: How do you "delete" someone from a trained model?

### CCPA (California)
- Similar rights to GDPR
- Right to know, delete, opt-out

### HIPAA (US Healthcare)
- Strict rules on protected health information (PHI)
- Training on medical data requires de-identification

## Privacy-Preserving Techniques

### Data Sanitization

Before training:
```python
# Remove PII patterns
import re

def sanitize(text):
    # Email
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
    # Phone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text
```

**Limitations**: Pattern matching misses context-dependent PII

### Differential Privacy

Add noise during training so individual records can't be identified:

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

**Tradeoff**: Privacy vs model quality

### Federated Learning

Train on distributed data without centralizing it:
```
Device 1: Local training → Send gradients (not data)
Device 2: Local training → Send gradients
Server: Aggregate gradients → Update global model
```

## Machine Unlearning

How do you remove someone's data from a trained model?

**Approaches:**
- **Retraining**: Expensive, but guaranteed
- **Fine-tuning to forget**: Update model to "unlearn" specific data
- **Approximate unlearning**: Remove influence without full retrain

**Current state**: Active research area, no perfect solution

## Consent and Data Provenance

### Questions to ask:
- Was consent obtained for this use?
- Does the license allow commercial training?
- Is the data source documented?

### Data documentation:
- **Datasheets for Datasets** (Gebru et al.): Standardized documentation
- Track provenance through the pipeline

## Exercises

1. Implement PII detection and redaction for a text dataset
2. Train a simple model with differential privacy using Opacus
3. Attempt a memorization extraction attack on a fine-tuned model

## Resources

- GDPR text: https://gdpr-info.eu
- Opacus (differential privacy): https://opacus.ai
- "Extracting Training Data from Large Language Models" (Carlini et al., 2021)

## What's Next?

Module 6 covers **Misinformation & Deepfakes** — AI-generated content and truth.
