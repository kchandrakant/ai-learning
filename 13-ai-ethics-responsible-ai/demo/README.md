# AI Ethics & Responsible AI - Demos

Hands-on demonstrations for ethical AI practices.

## Available Demos

### 1. Bias Audit with Fairlearn
```bash
python demo_fairlearn_audit.py
```
Audit a model for demographic disparities using Fairlearn.

### 2. SHAP Explanations
```bash
python demo_shap_explanations.py
```
Generate and visualize SHAP explanations for model predictions.

### 3. PII Detection and Redaction
```bash
python demo_pii_detection.py
```
Detect and redact personally identifiable information from text.

### 4. Red Team Prompts
```bash
python demo_red_team.py
```
Test a model against common jailbreak and safety prompts.

### 5. Model Card Generator
```bash
python demo_model_card.py
```
Generate a model card template for documentation.

### 6. Carbon Footprint Tracking
```bash
python demo_carbon_tracking.py
```
Track the carbon footprint of model training/inference using CodeCarbon.

## Running Demos

```bash
# Navigate to demo folder
cd demo/

# Run any demo
python <demo_name>.py
```

## Prerequisites

Most demos require:
- PyTorch
- Transformers
- Fairlearn (for bias demos)
- SHAP (for interpretability demos)

See `requirements.txt` in the course root.

## Creating Your Own Demos

Use these as templates to build demos for your specific use cases.
