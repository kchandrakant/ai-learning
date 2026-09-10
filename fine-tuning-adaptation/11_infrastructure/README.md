# Step 11: Training Infrastructure

## Options Overview

| Option | Cost | Ease | Control |
|--------|------|------|---------|
| Local GPU | $$$ upfront | Medium | Full |
| Cloud (AWS/GCP) | $/hour | Medium | Full |
| Lambda Labs | $/hour | Easy | Full |
| Managed (Together, Fireworks) | $/run | Easy | Limited |

## Local Setup

```bash
# Check GPU
nvidia-smi

# CUDA version
nvcc --version

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Requirements:**
- NVIDIA GPU (RTX 3090+ for 7B, A100 for 70B)
- 24GB+ VRAM for 7B with QLoRA
- Proper CUDA installation

## Cloud Options

### AWS

```bash
# Launch GPU instance
aws ec2 run-instances \
    --instance-type p4d.24xlarge \
    --image-id ami-xxx  # Deep Learning AMI
```

### Lambda Labs

```bash
# Simple API
lambda instance create \
    --instance-type gpu_1x_a100
```

### Google Colab (Free tier)

```python
# Check GPU in Colab
!nvidia-smi

# Limited to T4 (free) or A100 (Pro)
```

## Managed Platforms

### Together AI

```python
import together

# Fine-tune through API
together.Finetune.create(
    training_file="file-xxx",
    model="meta-llama/Llama-2-7b-hf",
    n_epochs=3,
)
```

### Fireworks AI

```python
# Upload and fine-tune
fireworks.fine_tune(
    base_model="llama-2-7b",
    dataset="your-dataset",
    method="lora",
)
```

## Cost Comparison

```
7B model, 1000 examples, 3 epochs:

Local (RTX 4090):     ~$0 (if owned), 1-2 hours
AWS p4d.24xlarge:     ~$30/hour × 2 hours = $60
Lambda A100:          ~$1.10/hour × 2 hours = $2.20
Together/Fireworks:   ~$5-20 (managed)
```

## Files

- `infrastructure.py` - Setup scripts

## Key Takeaways

1. Local is cheapest if you have hardware
2. Lambda Labs for affordable cloud GPUs
3. Managed platforms for simplicity
4. QLoRA makes consumer GPUs viable
5. Match infrastructure to model size
