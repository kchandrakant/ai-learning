# Step 5: QLoRA (Quantized LoRA)

## The Breakthrough

QLoRA = 4-bit quantized base model + LoRA adapters in 16-bit

This enables fine-tuning 70B models on a single GPU.

## Memory Comparison

```
Model        Full FT     LoRA      QLoRA
-----------------------------------------
Llama-2-7B   ~56GB      ~16GB     ~6GB
Llama-2-13B  ~104GB     ~32GB     ~12GB
Llama-2-70B  ~560GB     ~160GB    ~48GB
```

## Key Innovations

### 1. NF4 (4-bit NormalFloat)
Optimized 4-bit data type for normally distributed weights.

### 2. Double Quantization
Quantize the quantization constants themselves.

### 3. Paged Optimizers
Handle memory spikes with CPU offloading.

## Implementation

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

## Training Tips

```python
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=1,      # Small batch for memory
    gradient_accumulation_steps=16,      # Effective batch = 16
    learning_rate=2e-4,
    max_grad_norm=0.3,                   # Gradient clipping
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=False,                          # Use bf16 instead
    bf16=True,
    optim="paged_adamw_32bit",           # Paged optimizer
)
```

## When to Use QLoRA vs LoRA

| Scenario | Recommendation |
|----------|---------------|
| Limited VRAM (<24GB) | QLoRA |
| Larger models (>13B) | QLoRA |
| Maximum quality | LoRA (slightly better) |
| Fast iteration | LoRA (faster training) |

## Files

- `qlora.py` - QLoRA training implementation

## Key Takeaways

1. QLoRA = 4-bit base + 16-bit LoRA
2. Enables 70B fine-tuning on single GPU
3. ~97% of full fine-tuning quality
4. Use NF4 + double quantization
5. Use paged optimizers for memory spikes
