# Fine-tuning & Adaptation Demos

## Available Demos

### 1. Data Preparation (`data_prep_demo.py`)
Clean and format a dataset for fine-tuning.

```bash
python demo/data_prep_demo.py --input raw_data.json --output prepared_data.json
```

### 2. LoRA Training (`lora_demo.py`)
Fine-tune a model with LoRA.

```bash
python demo/lora_demo.py --model meta-llama/Llama-2-7b-hf --dataset ./data
```

### 3. QLoRA Training (`qlora_demo.py`)
Memory-efficient fine-tuning with QLoRA.

```bash
python demo/qlora_demo.py --model meta-llama/Llama-2-70b-hf
```

### 4. SFT Demo (`sft_demo.py`)
Supervised fine-tuning for instruction following.

```bash
python demo/sft_demo.py --dataset alpaca
```

### 5. DPO Training (`dpo_demo.py`)
Direct Preference Optimization.

```bash
python demo/dpo_demo.py --sft-model ./sft-output --preferences ./prefs.json
```

### 6. Evaluation (`eval_demo.py`)
Evaluate your fine-tuned model.

```bash
python demo/eval_demo.py --model ./fine-tuned --benchmark mmlu
```

### 7. Merge & Deploy (`deploy_demo.py`)
Merge adapters and deploy.

```bash
python demo/deploy_demo.py --base-model llama-2-7b --adapter ./lora-output
```

## Quick Start

```bash
# 1. Prepare data
python demo/data_prep_demo.py

# 2. Fine-tune with LoRA
python demo/lora_demo.py

# 3. Evaluate
python demo/eval_demo.py

# 4. Deploy
python demo/deploy_demo.py
```

## Sample Data

The `sample_data/` folder contains:
- `instructions.json` - Instruction dataset
- `preferences.json` - Preference pairs for DPO

---

Start with `lora_demo.py` for a quick fine-tuning example.
