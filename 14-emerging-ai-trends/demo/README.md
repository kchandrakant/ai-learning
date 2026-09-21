# Emerging AI Trends - Demos

Hands-on demonstrations of cutting-edge AI techniques.

## Available Demos

### 1. Chain-of-Thought Comparison
```bash
python demo_cot_comparison.py
```
Compare direct prompting vs chain-of-thought on reasoning tasks.

### 2. Self-Consistency Voting
```bash
python demo_self_consistency.py
```
Implement self-consistency with majority voting for improved accuracy.

### 3. TabPFN Quick Start
```bash
python demo_tabpfn.py
```
Zero-shot tabular classification with TabPFN.

### 4. Model Quantization
```bash
python demo_quantization.py
```
Quantize a model and measure quality vs speed tradeoffs.

### 5. Local LLM with Ollama
```bash
python demo_local_llm.py
```
Run efficient models locally using Ollama.

### 6. Mamba vs Transformer
```bash
python demo_mamba_comparison.py
```
Compare Mamba and Transformer on sequence tasks (requires CUDA).

## Running Demos

```bash
# Navigate to demo folder
cd demo/

# Run any demo
python <demo_name>.py
```

## Hardware Notes

Some demos require specific hardware:
- **Mamba demos**: CUDA GPU required
- **Quantization**: GPU recommended
- **TabPFN**: CPU sufficient for small datasets

## Creating Your Own Experiments

Use these demos as starting points to explore:
- Different reasoning techniques
- Architecture comparisons
- Efficiency optimizations
