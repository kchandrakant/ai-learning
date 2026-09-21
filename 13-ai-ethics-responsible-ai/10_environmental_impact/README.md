# Module 10: Environmental Impact

## The Hidden Cost of AI

Training and running large AI models consumes significant resources:
- Electricity for computation
- Water for cooling data centers
- Hardware manufacturing footprint
- Electronic waste

## Training Compute Emissions

### Scale of Training

| Model | Estimated Training Emissions |
|-------|------------------------------|
| BERT | ~300 kg CO2 |
| GPT-3 | ~500 tons CO2 |
| GPT-4 | Unknown, likely higher |
| Llama 2 70B | ~300 tons CO2 |

For context: Average American emits ~15 tons CO2/year.

### What Drives Emissions

```
Emissions = Compute × Energy per compute × Carbon intensity of grid
```

**Compute**: FLOPs needed (scales with model size, data, training time)
**Energy efficiency**: Hardware generation matters (H100 >> A100 >> V100)
**Grid carbon intensity**: Varies 10-100× by location

### Location Matters

| Location | Grid Intensity | Relative Emissions |
|----------|----------------|-------------------|
| Quebec (hydro) | ~20 g CO2/kWh | Baseline |
| France (nuclear) | ~50 g CO2/kWh | 2.5× |
| US average | ~400 g CO2/kWh | 20× |
| China average | ~600 g CO2/kWh | 30× |

Same training job: 30× different emissions based on location.

## Inference at Scale

Training happens once. Inference happens billions of times.

```
Total impact = Training + (Queries × Energy per query × Deployment lifetime)
```

For widely-used models, inference can exceed training impact.

### Example Calculation
```
Model serves 1 billion queries/day
Each query: 0.001 kWh
Daily energy: 1 million kWh
If grid is 400g CO2/kWh: 400 tons CO2/day
Annual: ~150,000 tons CO2
```

## Water Consumption

Data centers need cooling. Often use water:

**Estimated water use:**
- Training GPT-3: ~700,000 liters
- Each ChatGPT conversation: ~500ml

In water-stressed regions, this is significant.

## Measuring AI Carbon Footprint

### Tools

**CodeCarbon** (Python):
```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
# Your training code
trainer.train()
emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO2")
```

**ML CO2 Impact**: https://mlco2.github.io/impact/

### What to Measure
- Training compute
- Hyperparameter search (often 10-100× training cost!)
- Inference at scale
- Hardware manufacturing (embodied carbon)

## Mitigation Strategies

### Efficient Training
- **Smaller models**: Often sufficient for task
- **Transfer learning**: Don't train from scratch
- **Efficient architectures**: MoE, sparse models
- **Mixed precision**: FP16/BF16 training

### Efficient Inference
- **Quantization**: INT8, INT4
- **Distillation**: Smaller student models
- **Caching**: Don't recompute same queries
- **Batching**: Better GPU utilization

### Location and Timing
- **Green data centers**: Locate in low-carbon regions
- **Load shifting**: Run jobs when grid is cleaner
- **Renewable energy**: PPAs, on-site generation

### Carbon Offsets
- Purchase offsets for residual emissions
- Controversial: additionality concerns
- Should be last resort, not first solution

## Reporting and Transparency

### What to Report
- Training compute (GPU-hours, FLOPs)
- Energy consumption
- Carbon emissions
- Location and grid information

### Standards
- Emerging requirements for AI carbon reporting
- GHG Protocol for Scope 2/3 emissions
- Science-based targets initiative

## The Efficiency vs Capability Tension

```
Trend: Larger models, more compute
Counter-trend: Efficiency improvements

Net effect: AI energy use growing faster than efficiency gains
```

## Exercises

1. Measure the carbon footprint of a training run using CodeCarbon
2. Compare emissions of training in different cloud regions
3. Calculate the inference footprint of a model at scale

## Resources

- CodeCarbon: https://codecarbon.io
- ML CO2 Impact calculator: https://mlco2.github.io/impact/
- "Energy and Policy Considerations for Deep Learning in NLP" (Strubell et al., 2019)

## What's Next?

Module 11 covers **Red Teaming** — proactively finding problems in AI systems.
