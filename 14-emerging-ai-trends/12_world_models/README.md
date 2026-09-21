# Module 12: World Models

## What Is a World Model?

A model that understands how the world works — physics, dynamics, cause and effect.

```
Current LLMs:  Predict next token (language patterns)
World models:  Predict next state (physical reality)
```

## The Vision

Imagine an AI that can:
- Watch a video and understand what's physically happening
- Predict what will happen next
- "Imagine" scenarios that haven't happened
- Plan actions in a simulated world

This is the world model vision.

## Video as World Understanding

### Sora (OpenAI, 2024)
Video generation that shows physical understanding:
- Objects persist through occlusion
- Consistent physics (mostly)
- Coherent 3D space

**Key insight**: To generate realistic video, you must understand reality.

### Genie (DeepMind, 2024)
Learns controllable world models from video:
- Watch gameplay → Learn world dynamics
- Generate new playable environments
- No explicit physics programming

## How World Models Work

### Latent World Model Architecture

```
Video frames → Encoder → Latent states → Predictor → Future latents → Decoder → Future frames
                              ↑
                         Actions (optional)
```

1. **Encode**: Compress video to latent representation
2. **Predict**: Model dynamics in latent space
3. **Decode**: Generate predicted futures

### Training

Learn to predict future states:
```
Loss = ||Predicted_state(t+1) - Actual_state(t+1)||²
```

With actions:
```
Loss = ||Predictor(state_t, action_t) - state_{t+1}||²
```

## Physical Understanding

Do these models truly understand physics?

**Evidence for:**
- Consistent object permanence
- Reasonable dynamics
- Generalization to novel scenarios

**Evidence against:**
- Still makes physics errors
- May be sophisticated pattern matching
- Limited to training distribution

## Applications

### Robotics
Plan actions in simulated world before acting in real world:
```
Observe → Build world model → Plan in simulation → 
Execute best plan in reality
```

### Video Prediction
Generate plausible futures for:
- Autonomous driving
- Weather forecasting
- Video compression

### Game AI
Imagine and evaluate strategies:
```
Current state → Imagine 1000 possible futures → 
Choose action leading to best outcomes
```

## Current Limitations

1. **Training data**: Need massive video datasets
2. **Compute**: Very expensive to train
3. **Evaluation**: Hard to measure "understanding"
4. **Generalization**: May not transfer to new domains
5. **Long-term coherence**: Breaks down over time

## The Big Question

Is video generation → world understanding?

Or is it sophisticated interpolation?

**Active debate in the field.**

## Exercises

1. Explore Sora/Runway generated videos for physics errors
2. Discuss: What would prove "true" world understanding?
3. Design an evaluation for world model quality

## Resources

- "Video generation models as world simulators" (OpenAI, 2024)
- "Genie: Generative Interactive Environments" (DeepMind, 2024)
- "World Models" (Ha & Schmidhuber, 2018) — foundational paper

## What's Next?

Module 13 covers **Multi-head Latent Attention** — extreme KV cache compression for long contexts.
