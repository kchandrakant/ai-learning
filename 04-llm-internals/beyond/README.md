# LLM Internals: Beyond the Basics

Emerging research and future directions in LLM development.

**Note**: Many advanced topics are now covered in dedicated courses:
- **[Course 13: AI Ethics & Responsible AI](../../13-ai-ethics-responsible-ai/)** — Alignment, Constitutional AI, safety
- **[Course 14: Emerging AI Trends](../../14-emerging-ai-trends/)** — MoE, test-time compute, reasoning paradigms

---

## Topics Covered Here (Brief)

### Long Context
- 100K+ token contexts (Claude, Gemini)
- Ring attention, streaming attention
- Memory-efficient training
- Positional encoding challenges (RoPE covered in Course 03)

### Efficient Training
- Lower precision training (FP8)
- Gradient compression
- Curriculum learning
- Data-efficient pre-training

---

## Topics Moved to Other Courses

| Topic | Now In | Why |
|-------|--------|-----|
| **Mixture of Experts (MoE)** | Course 14, Module 10 | Scaling paradigm with broader context |
| **Test-Time Compute** | Course 14, Module 2 | System 1/2 reasoning framework |
| **Constitutional AI** | Course 13, Module 4 | Part of alignment/ethics curriculum |
| **Alignment & Safety** | Course 13, Modules 1, 11 | Dedicated ethics treatment |
| **Self-Improving Systems** | Course 14, Module 14 | Frontier research |

---

## Open Research Questions

1. **Data efficiency:** Can we train good models with less data?
2. **Continual learning:** How to update models without catastrophic forgetting?
3. **Interpretability:** Can we understand what models learn?
4. **Alignment:** How to ensure models do what we want? → See **Course 13**
5. **Efficiency:** Can we match capability at lower cost?

---

## What's Next?

After completing this course:
- **Course 13** (AI Ethics): Alignment, safety, responsible deployment
- **Course 14** (Emerging Trends): MoE, reasoning, alternative architectures

---

## Resources

- arXiv cs.CL for latest papers
- AI safety research (Anthropic, DeepMind, OpenAI)
- EleutherAI for open research
