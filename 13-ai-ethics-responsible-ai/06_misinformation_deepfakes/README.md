# Module 6: Misinformation & Deepfakes

## The Content Authenticity Crisis

AI can generate:
- Realistic fake images (faces that don't exist)
- Convincing fake text (articles, reviews)
- Synthetic video (deepfakes)
- Cloned voices

How do we maintain trust in what we see and hear?

## Hallucination as Misinformation

LLMs confidently state false information:

```
User: Who won the 2028 Olympics 100m sprint?
Model: [Confidently states a plausible but wrong answer]
```

**Why it happens:**
- Models predict plausible text, not verified facts
- Training data has errors
- No mechanism to say "I don't know"

**Mitigation:**
- RAG (ground in retrieved documents)
- Uncertainty quantification
- Teaching models to abstain

## Deepfakes

### How They Work
- Face swapping (put one face on another's body)
- Lip sync (make someone say anything)
- Full synthesis (generate entire fake person)

### Detection Approaches

**Artifact-based:**
- Inconsistent blinking
- Lighting anomalies
- Edge artifacts around face

**Learning-based:**
- Train classifiers on real vs fake
- Cat-and-mouse with generators

**Problem**: Detection lags generation capability

## Content Authenticity

### C2PA (Coalition for Content Provenance and Authenticity)

Cryptographic provenance for media:
```
Image created → Sign with camera credentials →
Edited in Photoshop → Sign edit history →
Published → Verifiable chain of custody
```

### AI Watermarking

Embed invisible signals in AI-generated content:
```python
# Conceptual
def generate_image(prompt):
    image = model.generate(prompt)
    image = embed_watermark(image, "AI_GENERATED")
    return image
```

**Challenges:**
- Watermarks can be removed
- Open-source models won't include them
- False positives/negatives

## Platform Responsibilities

### Content Policies
- Label AI-generated content
- Remove harmful deepfakes
- Limit viral spread of misinformation

### Technical Measures
- Detection systems
- Provenance tracking
- Rate limiting

### Challenges
- Scale (billions of posts)
- Context dependence (satire vs misinformation)
- Global/cultural variation

## Election Integrity

AI-generated content poses risks:
- Fake candidate statements
- Fabricated "evidence"
- Micro-targeted disinformation

### Defenses
- Media literacy education
- Rapid fact-checking
- Pre-bunking (warn before exposure)
- Legal frameworks

## Building Trustworthy Systems

If you're building AI systems:
1. **Be transparent** about AI-generated content
2. **Implement safeguards** against misuse
3. **Support provenance** standards
4. **Monitor for abuse** of your systems

## Exercises

1. Test a deepfake detection tool on real/fake images
2. Implement a simple text watermarking scheme
3. Design a content authenticity system for a platform

## Resources

- C2PA specification: https://c2pa.org
- "The Deepfake Detection Challenge" (Facebook)
- Content Authenticity Initiative: https://contentauthenticity.org

## What's Next?

Module 7 covers **Dual Use & Misuse** — when AI capabilities enable harm.
