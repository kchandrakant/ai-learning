# Module 9: Regulation & Compliance

## The Regulatory Landscape

AI regulation is evolving rapidly. Key frameworks:

| Jurisdiction | Framework | Status |
|--------------|-----------|--------|
| EU | AI Act | In force (2024) |
| US | Executive Order + sector rules | Evolving |
| China | Multiple AI regulations | In force |
| UK | Pro-innovation approach | Principles-based |

## EU AI Act (Regulation 2024/1689)

The world's most comprehensive AI regulation.

### Risk-Based Approach

**Unacceptable Risk (Banned):**
- Social scoring by governments
- Real-time remote biometric ID in public (with exceptions)
- Subliminal manipulation
- Exploitation of vulnerabilities

**High Risk (Strict Requirements):**
- Biometric identification
- Critical infrastructure management
- Educational/vocational access
- Employment decisions
- Essential services access
- Law enforcement
- Migration and border control

**Limited Risk (Transparency):**
- Chatbots must disclose AI nature
- Deepfakes must be labeled
- Emotion recognition must inform users

**Minimal Risk (No Requirements):**
- Most AI applications
- Spam filters, video games, etc.

### High-Risk Requirements

If your system is high-risk:

1. **Risk management system**: Continuous identification and mitigation
2. **Data governance**: Quality, relevance, representativeness
3. **Technical documentation**: Comprehensive records
4. **Record-keeping**: Logs for traceability
5. **Transparency**: Clear info to users
6. **Human oversight**: Ability for human intervention
7. **Accuracy, robustness, security**: Meet appropriate levels
8. **Conformity assessment**: Before market placement

### Penalties
- Up to €35 million or 7% global turnover for prohibited practices
- Up to €15 million or 3% for other violations

## US Approach

### Executive Order on AI Safety (October 2023)

For federal agencies and contractors:
- Safety testing for dual-use foundation models
- Reporting requirements
- Red-teaming mandates
- Watermarking guidance

### Sector-Specific Regulation

| Sector | Regulator | Requirements |
|--------|-----------|--------------|
| Finance | CFPB, SEC | Fair lending, model risk management |
| Healthcare | FDA | Medical device approval for AI |
| Employment | EEOC | Non-discrimination |
| Housing | HUD | Fair housing compliance |

### State Laws
- NYC Local Law 144: Bias audits for hiring AI
- California, Illinois: Various AI requirements

## China's Approach

Multiple targeted regulations:
- **Recommendation algorithms** (2022): Must offer opt-out, can't discriminate on price
- **Deep synthesis** (2023): Watermarking requirements
- **Generative AI** (2023): Licensing, content requirements

## Compliance Strategies

### 1. Classification
First, determine what your system is:
- Is it an AI system under the regulation?
- What risk category?
- What sector-specific rules apply?

### 2. Gap Analysis
Compare current state to requirements:
```
Requirement          | Current State | Gap | Action Needed
Documentation        | Partial       | Yes | Create model cards
Bias testing         | None          | Yes | Implement Fairlearn
Human oversight      | Limited       | Yes | Add review step
```

### 3. Implementation
Prioritize and implement changes:
- Technical controls
- Process changes
- Documentation
- Training

### 4. Ongoing Compliance
- Regular audits
- Monitoring for regulatory changes
- Incident response

## Conformity Assessment

For EU high-risk systems:

**Self-assessment**: Most high-risk systems
**Third-party assessment**: Biometric systems, critical infrastructure

Documentation required:
- Technical documentation
- Quality management system
- EU declaration of conformity
- CE marking

## Exercises

1. Classify an AI system under the EU AI Act risk tiers
2. Create a compliance checklist for a high-risk application
3. Compare EU, US, and China approaches on a specific use case

## Resources

- EU AI Act text: https://eur-lex.europa.eu/eli/reg/2024/1689
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- Future of Life Institute AI Policy tracker

## What's Next?

Module 10 covers **Environmental Impact** — the carbon footprint of AI.
