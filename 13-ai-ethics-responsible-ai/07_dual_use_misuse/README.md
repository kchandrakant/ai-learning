# Module 7: Dual Use & Misuse

## The Dual-Use Dilemma

The same AI capabilities that help can also harm:

| Capability | Beneficial Use | Harmful Use |
|------------|---------------|-------------|
| Code generation | Developer productivity | Malware creation |
| Chemistry knowledge | Drug discovery | Weapon synthesis |
| Persuasion | Education | Manipulation |
| Face generation | Entertainment | Fraud |

## High-Risk Capability Areas

### CBRN Risks (Chemical, Biological, Radiological, Nuclear)

Can AI help create dangerous materials?

**Current assessment:**
- Models have chemistry/biology knowledge
- Detailed synthesis instructions are sometimes available
- But practical barriers remain high
- Models are tested and trained to refuse

**Mitigation:**
- Refuse dangerous queries
- Evaluate for uplift (does AI provide meaningful help?)
- Monitor for emerging risks

### Cyber Offense

AI can assist with:
- Vulnerability discovery
- Exploit development
- Phishing at scale
- Social engineering

**The asymmetry**: Defense also benefits, but offense may be easier to automate.

### Surveillance and Tracking

AI enables:
- Facial recognition at scale
- Movement tracking
- Behavior prediction
- Social network analysis

**Concerns:**
- Authoritarian use
- Chilling effects on speech/assembly
- Disproportionate impact on minorities

## Access Controls and Release Strategies

### Staged Release
1. Internal testing
2. Red team evaluation
3. Limited beta
4. Gradual public rollout

### Structured Access
- API access with rate limits
- No weights released
- Monitoring and enforcement

### Open Release
- Full weights available
- Community can audit
- But also misuse potential

### The Tradeoff
```
Closed: More control, less scrutiny, centralized risk
Open:   Less control, more scrutiny, distributed risk
```

## Responsible Disclosure

When you find a vulnerability or misuse potential:

1. **Document** clearly
2. **Report** to the model provider
3. **Allow time** for mitigation
4. **Coordinate** public disclosure

Don't: Publish exploits without giving time to fix.

## Evaluating Dual-Use Risks

### Marginal Risk Analysis
Does AI provide meaningful uplift over existing resources?

```
Risk = P(harm | AI access) - P(harm | no AI access)
```

If information is already widely available, AI adds little marginal risk.

### The "Determined Adversary" Test
Would a determined, resourced adversary be significantly helped?
- Nation-states have access to experts
- The question is mass enablement

## Building Responsibly

If you're developing AI systems:

1. **Threat model** your application
2. **Test for misuse** before release
3. **Implement safeguards** proportional to risk
4. **Monitor usage** for abuse
5. **Have a response plan** for incidents

## Exercises

1. Conduct a threat model for an AI coding assistant
2. Design red-teaming prompts to test for dangerous capabilities
3. Debate the open vs closed release tradeoffs

## Resources

- "Model Evaluation for Extreme Risks" (DeepMind, 2023)
- NIST AI Risk Management Framework
- Responsible AI Licenses (RAIL)

## What's Next?

Module 8 covers **AI Governance Frameworks** — the standards and best practices for responsible AI.
