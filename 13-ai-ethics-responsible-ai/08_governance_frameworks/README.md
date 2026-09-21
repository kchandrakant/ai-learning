# Module 8: AI Governance Frameworks

## Why Governance?

As AI systems become more impactful, organizations need:
- Clear accountability for AI decisions
- Processes for risk assessment
- Documentation for compliance and audits
- Mechanisms for oversight

## NIST AI Risk Management Framework

The US National Institute of Standards and Technology framework (2023).

### Core Functions

**GOVERN**: Establish accountability
- Define roles and responsibilities
- Create oversight mechanisms
- Allocate resources for AI risk management

**MAP**: Understand context
- Identify AI systems and their purposes
- Document intended uses and limitations
- Assess potential impacts

**MEASURE**: Evaluate risks
- Test for accuracy, bias, security
- Conduct regular assessments
- Track metrics over time

**MANAGE**: Take action
- Prioritize identified risks
- Implement mitigations
- Monitor effectiveness

### Implementation Tiers
- Partial (ad hoc)
- Risk Informed (some processes)
- Repeatable (consistent processes)
- Adaptive (continuous improvement)

## ISO/IEC AI Standards

### ISO/IEC 42001: AI Management System
- Requirements for establishing, implementing, maintaining AI systems
- Similar structure to other ISO management standards (9001, 27001)

### ISO/IEC 23894: AI Risk Management
- Guidance on managing AI-related risks
- Lifecycle approach

## Documentation Standards

### Model Cards

Standardized documentation for ML models (Mitchell et al., 2019):

```markdown
## Model Details
- Developer: 
- Model type:
- Training data:

## Intended Use
- Primary use cases:
- Out-of-scope uses:

## Metrics
- Performance metrics:
- Disaggregated by group:

## Limitations
- Known limitations:
- Failure modes:

## Ethical Considerations
- Sensitive uses:
- Risks and harms:
```

### Datasheets for Datasets

Standardized documentation for datasets (Gebru et al., 2021):

- Motivation (why was it created?)
- Composition (what's in it?)
- Collection process (how was data gathered?)
- Uses (what should/shouldn't it be used for?)
- Distribution (how is it shared?)
- Maintenance (who maintains it?)

## Internal Governance Structures

### AI Ethics Boards
- Cross-functional review of high-risk AI projects
- Typically include: technical, legal, ethics, business, external

### Review Processes
```
New AI project → Risk assessment → If high-risk → Ethics board review
                                 → If approved → Ongoing monitoring
```

### Incident Response
- Clear escalation paths
- Post-incident review
- Public communication plan

## Audit and Assessment

### Internal Audits
- Regular review of AI systems
- Compliance with policies
- Performance against metrics

### External Audits
- Third-party assessment
- May be required by regulation
- Provides independent verification

### Algorithmic Impact Assessments
- Before deployment: assess potential impacts
- Ongoing: monitor actual impacts
- Standardized frameworks emerging

## Exercises

1. Create a model card for a project you've worked on
2. Map your organization's AI systems to NIST AI RMF categories
3. Design an AI review process for a hypothetical organization

## Resources

- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- Model Cards: https://modelcards.withgoogle.com
- Datasheets for Datasets: https://arxiv.org/abs/1803.09010

## What's Next?

Module 9 covers **Regulation & Compliance** — the legal requirements for AI systems.
