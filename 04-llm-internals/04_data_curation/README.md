# Module 4: Data Curation

Building the training dataset.

## Overview

Data quality is the secret sauce. The same model architecture with better data significantly outperforms.

## Key Topics

### The Data Pipeline
```
Raw Web (Common Crawl)
    ↓
Language Filtering
    ↓
Deduplication (exact + fuzzy)
    ↓
Quality Filtering (perplexity, classifiers)
    ↓
PII/Toxicity Removal
    ↓
Data Mixing (web + books + code + curated)
    ↓
Final Training Set
```

### Deduplication
- **Exact dedup**: Hash-based, removes identical documents
- **Fuzzy dedup**: MinHash/SimHash, removes near-duplicates
- Impact: Significant quality improvement, training stability

### Quality Filtering
- **Perplexity filtering**: Remove text a small LM finds "surprising"
- **Classifier filtering**: Train classifier on high/low quality examples
- **Heuristics**: Length, formatting, language detection

### Data Mixing
```
Typical mix:
- Web text: 70-80%
- Books: 5-10%
- Code: 5-10%
- Wikipedia/curated: 5-10%
- Math/science: small but high weight
```

### Contamination
- Test set leakage into training data
- Benchmark results may be inflated
- Detection and prevention methods

## Exercises

1. Build a deduplication pipeline
2. Train a quality classifier
3. Analyze data distribution of a public dataset

## Key Insight

Models are only as good as their data. Curation often matters more than architecture.
