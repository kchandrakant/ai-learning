# Module 10: Multimodal RAG

Retrieval with images and text.

## Overview

RAG traditionally retrieves text. Multimodal RAG retrieves images, figures, and documents alongside text.

## Key Topics

### Image Retrieval
```
CLIP-based retrieval:
- Encode all images with CLIP image encoder
- Encode query with CLIP text encoder
- Find nearest neighbor images

"A sunset over mountains" → retrieve matching photos
```

### Document Retrieval with Figures
```
Problem: Documents contain images, charts, tables
Traditional RAG: Only indexes text, loses visual info

Solution: Index both text AND visual elements
- Extract figures
- Embed figures with CLIP
- Hybrid search: text + image embeddings
```

### ColPali & Late Interaction
```
ColPali approach:
- Treat document pages as images
- Generate multiple embeddings per page
- Late interaction for fine-grained matching

Better than separate text/image indexing
```

### Multimodal RAG Pipeline
```
Document → Extract text + images → Embed both
    ↓
Query → Embed → Search text AND images
    ↓
Retrieve relevant content → VLM generates answer
```

### Production Considerations
```
- Index size: Images need more storage
- Latency: Image embedding slower
- Cost: More compute for multimodal search
- Quality: Better answers for visual content
```

## Exercises

1. Build CLIP-based image search
2. Implement multimodal document indexing
3. Create RAG system for PDFs with figures

## Key Insight

Multimodal RAG captures information that text-only RAG misses (charts, diagrams, photos).
