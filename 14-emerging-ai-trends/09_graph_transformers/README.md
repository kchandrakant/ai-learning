# Module 9: Graph Transformers

## Graphs in ML

Graphs are everywhere:
- Social networks (users, connections)
- Molecules (atoms, bonds)
- Knowledge graphs (entities, relations)
- Code (functions, calls)

**Challenge**: Standard transformers expect sequences, not graphs.

## Background: Message Passing

Traditional graph neural networks use message passing:

```python
def message_passing(node_features, edges):
    for node in nodes:
        # Aggregate messages from neighbors
        messages = [node_features[neighbor] for neighbor in node.neighbors]
        aggregated = aggregate(messages)  # sum, mean, max
        
        # Update node
        node_features[node] = update(node_features[node], aggregated)
    
    return node_features
```

**Limitation**: Only local neighborhood information per layer.

## Graph Attention Networks (GAT)

Add attention to message passing:

```python
def graph_attention(node_features, edges):
    for node in nodes:
        # Compute attention weights to neighbors
        attentions = []
        for neighbor in node.neighbors:
            a = attention_score(node_features[node], node_features[neighbor])
            attentions.append(a)
        attentions = softmax(attentions)
        
        # Weighted aggregation
        message = sum(a * node_features[n] for a, n in zip(attentions, neighbors))
        node_features[node] = update(node_features[node], message)
```

**Still limited**: Attention only over direct neighbors.

## Full Graph Transformers

**Idea**: Every node attends to every other node!

```python
def graph_transformer(node_features, edge_features):
    # Self-attention over ALL nodes
    attention_scores = compute_attention(node_features, node_features)
    
    # Optionally incorporate edge information
    if edge_features:
        attention_scores += edge_bias(edge_features)
    
    output = attention_scores @ node_features
    return output
```

**Challenge**: O(n²) in number of nodes — expensive for large graphs.

## Positional Encodings for Graphs

Graphs don't have natural positions. Options:

### Laplacian Eigenvectors
Use eigenvectors of the graph Laplacian:
```python
L = D - A  # Laplacian = Degree - Adjacency
eigenvalues, eigenvectors = eig(L)
positional_encoding = eigenvectors[:, :k]
```

### Random Walk Encodings
Encode local structure via random walks:
```python
def random_walk_encoding(node, steps=20):
    return [P_random_walk(node → node in k steps) for k in range(steps)]
```

## GraphGPS (2022)

State-of-the-art hybrid approach:

```
Node features → Positional encoding →
                                      ↓
        ┌─ Local: Message passing (MPNN) ─┐
        │                                   │ → Add → FFN → Output
        └─ Global: Transformer attention ──┘
```

**Key insight**: Combine local (MPNN) and global (Transformer) information.

## Scalability

Full attention is O(n²). For large graphs:

### Sparse Attention
Only attend to nearby nodes (by graph distance)

### Sampling
Subsample nodes for attention

### Hierarchical
Build hierarchy of graph summaries

## Applications

| Domain | Graph Type | Task |
|--------|------------|------|
| Chemistry | Molecular graphs | Property prediction |
| Drug discovery | Protein-ligand | Binding affinity |
| Social | User networks | Link prediction |
| Knowledge | KGs | Reasoning |
| Code | AST/CFG | Bug detection |

## Practical Implementation

```python
import torch_geometric as pyg
from torch_geometric.nn import TransformerConv

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, 64, heads=heads)
        self.conv2 = TransformerConv(64 * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

## Exercises

1. Implement GAT on a node classification task
2. Compare message passing vs graph transformer on molecular data
3. Experiment with different positional encodings

## Resources

- "Recipe for a General, Powerful, Scalable Graph Transformer (GPS)" (2022)
- "Attention is All You Need for Graphs" (2021)
- PyTorch Geometric documentation

## What's Next?

Module 10 covers **Mixture of Experts** — sparse scaling for transformers.
