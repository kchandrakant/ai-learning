# Step 11: Unsupervised Learning

## Learning Without Labels

Find structure in data without being told the answers.

## K-Means Clustering

Group data into K clusters.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Cluster centers
print(kmeans.cluster_centers_)
```

### How It Works
1. Initialize K random centers
2. Assign each point to nearest center
3. Update centers to mean of assigned points
4. Repeat until convergence

### Choosing K
```python
# Elbow method: plot inertia vs K
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)

# Look for "elbow" in the plot
```

## Hierarchical Clustering

Build a tree of clusters.

```python
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
clusters = hierarchical.fit_predict(X)
```

**Dendrogram:** Visualizes the hierarchy.

## Principal Component Analysis (PCA)

Reduce dimensions while preserving variance.

```python
from sklearn.decomposition import PCA

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance
print(pca.explained_variance_ratio_)
```

### What PCA Does
1. Find directions of maximum variance
2. Project data onto these directions
3. First component = most variance, etc.

### Uses
- Visualization (reduce to 2D/3D)
- Preprocessing (remove noise)
- Feature compression

## t-SNE and UMAP (Visualization)

Non-linear dimensionality reduction for visualization.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# UMAP (install: pip install umap-learn)
import umap
reducer = umap.UMAP()
X_2d = reducer.fit_transform(X)
```

**t-SNE/UMAP:** Great for visualizing clusters, not for preprocessing.

## Anomaly Detection

Find outliers/unusual data points.

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1)
predictions = iso_forest.fit_predict(X)
# -1 = anomaly, 1 = normal
```

## Summary

| Method | Use Case |
|--------|----------|
| K-Means | Simple clustering |
| Hierarchical | Cluster hierarchy |
| PCA | Dimensionality reduction |
| t-SNE/UMAP | Visualization |
| Isolation Forest | Anomaly detection |

## Files

- `unsupervised.py` - Clustering and dimensionality reduction

## Key Takeaways

1. Unsupervised = no labels
2. K-Means: assign to K clusters
3. PCA: find principal components
4. t-SNE/UMAP: visualize high-dim data
5. Useful for exploration and preprocessing

## What's Next?

Step 12: **Model Evaluation** — choosing and comparing models.
