# Step 11: Unsupervised Learning

## Learning Without Labels

Unsupervised learning finds patterns in data **without labels**. Instead of predicting a target, we discover structure, groupings, and representations hidden in the data.

---

## Supervised vs Unsupervised

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| Data | (X, y) pairs | Only X |
| Goal | Predict y | Find structure |
| Question | "What is this?" | "What groups exist?" |
| Evaluation | Compare to labels | Internal metrics |
| Examples | Classification, Regression | Clustering, Dim. Reduction |

---

## Part 1: Clustering

### What is Clustering?

Partition data into groups where items in the same group are similar.

```
Raw data:                    After clustering:
    *  *                        [A] [A]
  *    *  *                   [A]   [A] [A]
       *                           [A]
    
  *  *                          [B] [B]
    *   *                         [B]  [B]
```

**Use cases:**
- Customer segmentation
- Document grouping
- Image compression
- Anomaly detection
- Gene expression analysis

---

## K-Means Clustering

The most popular clustering algorithm.

### Algorithm

```
1. Choose K (number of clusters)
2. Initialize K centroids randomly
3. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Move centroid to mean of assigned points
```

### Visual

```
Step 1:              Step 2:              Step 3:
Random centroids     Assign points        Update centroids
    +    *  *           +A   A  A            A  A
  *    +   *          A    +   A           A  +  A
       *                    A                  A
  *  *                 B  B                 B  B
    *   *     +          B   B     +          B   B
                                                  +
+ = centroid         A/B = assignments     Repeat...
```

### Implementation

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Inertia: sum of squared distances to centroids
print(kmeans.inertia_)
```

---

## Choosing K

### Elbow Method

Plot inertia vs K, look for the "elbow":

```
Inertia
  |
  |\
  | \
  |  \____    <- "Elbow" suggests optimal K
  |       \_____
  |______________ K
```

### Silhouette Score

Measures how similar points are to their own cluster vs other clusters:

```
s = (b - a) / max(a, b)

a = mean distance to points in same cluster
b = mean distance to nearest other cluster

Range: -1 to 1
- Close to 1: Well clustered
- Close to 0: Overlapping clusters
- Negative: Wrong cluster assignment
```

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")
```

---

## K-Means Limitations

### 1. Must Specify K in Advance

```
K=2: Underclusters          K=10: Overclusters
   [AAAA]                     [A][B][C][D]
   [BBBB]                     [E][F][G][H]
```

### 2. Assumes Spherical Clusters

```
K-Means works:              K-Means FAILS:
    OOO                        OOOOOO
    OOO                            OOOOOO
    XXX                        XXXXXX
    XXX                            XXXXXX
  Spherical                  Elongated shapes
```

### 3. Sensitive to Initialization

- Different starting points → different results
- Solution: K-Means++ or run multiple times

### 4. Sensitive to Outliers

- Outliers can pull centroids away from true centers

---

## DBSCAN: Density-Based Clustering

Finds clusters based on **density**, not distance to centroids.

### Key Concepts

- **Core point:** Has at least `min_samples` neighbors within `eps` radius
- **Border point:** Within `eps` of a core point
- **Noise point:** Neither core nor border (outlier!)

### Parameters

```
eps: Radius to look for neighbors
min_samples: Minimum points to form dense region
```

### Advantages

- Finds arbitrarily shaped clusters
- Automatically identifies outliers
- No need to specify number of clusters

### Disadvantages

- Sensitive to `eps` and `min_samples`
- Struggles with varying density clusters

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# -1 labels are noise/outliers
n_noise = list(labels).count(-1)
```

---

## Hierarchical Clustering

Builds a tree (dendrogram) of clusters.

### Agglomerative (Bottom-Up)

```
1. Start: each point is its own cluster
2. Merge two closest clusters
3. Repeat until one cluster remains

Dendrogram:
    _____|_____
    |         |
  __|__     __|__
  |   |     |   |
  A   B     C   D
```

### Advantages

- No need to specify K upfront
- Produces hierarchy (cut at any level)
- Interpretable via dendrogram

### Disadvantages

- Slow: O(n^2) to O(n^3)
- Can't undo merges

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# For clustering
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X)

# For dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
```

---

## Clustering Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| K-Means | Fast, simple, scalable | Need K, spherical only | Large datasets, spherical clusters |
| DBSCAN | Arbitrary shapes, finds outliers | Sensitive to params | Spatial data, anomaly detection |
| Hierarchical | No K needed, dendrogram | Slow | Small datasets, need hierarchy |

---

## Part 2: Dimensionality Reduction

### Why Reduce Dimensions?

- **Visualization:** Project to 2D/3D
- **Remove noise:** Keep signal, drop noise
- **Speed:** Fewer features = faster training
- **Curse of dimensionality:** High-D spaces are sparse

---

## PCA (Principal Component Analysis)

Find directions of maximum variance and project onto them.

### Intuition

```
Original 2D data:            PCA finds:
    *   *                    Direction of max variance
  *   *   *                        /
    *   *                         /
                                 /
                          PC1 ---> 
```

### Algorithm

1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project onto top K eigenvectors

### Key Properties

- **Principal components are orthogonal**
- **PC1 has most variance, PC2 second most, etc.**
- **Linear transformation** (can be reversed)

### Implementation

```python
from sklearn.decomposition import PCA

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# How much variance explained?
print(pca.explained_variance_ratio_)
# e.g., [0.72, 0.23] means PC1 explains 72%, PC2 explains 23%

# Total variance retained
print(sum(pca.explained_variance_ratio_))  # 0.95 = 95% retained
```

### Choosing Number of Components

```python
# Keep components explaining 95% of variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(f"Components kept: {pca.n_components_}")
```

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for **visualization**.

### Key Idea

- Preserves **local structure:** nearby points stay nearby
- Does NOT preserve global structure: cluster distances meaningless

### When to Use

- Visualizing high-dimensional data (images, text embeddings, etc.)
- Exploring cluster structure
- **NOT** for feature extraction (use PCA instead)

### Implementation

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Warning: Only for visualization!
# Do NOT use X_tsne as features for another model
```

### Important Caveats

1. **Distances between clusters are NOT meaningful**
2. **Results depend on perplexity parameter**
3. **Different runs give different layouts**
4. **Slow for large datasets**
5. **Use for visualization only, not as features**

---

## PCA vs t-SNE

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Preserves | Global variance | Local structure |
| Speed | Fast | Slow |
| Deterministic | Yes | No |
| Invertible | Yes | No |
| Use as features | Yes | No |
| Interpretable | Yes (loadings) | No |

**Rule of thumb:**
- **PCA:** Preprocessing, feature reduction, when you need interpretability
- **t-SNE:** Visualization of high-dimensional data

---

## Evaluating Unsupervised Learning

### With Ground Truth Labels (If Available)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
```

### Without Labels (Most Common Case)

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)  # -1 to 1, higher better
```

**Visual Inspection:**
- Always plot your clusters!
- Domain expertise matters

---

## Practical Applications

| Task | Algorithm | Example |
|------|-----------|---------|
| Customer segmentation | K-Means | Group customers by behavior |
| Anomaly detection | DBSCAN | Fraud detection |
| Document clustering | K-Means, Hierarchical | News article grouping |
| Image compression | K-Means (on colors) | Reduce color palette |
| Visualization | t-SNE, UMAP | Visualize embeddings |
| Feature extraction | PCA | Reduce before classification |
| Noise reduction | PCA | Remove low-variance components |

---

## Common Pitfalls

1. **Forgetting to scale features** before clustering/PCA
2. **Using t-SNE features** for downstream ML (don't!)
3. **Over-interpreting t-SNE** cluster distances
4. **Choosing K arbitrarily** without elbow/silhouette analysis
5. **Not visualizing** your clusters
6. **Assuming clusters are meaningful** without domain validation

---

## Files

- `unsupervised.py` - K-Means from scratch, DBSCAN, hierarchical clustering, PCA, t-SNE examples

## Key Takeaways

1. **Clustering** groups similar data without labels
2. **K-Means** is fast but needs K and assumes spherical clusters
3. **DBSCAN** finds arbitrary shapes and identifies outliers
4. **Hierarchical** produces interpretable dendrograms
5. **PCA** finds directions of max variance (linear, fast)
6. **t-SNE** preserves local structure (non-linear, visualization only)
7. **Silhouette score** evaluates clusters without labels
8. **Always visualize** - domain knowledge guides interpretation

## What's Next?

Step 12: **Model Evaluation** - metrics, cross-validation, and understanding when your model is actually working.
