"""
Step 11: Unsupervised Learning
==============================

This module covers:
1. K-Means clustering (from scratch and sklearn)
2. Choosing K with Elbow method and Silhouette score
3. Hierarchical clustering with dendrograms
4. DBSCAN for density-based clustering
5. PCA for dimensionality reduction
6. t-SNE for visualization

Run this file to see all concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_digits
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("STEP 11: UNSUPERVISED LEARNING")
print("=" * 60)


# =============================================================================
# PART 1: K-Means Clustering From Scratch
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: K-Means From Scratch")
print("=" * 60)


class KMeansScratch:
    """K-Means clustering implemented from scratch."""
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)
            
            # Check for convergence
            if self.labels is not None and np.all(new_labels == self.labels):
                print(f"  Converged at iteration {iteration}")
                break
            
            self.labels = new_labels
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)
        
        return self
    
    def _compute_distances(self, X):
        """Compute distance from each point to each centroid."""
        distances = np.zeros((len(X), self.n_clusters))
        for k, centroid in enumerate(self.centroids):
            distances[:, k] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def inertia(self, X):
        """Sum of squared distances to nearest centroid."""
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)


# Create sample data
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2,
                               cluster_std=0.8, random_state=42)

# Test our K-Means
print("\n--- Testing Our K-Means ---")
kmeans_scratch = KMeansScratch(n_clusters=4, random_state=42)
kmeans_scratch.fit(X_blobs)
print(f"Inertia: {kmeans_scratch.inertia(X_blobs):.2f}")

# Compare with sklearn
from sklearn.cluster import KMeans

kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_sklearn.fit(X_blobs)
print(f"Sklearn Inertia: {kmeans_sklearn.inertia_:.2f}")


# =============================================================================
# PART 2: Visualizing K-Means Clustering
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Visualizing K-Means")
print("=" * 60)


def plot_clusters(X, labels, centroids, title, ax):
    """Plot clustered data with centroids."""
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                        alpha=0.6, edgecolors='black', s=50)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
              s=200, edgecolors='black', linewidths=2, label='Centroids')
    ax.set_title(title)
    ax.legend()
    return scatter


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Different K values
for ax, k in zip(axes, [2, 4, 6]):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_blobs)
    plot_clusters(X_blobs, labels, kmeans.cluster_centers_, f'K = {k}', ax)

plt.suptitle('K-Means: Effect of K on Clustering', fontsize=14)
plt.tight_layout()
plt.savefig('kmeans_different_k.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: kmeans_different_k.png")


# =============================================================================
# PART 3: Choosing K - Elbow Method and Silhouette Score
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Choosing K")
print("=" * 60)

from sklearn.metrics import silhouette_score

K_range = range(2, 10)
inertias = []
silhouette_scores = []

print("\n--- Elbow Method & Silhouette Score ---")
print(f"{'K':>5} {'Inertia':>15} {'Silhouette':>15}")
print("-" * 40)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_blobs)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_blobs, labels)
    silhouette_scores.append(sil_score)
    print(f"{k:>5} {kmeans.inertia_:>15.2f} {sil_score:>15.4f}")

# Plot both methods
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=4, color='r', linestyle='--', label='Elbow at K=4')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score Method')
axes[1].axvline(x=4, color='r', linestyle='--', label='Best at K=4')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('choosing_k.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: choosing_k.png")


# =============================================================================
# PART 4: K-Means Limitations
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: K-Means Limitations")
print("=" * 60)

# Create non-spherical data
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# K-Means fails on these!
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Moons
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = kmeans_moons.fit_predict(X_moons)

axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', edgecolors='black')
axes[0, 0].set_title('Moons: True Labels')

axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', edgecolors='black')
axes[0, 1].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black')
axes[0, 1].set_title('Moons: K-Means (Fails!)')

# Circles
kmeans_circles = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_circles = kmeans_circles.fit_predict(X_circles)

axes[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', edgecolors='black')
axes[1, 0].set_title('Circles: True Labels')

axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_circles, cmap='viridis', edgecolors='black')
axes[1, 1].scatter(kmeans_circles.cluster_centers_[:, 0], kmeans_circles.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black')
axes[1, 1].set_title('Circles: K-Means (Fails!)')

plt.suptitle('K-Means Fails on Non-Spherical Clusters', fontsize=14)
plt.tight_layout()
plt.savefig('kmeans_limitations.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nK-Means assumes spherical clusters - fails on moons and circles!")
print("Saved: kmeans_limitations.png")


# =============================================================================
# PART 5: DBSCAN - Density-Based Clustering
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: DBSCAN - Density-Based Clustering")
print("=" * 60)

from sklearn.cluster import DBSCAN

print("\nDBSCAN can handle non-spherical clusters!")

# DBSCAN on moons and circles
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Moons with DBSCAN
dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan_moons = dbscan_moons.fit_predict(X_moons)

axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', edgecolors='black')
axes[0, 0].set_title('Moons: True Labels')

axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan_moons, cmap='viridis', edgecolors='black')
axes[0, 1].set_title('Moons: DBSCAN (Works!)')

# Circles with DBSCAN
dbscan_circles = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan_circles = dbscan_circles.fit_predict(X_circles)

axes[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', edgecolors='black')
axes[1, 0].set_title('Circles: True Labels')

axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_dbscan_circles, cmap='viridis', edgecolors='black')
axes[1, 1].set_title('Circles: DBSCAN (Works!)')

plt.suptitle('DBSCAN Handles Non-Spherical Clusters', fontsize=14)
plt.tight_layout()
plt.savefig('dbscan_success.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: dbscan_success.png")

# DBSCAN parameters
print("\n--- DBSCAN Parameter Effects ---")
print("eps = radius to look for neighbors")
print("min_samples = minimum points to form dense region")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

eps_values = [0.1, 0.2, 0.5]
min_samples_values = [3, 10]

for i, min_samples in enumerate(min_samples_values):
    for j, eps in enumerate(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_moons)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        axes[i, j].scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis', edgecolors='black')
        axes[i, j].set_title(f'eps={eps}, min_samples={min_samples}\n'
                             f'Clusters: {n_clusters}, Noise: {n_noise}')

plt.suptitle('DBSCAN: Effect of Parameters', fontsize=14)
plt.tight_layout()
plt.savefig('dbscan_parameters.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: dbscan_parameters.png")


# =============================================================================
# PART 6: Hierarchical Clustering
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Hierarchical Clustering")
print("=" * 60)

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create smaller dataset for dendrogram visualization
X_small, _ = make_blobs(n_samples=50, centers=3, n_features=2,
                         cluster_std=1.0, random_state=42)

# Compute linkage matrix for dendrogram
linkage_matrix = linkage(X_small, method='ward')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dendrogram
dendrogram(linkage_matrix, ax=axes[0], truncate_mode='level', p=5)
axes[0].set_title('Hierarchical Clustering Dendrogram')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Distance')

# Clustering result
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_small)
axes[1].scatter(X_small[:, 0], X_small[:, 1], c=labels_agg, cmap='viridis', 
               edgecolors='black', s=100)
axes[1].set_title('Agglomerative Clustering (K=3)')

plt.tight_layout()
plt.savefig('hierarchical_clustering.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: hierarchical_clustering.png")


# =============================================================================
# PART 7: PCA - Dimensionality Reduction
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: PCA - Dimensionality Reduction")
print("=" * 60)

from sklearn.decomposition import PCA

# Load iris dataset (4 features)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\nIris dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")

# Apply PCA
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)

print(f"After PCA: {X_iris_pca.shape[1]} features")
print(f"\nExplained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.1f}%)")
print(f"  Total: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original 2 features
axes[0].scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='viridis', edgecolors='black')
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])
axes[0].set_title('Original: First 2 Features')

# PCA 2 components
axes[1].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, cmap='viridis', edgecolors='black')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('PCA: 2 Principal Components')

plt.suptitle('PCA on Iris Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('pca_iris.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: pca_iris.png")

# Explained variance curve
pca_full = PCA()
pca_full.fit(X_iris)

plt.figure(figsize=(8, 5))
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.6, label='Individual')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance,
         where='mid', color='red', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance by Component')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: pca_variance.png")


# =============================================================================
# PART 8: t-SNE for Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: t-SNE for Visualization")
print("=" * 60)

from sklearn.manifold import TSNE

# Load digits dataset (64 features - 8x8 images)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"\nDigits dataset: {X_digits.shape[0]} samples, {X_digits.shape[1]} features")

# Apply t-SNE
print("Applying t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_digits_tsne = tsne.fit_transform(X_digits)

# Compare PCA and t-SNE
pca_digits = PCA(n_components=2)
X_digits_pca = pca_digits.fit_transform(X_digits)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
scatter1 = axes[0].scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], 
                           c=y_digits, cmap='tab10', alpha=0.6, s=10)
axes[0].set_title('PCA on Digits')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE
scatter2 = axes[1].scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1], 
                           c=y_digits, cmap='tab10', alpha=0.6, s=10)
axes[1].set_title('t-SNE on Digits')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.colorbar(scatter2, ax=axes[1], label='Digit')
plt.suptitle('PCA vs t-SNE on Handwritten Digits', fontsize=14)
plt.tight_layout()
plt.savefig('pca_vs_tsne.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: pca_vs_tsne.png")

print("\nt-SNE creates much better separated clusters for visualization!")
print("But remember: t-SNE distances between clusters are NOT meaningful.")


# =============================================================================
# PART 9: Clustering Comparison Summary
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: Algorithm Comparison")
print("=" * 60)

print("""
CLUSTERING ALGORITHMS COMPARISON:

| Algorithm    | Pros                          | Cons                           |
|--------------|-------------------------------|--------------------------------|
| K-Means      | Fast, simple, scalable        | Need K, spherical clusters only|
| DBSCAN       | Arbitrary shapes, finds noise | Sensitive to eps/min_samples   |
| Hierarchical | No K needed, dendrogram       | Slow O(n^2) or O(n^3)          |

DIMENSIONALITY REDUCTION COMPARISON:

| Algorithm | Type       | Preserves       | Speed | Use For              |
|-----------|------------|-----------------|-------|----------------------|
| PCA       | Linear     | Global variance | Fast  | Preprocessing, features |
| t-SNE     | Non-linear | Local structure | Slow  | Visualization only   |
""")


# =============================================================================
# PART 10: Practical Example - Customer Segmentation
# =============================================================================

print("\n" + "=" * 60)
print("PART 10: Practical Example - Customer Segmentation")
print("=" * 60)

# Simulate customer data
n_customers = 500
np.random.seed(42)

# Features: spending_score, annual_income (scaled 0-100)
# Create 4 customer segments
segment_centers = np.array([
    [20, 80],   # High income, low spending
    [80, 80],   # High income, high spending
    [20, 20],   # Low income, low spending
    [80, 20],   # Low income, high spending (rare)
])
segment_sizes = [150, 150, 150, 50]

X_customers = []
for center, size in zip(segment_centers, segment_sizes):
    cluster = np.random.randn(size, 2) * 10 + center
    X_customers.append(cluster)
X_customers = np.vstack(X_customers)
X_customers = np.clip(X_customers, 0, 100)  # Keep in 0-100 range

# Apply K-Means
kmeans_customers = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_labels = kmeans_customers.fit_predict(X_customers)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_customers[:, 0], X_customers[:, 1], 
                     c=customer_labels, cmap='viridis', alpha=0.6, edgecolors='black')
plt.scatter(kmeans_customers.cluster_centers_[:, 0], 
           kmeans_customers.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Centroids')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation with K-Means')
plt.colorbar(scatter, label='Segment')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_segmentation.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: customer_segmentation.png")

# Analyze segments
print("\n--- Customer Segment Analysis ---")
for i in range(4):
    mask = customer_labels == i
    segment_data = X_customers[mask]
    print(f"\nSegment {i} ({np.sum(mask)} customers):")
    print(f"  Avg Income: {segment_data[:, 0].mean():.1f}")
    print(f"  Avg Spending: {segment_data[:, 1].mean():.1f}")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. CLUSTERING
   - Groups similar data without labels
   - K-Means: Fast, needs K, spherical clusters
   - DBSCAN: Arbitrary shapes, finds outliers
   - Hierarchical: Produces dendrogram

2. CHOOSING K
   - Elbow method: Look for bend in inertia plot
   - Silhouette score: Higher is better
   - Domain knowledge often most important

3. DIMENSIONALITY REDUCTION
   - PCA: Linear, preserves global variance
   - t-SNE: Non-linear, preserves local structure
   - PCA for features, t-SNE for visualization only

4. EVALUATION (Without Labels)
   - Silhouette score: -1 to 1 (higher better)
   - Inertia: Lower better (but always decreases)
   - Visual inspection is crucial!

5. PRACTICAL TIPS
   - Always visualize your clusters
   - Try multiple algorithms
   - Scale features before clustering
   - Domain knowledge guides interpretation
""")

print("\n" + "=" * 60)
print("Module 11 Complete!")
print("=" * 60)
