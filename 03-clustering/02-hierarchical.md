# Hierarchical Clustering

## Learning Objectives

- Understand hierarchical clustering concepts
- Build and interpret dendrograms
- Apply agglomerative and divisive clustering
- Choose appropriate linkage methods
- Compare hierarchical and K-Means clustering

## Introduction

Hierarchical clustering creates a tree of clusters (dendrogram) without requiring the number of clusters *K* to be specified in advance. It's particularly useful for understanding data structure, creating taxonomies, and when the number of clusters is unknown.

## Hierarchical Clustering Types

### Agglomerative (Bottom-Up)

**Most common approach**: Start with each point as its own cluster, then merge

```
Step 1: N clusters (each point is a cluster)
Step 2: Merge two closest clusters → N-1 clusters
Step 3: Merge two closest clusters → N-2 clusters
...
Step N: All points in one cluster
```

### Divisive (Top-Down)

**Less common**: Start with all points in one cluster, then split

```
Step 1: 1 cluster (all points)
Step 2: Split into 2 clusters
Step 3: Split one cluster → 3 clusters
...
Step N: N clusters (each point alone)
```

## Linkage Methods

### Single Linkage (Minimum)

**Distance between clusters** = minimum distance between any two points

\[
d(C_1, C_2) = \min_{x \in C_1, y \in C_2} d(x, y)
\]

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=50, centers=3, random_state=42)

# Single linkage
Z_single = linkage(X, method='single')

plt.figure(figsize=(10, 6))
dendrogram(Z_single)
plt.title('Single Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

**Characteristics**:
- ✅ Can handle non-elliptical shapes
- ✅ Good for elongated clusters
- ❌ Sensitive to noise (chaining effect)
- ❌ Can create unbalanced clusters

### Complete Linkage (Maximum)

**Distance between clusters** = maximum distance between any two points

\[
d(C_1, C_2) = \max_{x \in C_1, y \in C_2} d(x, y)
\]

```python
# Complete linkage
Z_complete = linkage(X, method='complete')

plt.figure(figsize=(10, 6))
dendrogram(Z_complete)
plt.title('Complete Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

**Characteristics**:
- ✅ Produces compact, balanced clusters
- ✅ Less sensitive to outliers than single linkage
- ❌ Breaks large clusters
- ❌ Assumes spherical clusters

### Average Linkage

**Distance between clusters** = average distance between all pairs

\[
d(C_1, C_2) = \frac{1}{|C_1| |C_2|} \sum_{x \in C_1} \sum_{y \in C_2} d(x, y)
\]

```python
# Average linkage
Z_average = linkage(X, method='average')

plt.figure(figsize=(10, 6))
dendrogram(Z_average)
plt.title('Average Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

**Characteristics**:
- ✅ Compromise between single and complete
- ✅ Generally good performance
- ✅ Most commonly used

### Ward's Method

**Minimize within-cluster variance** when merging

\[
\Delta(C_1, C_2) = \sum_{x \in C_1 \cup C_2} \|x - \mu_{C_1 \cup C_2}\|^2 - \sum_{x \in C_1} \|x - \mu_{C_1}\|^2 - \sum_{x \in C_2} \|x - \mu_{C_2}\|^2
\]

```python
# Ward linkage
Z_ward = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(Z_ward)
plt.title("Ward's Linkage Dendrogram")
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

**Characteristics**:
- ✅ Produces compact, spherical clusters
- ✅ Minimizes variance (similar to K-Means)
- ✅ Often best choice for general use
- ❌ Assumes spherical clusters
- ❌ Sensitive to outliers

## Comparing Linkage Methods

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Compare all methods
methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, method in enumerate(methods):
    Z = linkage(X, method=method)
    
    axes[idx].set_title(f'{method.capitalize()} Linkage', fontsize=14)
    dendrogram(Z, ax=axes[idx])
    axes[idx].set_xlabel('Sample Index')
    axes[idx].set_ylabel('Distance')

plt.tight_layout()
plt.show()
```

## Reading Dendrograms

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Simple example
X = np.array([[1, 2], [2, 3], [2, 2], [8, 7], [8, 8], [25, 80]])

Z = linkage(X, method='ward')

fig, ax = plt.subplots(figsize=(12, 6))
dend = dendrogram(Z, labels=['A', 'B', 'C', 'D', 'E', 'F'])

ax.set_title('How to Read a Dendrogram', fontsize=16)
ax.set_xlabel('Sample')
ax.set_ylabel('Distance')

# Add annotations
ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Cut here for 3 clusters')
ax.axhline(y=10, color='g', linestyle='--', linewidth=2, label='Cut here for 5 clusters')
ax.legend()

plt.show()

print("""Dendrogram Interpretation:
- Height of merge = dissimilarity between clusters
- Horizontal line = cluster merge
- Cut dendrogram at desired height to get K clusters
""")
```

## Implementing Agglomerative Clustering

### Using Scikit-Learn

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Agglomerative clustering
agg_cluster = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'  # or 'complete', 'average', 'single'
)

y_pred = agg_cluster.fit_predict(X)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
plt.title('True Labels')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6)
plt.title('Hierarchical Clustering')

plt.tight_layout()
plt.show()
```

### With Distance Threshold

```python
# Don't specify n_clusters, use distance threshold instead
agg_cluster = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=10,
    linkage='ward'
)

y_pred = agg_cluster.fit_predict(X)

print(f"Number of clusters formed: {agg_cluster.n_clusters_}")
print(f"Cluster sizes: {np.bincount(y_pred)}")
```

## Determining Number of Clusters

### Dendrogram Cut Height

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Compute linkage
Z = linkage(X, method='ward')

# Plot dendrogram with color threshold
plt.figure(figsize=(12, 6))
dend = dendrogram(Z, color_threshold=50)
plt.title('Dendrogram with Color Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.axhline(y=50, c='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Form flat clusters from threshold
clusters = fcluster(Z, t=50, criterion='distance')
print(f"Number of clusters: {len(np.unique(clusters))}")
print(f"Cluster assignments: {clusters}")
```

### Inconsistency Method

```python
from scipy.cluster.hierarchy import inconsistent

# Calculate inconsistency coefficient
incons = inconsistent(Z, d=2)

plt.figure(figsize=(10, 6))
plt.plot(incons[:, 3])
plt.xlabel('Cluster Merge Step')
plt.ylabel('Inconsistency Coefficient')
plt.title('Inconsistency Coefficient vs. Merge Step')
plt.grid(alpha=0.3)
plt.show()

# Large jumps indicate natural cluster boundaries
```

## Complete Example: Customer Segmentation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('data/customer_data.csv')

# Select features
features = ['age', 'income', 'spending_score', 'purchase_frequency']
X = df[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 7))
dend = dendrogram(
    Z,
    truncate_mode='lastp',  # Show only last p merged clusters
    p=30,
    leaf_font_size=10,
    show_contracted=True
)
plt.title('Customer Segmentation Dendrogram (Ward Linkage)', fontsize=16)
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline(y=20, color='r', linestyle='--', linewidth=2, label='Cut threshold')
plt.legend()
plt.show()

# Form clusters
threshold = 20
clusters = fcluster(Z, t=threshold, criterion='distance')
df['cluster'] = clusters

print(f"Number of clusters: {len(np.unique(clusters))}")
print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index())

# Analyze clusters
cluster_summary = df.groupby('cluster')[features].mean()
print("\nCluster Centers:")
print(cluster_summary)

# Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
for cluster in np.unique(clusters):
    mask = clusters == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=f'Cluster {cluster}', s=50, alpha=0.6)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments (Hierarchical Clustering)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Heatmap of cluster characteristics
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary.T, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Cluster Characteristics Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
```

## Hierarchical vs K-Means

### Comparison

| Aspect | Hierarchical | K-Means |
|--------|-------------|--------|
| **K specification** | Not required | Required |
| **Output** | Dendrogram + clusters | Clusters only |
| **Deterministic** | Yes | No (random init) |
| **Scalability** | O(n²) or O(n³) | O(n) |
| **Best for** | Small datasets, hierarchy | Large datasets |
| **Cluster shape** | Any shape (single linkage) | Spherical |
| **Re-run needed** | No (cut at different heights) | Yes (for different K) |

### When to Use Each

```python
def choose_clustering_method(n_samples, need_hierarchy, know_k, data_shape):
    """
    Decision helper for clustering method
    """
    if n_samples > 10000:
        return "K-Means (hierarchical too slow)"
    
    if need_hierarchy:
        return "Hierarchical (provides dendrogram)"
    
    if not know_k:
        return "Hierarchical (can explore different K)"
    
    if data_shape == 'non-spherical':
        return "Hierarchical with single linkage"
    
    return "K-Means (faster and efficient)"

# Examples
print(choose_clustering_method(100, True, False, 'any'))  
print(choose_clustering_method(100000, False, True, 'spherical'))
print(choose_clustering_method(500, False, False, 'elongated'))
```

## Advanced: Connectivity Constraints

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Generate data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Compute connectivity graph (spatial constraint)
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# Cluster with connectivity constraint
agg_with_connectivity = AgglomerativeClustering(
    n_clusters=2,
    connectivity=connectivity,
    linkage='average'
)

y_pred = agg_with_connectivity.fit_predict(X)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('True Labels (Moons)')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Hierarchical with Connectivity')

plt.tight_layout()
plt.show()
```

## Computational Complexity

### Time Complexity

| Method | Time Complexity |
|--------|----------------|
| **Single Linkage** | O(n²) |
| **Complete Linkage** | O(n² log n) |
| **Average Linkage** | O(n² log n) |
| **Ward's Method** | O(n² log n) |

### Space Complexity

- **Distance matrix**: O(n²)
- **Linkage matrix**: O(n)

### Scalability Improvements

```python
# For large datasets, use approximate methods
from sklearn.cluster import OPTICS

# OPTICS: Ordering Points To Identify Clustering Structure
# More scalable alternative to hierarchical clustering
optics = OPTICS(min_samples=10, xi=0.05)
y_pred = optics.fit_predict(X)
```

## Practical Tips

### 1. Choose Appropriate Linkage

```python
# Guidelines:
# - Ward: Default choice, assumes spherical clusters
# - Average: Robust alternative
# - Single: For elongated/non-spherical clusters
# - Complete: For compact, well-separated clusters
```

### 2. Scale Features

```python
from sklearn.preprocessing import StandardScaler

# Always scale for hierarchical clustering!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')  # Use scaled data
```

### 3. Handle Large Datasets

```python
# Option 1: Sample data
from sklearn.utils import resample
X_sample = resample(X, n_samples=1000, random_state=42)

# Option 2: Use K-Means for large data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Hierarchical clustering builds tree structure** (dendrogram)
2. **No need to specify K** in advance
3. **Linkage method matters** - Ward is often best
4. **Dendrogram reveals data structure** at multiple scales
5. **Computationally expensive** - O(n²) or worse
6. **Best for small-medium datasets** (<10,000 points)
7. **Feature scaling critical** for distance-based methods
8. **Cut dendrogram at different heights** for different K values
:::

## Further Reading

- Murtagh, F. & Contreras, P. (2012). "Algorithms for hierarchical clustering: an overview"
- Ward, J. (1963). "Hierarchical Grouping to Optimize an Objective Function"
- Scikit-learn Hierarchical Clustering: [sklearn documentation](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
