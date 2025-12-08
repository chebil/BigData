# DBSCAN: Density-Based Clustering

## Learning Objectives

- Understand density-based clustering concepts
- Implement DBSCAN algorithm
- Tune epsilon and min_samples parameters
- Identify outliers and noise points
- Handle arbitrary-shaped clusters
- Apply DBSCAN to spatial and anomaly detection problems

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that can:
- Find arbitrarily-shaped clusters
- Identify outliers as noise
- Work without specifying number of clusters
- Handle clusters of different densities

## Core Concepts

### Density-Based Definitions

**Epsilon (ε)**: Maximum radius of neighborhood

**MinPts**: Minimum number of points to form dense region

**Core Point**: Point with at least MinPts within ε radius

**Border Point**: Within ε of core point but has fewer than MinPts neighbors

**Noise Point**: Neither core nor border point (outlier)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Visualize concepts
fig, ax = plt.subplots(figsize=(10, 8))

# Sample points
np.random.seed(42)
points = np.random.randn(20, 2)

# Plot points
ax.scatter(points[:, 0], points[:, 1], s=100, c='lightblue', edgecolors='black', linewidths=2)

# Show epsilon radius for one point
center_point = points[0]
circle = plt.Circle(center_point, radius=0.5, fill=False, color='red', linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.plot(center_point[0], center_point[1], 'r*', markersize=20, label='Core Point (example)')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.legend()
ax.set_title(r'DBSCAN Concepts: $\epsilon$ = 0.5 radius', fontsize=14)
ax.grid(alpha=0.3)
plt.show()
```

## DBSCAN Algorithm

### Pseudocode

```
DBSCAN(D, eps, MinPts):
    C = 0  # Cluster counter
    
    for each point P in dataset D:
        if P is visited:
            continue
        
        mark P as visited
        NeighborPts = regionQuery(P, eps)
        
        if |NeighborPts| < MinPts:
            mark P as NOISE
        else:
            C = next cluster
            expandCluster(P, NeighborPts, C, eps, MinPts)

expandCluster(P, NeighborPts, C, eps, MinPts):
    add P to cluster C
    
    for each point P' in NeighborPts:
        if P' is not visited:
            mark P' as visited
            NeighborPts' = regionQuery(P', eps)
            
            if |NeighborPts'| >= MinPts:
                NeighborPts = NeighborPts + NeighborPts'
        
        if P' is not member of any cluster:
            add P' to cluster C

regionQuery(P, eps):
    return all points within eps distance of P
```

### Implementation from Scratch

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class DBSCANScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1 = noise
        
        # Find neighbors for all points
        nbrs = NearestNeighbors(radius=self.eps).fit(X)
        neighborhoods = nbrs.radius_neighbors(X, return_distance=False)
        
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already processed
                continue
            
            neighbors = neighborhoods[i]
            
            if len(neighbors) < self.min_samples:  # Noise
                continue
            
            # Start new cluster
            self.labels_[i] = cluster_id
            
            # Expand cluster
            seeds = set(neighbors)
            seeds.discard(i)
            
            while seeds:
                current = seeds.pop()
                
                if self.labels_[current] == -1:  # Was noise, now border
                    self.labels_[current] = cluster_id
                
                if self.labels_[current] != -1:  # Already in cluster
                    continue
                
                self.labels_[current] = cluster_id
                current_neighbors = neighborhoods[current]
                
                if len(current_neighbors) >= self.min_samples:  # Core point
                    seeds.update(current_neighbors)
            
            cluster_id += 1
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Test
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

dbscan_custom = DBSCANScratch(eps=0.3, min_samples=5)
labels = dbscan_custom.fit_predict(X)

print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of noise points: {list(labels).count(-1)}")
```

## Using Scikit-Learn

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Visualize
plt.figure(figsize=(10, 6))

# Plot clusters
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise points in black
        col = 'black'
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                s=50, alpha=0.6, label=label, edgecolors='black', linewidths=0.5)

plt.title(f'DBSCAN Clustering (eps={dbscan.eps}, min_samples={dbscan.min_samples})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Parameter Selection

### Choosing Epsilon (ε)

**K-distance Graph Method**:

```python
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

def plot_k_distance(X, k=5):
    """
    Plot k-distance graph to help choose epsilon
    """
    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort distances
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.xlabel('Data Points (sorted)')
    plt.title('K-distance Graph for Epsilon Selection')
    plt.grid(alpha=0.3)
    
    # Add annotation
    plt.axhline(y=0.3, color='r', linestyle='--', label='Suggested eps (elbow)')
    plt.legend()
    plt.show()
    
    return distances

# Generate data
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Plot k-distance graph
distances = plot_k_distance(X, k=5)

# The "elbow" in the curve suggests good epsilon value
print("Suggested epsilon: Look for the elbow point")
print(f"Median k-distance: {np.median(distances):.3f}")
print(f"75th percentile: {np.percentile(distances, 75):.3f}")
```

### Choosing MinPts

**Rule of Thumb**:
- MinPts ≥ number of dimensions + 1
- For 2D data: MinPts ≥ 3
- For noisy data: Use larger MinPts (e.g., 2 × dimensions)

```python
# Experiment with different parameters
from sklearn.metrics import silhouette_score

eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = [3, 5, 10, 15]

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Skip if only one cluster or all noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        
        # Calculate silhouette score (excluding noise)
        mask = labels != -1
        if mask.sum() > n_clusters:
            score = silhouette_score(X[mask], labels[mask])
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': list(labels).count(-1),
                'silhouette': score
            })

# Best parameters
results_df = pd.DataFrame(results)
best_params = results_df.loc[results_df['silhouette'].idxmax()]

print("Best parameters:")
print(best_params)

print("\nTop 5 parameter combinations:")
print(results_df.nlargest(5, 'silhouette'))
```

## Handling Different Cluster Densities

### HDBSCAN (Hierarchical DBSCAN)

```python
# For varying density clusters, use HDBSCAN
try:
    import hdbscan
    
    # Generate data with varying densities
    from sklearn.datasets import make_blobs
    X1, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.3, center_box=(0, 0), random_state=42)
    X2, _ = make_blobs(n_samples=100, centers=1, cluster_std=1.0, center_box=(5, 5), random_state=42)
    X = np.vstack([X1, X2])
    
    # HDBSCAN handles varying densities better
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title('HDBSCAN: Handles Varying Densities')
    plt.colorbar(label='Cluster')
    plt.show()
    
except ImportError:
    print("Install hdbscan: pip install hdbscan")
```

## Complete Example: Geospatial Clustering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Simulate taxi pickup locations
np.random.seed(42)

# Dense areas (airports, downtown)
dense_area_1 = np.random.randn(200, 2) * 0.5 + [40.7, -74.0]  # NYC area
dense_area_2 = np.random.randn(150, 2) * 0.3 + [40.75, -73.95]

# Sparse pickups
sparse = np.random.randn(50, 2) * 2 + [40.7, -74.0]

# Combine
locations = np.vstack([dense_area_1, dense_area_2, sparse])

df = pd.DataFrame(locations, columns=['latitude', 'longitude'])

# Standardize coordinates (important for DBSCAN)
scaler = StandardScaler()
locations_scaled = scaler.fit_transform(locations)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
labels = dbscan.fit_predict(locations_scaled)

df['cluster'] = labels

# Analysis
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of hotspots (clusters): {n_clusters}")
print(f"Number of isolated pickups (noise): {n_noise}")
print(f"Noise percentage: {n_noise/len(labels)*100:.1f}%")

# Visualize
plt.figure(figsize=(12, 8))

for cluster_id in set(labels):
    if cluster_id == -1:
        # Noise in black
        color = 'black'
        marker = 'x'
        label = 'Isolated'
        size = 20
    else:
        color = plt.cm.Spectral(cluster_id / n_clusters)
        marker = 'o'
        label = f'Hotspot {cluster_id}'
        size = 50
    
    mask = labels == cluster_id
    plt.scatter(df.loc[mask, 'longitude'], 
                df.loc[mask, 'latitude'],
                c=[color], marker=marker, s=size, 
                alpha=0.6, label=label, edgecolors='black', linewidths=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Taxi Pickup Hotspots (DBSCAN)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Hotspot statistics
for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    center_lat = cluster_data['latitude'].mean()
    center_lon = cluster_data['longitude'].mean()
    size = len(cluster_data)
    
    print(f"\nHotspot {cluster_id}:")
    print(f"  Center: ({center_lat:.4f}, {center_lon:.4f})")
    print(f"  Size: {size} pickups")
```

## Anomaly Detection with DBSCAN

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate normal data with anomalies
np.random.seed(42)

# Normal transactions
normal = np.random.randn(500, 2) * [20, 100] + [50, 500]

# Anomalies (fraud)
anomalies = np.random.uniform(low=[0, 0], high=[100, 1000], size=(20, 2))

# Combine
X = np.vstack([normal, anomalies])
true_labels = np.array([0]*500 + [1]*20)  # 0=normal, 1=anomaly

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

# Noise points are anomalies
detected_anomalies = (labels == -1)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print("Anomaly Detection Results:")
print(f"True anomalies: {true_labels.sum()}")
print(f"Detected anomalies: {detected_anomalies.sum()}")

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, detected_anomalies))

print("\nClassification Report:")
print(classification_report(true_labels, detected_anomalies, 
                           target_names=['Normal', 'Anomaly']))

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# True labels
ax1.scatter(X[true_labels==0, 0], X[true_labels==0, 1], 
            c='blue', label='Normal', alpha=0.6, s=30)
ax1.scatter(X[true_labels==1, 0], X[true_labels==1, 1], 
            c='red', label='Anomaly', alpha=0.8, s=100, marker='x')
ax1.set_title('True Labels')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(alpha=0.3)

# Detected
ax2.scatter(X[~detected_anomalies, 0], X[~detected_anomalies, 1],
            c='blue', label='Normal', alpha=0.6, s=30)
ax2.scatter(X[detected_anomalies, 0], X[detected_anomalies, 1],
            c='red', label='Detected Anomaly', alpha=0.8, s=100, marker='x')
ax2.set_title('DBSCAN Detection')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Advantages and Limitations

### Advantages ✅

1. **No need to specify K**
2. **Finds arbitrarily-shaped clusters**
3. **Robust to outliers** (identifies them as noise)
4. **Works with spatial data**
5. **Only two parameters** (eps, min_samples)

### Limitations ❌

1. **Struggles with varying densities**
2. **Sensitive to parameters**
3. **High-dimensional data** (curse of dimensionality)
4. **Border points ambiguous** (can belong to multiple clusters)
5. **Not deterministic** (border point assignment order-dependent)

## DBSCAN vs Other Methods

```python
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

# Generate different dataset shapes
datasets = [
    ('Moons', make_moons(n_samples=200, noise=0.05, random_state=0)),
    ('Circles', make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=0)),
    ('Blobs', make_blobs(n_samples=200, centers=3, random_state=0))
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, (name, (X, y)) in enumerate(datasets):
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)
    
    # Hierarchical
    agg = AgglomerativeClustering(n_clusters=2)
    y_agg = agg.fit_predict(X)
    
    # Plot
    axes[row, 0].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30)
    axes[row, 0].set_title(f'{name} - K-Means')
    
    axes[row, 1].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', s=30)
    axes[row, 1].set_title(f'{name} - DBSCAN')
    
    axes[row, 2].scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis', s=30)
    axes[row, 2].set_title(f'{name} - Hierarchical')

plt.tight_layout()
plt.show()

print("""Observations:
- K-Means: Struggles with non-spherical shapes
- DBSCAN: Handles arbitrary shapes well
- Hierarchical: Moderate performance on complex shapes
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **DBSCAN finds clusters based on density**
2. **No need to specify number of clusters**
3. **Identifies outliers as noise**
4. **Handles arbitrary cluster shapes**
5. **Two key parameters**: epsilon (ε) and min_samples
6. **Use k-distance graph** to choose epsilon
7. **Struggles with varying densities** - consider HDBSCAN
8. **Excellent for spatial data** and anomaly detection
:::

## Further Reading

- Ester, M. et al. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
- Schubert, E. et al. (2017). "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN"
- Campello, R. et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates" (HDBSCAN)
