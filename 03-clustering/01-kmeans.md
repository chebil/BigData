# K-Means Clustering

## Learning Objectives

- Understand the K-Means algorithm
- Implement K-Means from scratch and with scikit-learn
- Determine optimal number of clusters
- Apply K-Means to real-world problems
- Scale K-Means to Big Data with Spark

## Algorithm Overview

### K-Means Goal

Partition *n* observations into *K* clusters where each observation belongs to the cluster with the nearest centroid.

**Objective**: Minimize within-cluster sum of squares (WCSS)

\[
\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2
\]

where:
- \(C_k\) = cluster \(k\)
- \(\mu_k\) = centroid of cluster \(k\)
- \(\|x - \mu_k\|\) = Euclidean distance

### K-Means Algorithm Steps

```
1. INITIALIZE: Randomly select K centroids

2. REPEAT until convergence:
   
   a. ASSIGN: Assign each point to nearest centroid
      
      cluster[i] = argmin_k ||x_i - centroid_k||^2
   
   b. UPDATE: Recalculate centroids as mean of assigned points
      
      centroid_k = mean(points in cluster k)

3. CONVERGE: Stop when centroids don't change (or max iterations)
```

### Visual Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, 
                       cluster_std=0.60, random_state=0)

plt.figure(figsize=(12, 4))

# Before clustering
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
plt.title('Data Before Clustering')

# K-Means clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_pred = kmeans.fit_predict(X)

# After clustering
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', edgecolors='black', linewidths=2,
            label='Centroids')
plt.title('K-Means Clustering (K=4)')
plt.legend()

# True labels (for comparison)
plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis', alpha=0.6)
plt.title('True Labels')

plt.tight_layout()
plt.show()
```

## Implementation from Scratch

```python
import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        """Fit K-Means to data"""
        np.random.seed(self.random_state)
        
        # 1. Initialize centroids randomly
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # 2. Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            
            # 3. Update centroids
            self.centroids = self._update_centroids(X)
            
            # 4. Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster
                new_centroids[k] = self.centroids[k]
        return new_centroids
    
    def predict(self, X):
        """Predict cluster for new data"""
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels

# Test custom implementation
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

kmeans_custom = KMeansFromScratch(n_clusters=4, random_state=42)
labels = kmeans_custom.fit_predict(X)

print(f"Centroids:\n{kmeans_custom.centroids}")
print(f"Unique labels: {np.unique(labels)}")
```

## Using Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('data/customer_data.csv')
features = ['age', 'income', 'spending_score']
X = df[features]

# Critical: Scale features!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',  # Smart initialization
    n_init=10,  # Run 10 times with different seeds
    max_iter=300,
    random_state=42
)

df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster centers (in original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=features)
print("Cluster Centers:")
print(centers_df)

# Cluster sizes
print("\nCluster Sizes:")
print(df['cluster'].value_counts().sort_index())

# Inertia (WCSS)
print(f"\nWithin-Cluster Sum of Squares: {kmeans.inertia_:.2f}")
```

## Determining Optimal K

### Elbow Method

```python
import matplotlib.pyplot as plt

# Try different K values
K_range = range(2, 11)
wcss = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# Mark the "elbow"
plt.axvline(x=4, color='r', linestyle='--', label='Elbow at K=4')
plt.legend()
plt.show()

print("WCSS for each K:")
for k, w in zip(K_range, wcss):
    print(f"K={k}: {w:.2f}")
```

### Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Calculate silhouette scores
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', label='Good threshold (0.5)')
plt.legend()
plt.show()

# Detailed silhouette plot for specific K
def plot_silhouette(X, k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Silhouette plot
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    
    y_lower = 10
    for i in range(k):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size
        
        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    
    ax1.set_title(f'Silhouette Plot for K={k}')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster')
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average: {silhouette_avg:.3f}')
    ax1.legend()
    
    # Cluster visualization
    colors = cm.nipy_spectral(labels.astype(float) / k)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=50, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    
    ax2.set_title(f'Clustered Data (K={k})')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

plot_silhouette(X_scaled[:, :2], k=4)  # Using first 2 features for visualization
```

### Gap Statistic

```python
import numpy as np
from sklearn.cluster import KMeans

def gap_statistic(X, K_max=10, n_refs=10, random_state=42):
    """
    Calculate gap statistic for optimal K
    """
    gaps = []
    errors = []
    
    for k in range(1, K_max + 1):
        # Cluster actual data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        log_Wk = np.log(kmeans.inertia_)
        
        # Generate reference datasets (uniform random)
        ref_log_Wks = []
        for _ in range(n_refs):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans_ref.fit(X_ref)
            ref_log_Wks.append(np.log(kmeans_ref.inertia_))
        
        # Calculate gap
        gap = np.mean(ref_log_Wks) - log_Wk
        error = np.std(ref_log_Wks) * np.sqrt(1 + 1/n_refs)
        
        gaps.append(gap)
        errors.append(error)
    
    return np.array(gaps), np.array(errors)

# Calculate gap statistics
gaps, errors = gap_statistic(X_scaled, K_max=10)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, 11), gaps, yerr=errors, fmt='o-', 
             capsize=5, capthick=2, linewidth=2)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic Method')
plt.grid(True, alpha=0.3)
plt.show()

# Optimal K: first K where gap[k] >= gap[k+1] - error[k+1]
optimal_k = 1
for k in range(len(gaps) - 1):
    if gaps[k] >= gaps[k+1] - errors[k+1]:
        optimal_k = k + 1
        break

print(f"Optimal K by gap statistic: {optimal_k}")
```

## Complete Example: Customer Segmentation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('data/mall_customers.csv')
print(df.head())
print(df.info())

# Select features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K (Elbow + Silhouette)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

K_range = range(2, 11)
wcss = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

ax1.plot(K_range, wcss, 'bo-')
ax1.set_xlabel('K')
ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Choose K=5 (from analysis)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Centers (original scale):")
print(cluster_summary)

print("\nCluster Sizes:")
print(df['Cluster'].value_counts().sort_index())

# Visualize (3D)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'orange', 'purple']
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['Age'],
               cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               c=colors[cluster],
               label=f'Cluster {cluster}',
               s=50,
               alpha=0.6)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.set_title('Customer Segments (3D)')
ax.legend()
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
for cluster in range(optimal_k):
    cluster_mask = df['Cluster'] == cluster
    plt.scatter(X_pca[cluster_mask, 0],
                X_pca[cluster_mask, 1],
                label=f'Cluster {cluster}',
                alpha=0.6,
                s=50)

# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            s=300, c='black', marker='X', edgecolors='white',
            linewidths=2, label='Centroids')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments (PCA Projection)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Business interpretation
segment_names = {
    0: 'Young Low Spenders',
    1: 'High Income Low Spenders',
    2: 'Average Customers',
    3: 'High Income High Spenders (VIP)',
    4: 'Young High Spenders'
}

df['Segment'] = df['Cluster'].map(segment_names)

# Save results
df.to_csv('output/customer_segments.csv', index=False)
print("\nSegmentation complete! Results saved.")
```

## K-Means with Spark MLlib

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()

# Load data
df = spark.read.csv("data/customers.csv", header=True, inferSchema=True)

# Feature engineering
feature_cols = ['age', 'income', 'spending_score']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# Find optimal K
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42)
    model = kmeans.fit(df_scaled)
    predictions = model.transform(df_scaled)
    
    evaluator = ClusteringEvaluator(featuresCol="scaled_features")
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    print(f"K={k}: Silhouette = {silhouette:.3f}")

# Train final model
kmeans = KMeans(featuresCol="scaled_features", k=5, seed=42)
model = kmeans.fit(df_scaled)

# Cluster centers
centers = model.clusterCenters()
print("\nCluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Make predictions
predictions = model.transform(df_scaled)
predictions.select("age", "income", "spending_score", "prediction").show(10)

# Save results
predictions.write.parquet("output/customer_clusters.parquet", mode="overwrite")

spark.stop()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **K-Means minimizes** within-cluster variance
2. **K must be specified** - use elbow method, silhouette analysis, or gap statistic
3. **Feature scaling critical** for K-Means
4. **k-means++ initialization** improves convergence
5. **Sensitive to outliers** - consider preprocessing
6. **Assumes spherical clusters** - may fail with complex shapes
7. **Efficient and scalable** - works well for Big Data with Spark
:::

## Further Reading

- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
