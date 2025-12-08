# Clustering Evaluation Metrics

## Learning Objectives

- Understand internal vs external evaluation metrics
- Calculate and interpret silhouette score
- Use Davies-Bouldin and Calinski-Harabasz indices
- Apply external metrics (ARI, NMI) when labels are available
- Choose appropriate metrics for different scenarios
- Visualize and compare clustering quality

## Introduction

Evaluating clustering quality is challenging because clustering is **unsupervised**—we typically don't have ground truth labels. Evaluation metrics help us:

1. **Compare different algorithms**
2. **Choose optimal number of clusters**
3. **Assess clustering quality**
4. **Validate results**

## Types of Evaluation Metrics

### Internal Metrics

**Use only the data and cluster assignments** (no ground truth)

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index
- Within-Cluster Sum of Squares (WCSS)

### External Metrics

**Compare to ground truth labels** (if available)

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- V-Measure
- Fowlkes-Mallows Index

## Internal Evaluation Metrics

### 1. Silhouette Score

**Measures how similar an object is to its own cluster** compared to other clusters

**Formula for single sample**:
\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

where:
- \(a(i)\) = average distance to points in same cluster
- \(b(i)\) = average distance to points in nearest cluster

**Range**: [-1, 1]
- **1**: Perfect clustering
- **0**: Overlapping clusters
- **-1**: Wrong cluster assignment

**Overall silhouette score**: Mean of all samples

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Generate data
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# Try different K values
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', label='Good threshold (0.5)')
plt.legend()
plt.show()

# Best K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nBest K: {best_k} (Silhouette = {max(silhouette_scores):.3f})")
```

### Detailed Silhouette Analysis

```python
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_silhouette_analysis(X, n_clusters):
    """
    Create detailed silhouette plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cluster the data
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X)
    
    # Silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    # Compute silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Silhouette plot
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate silhouette scores for cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with cluster numbers
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    ax1.set_title(f'Silhouette Plot (K={n_clusters})')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster Label')
    
    # Vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average: {silhouette_avg:.3f}')
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.legend()
    
    # 2D visualization
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    # Plot cluster centers
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    
    ax2.set_title(f'Clustered Data (K={n_clusters})')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    plt.suptitle(f'Silhouette Analysis: Average Score = {silhouette_avg:.3f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Analyze different K values
for k in [3, 4, 5]:
    plot_silhouette_analysis(X, k)
```

### 2. Davies-Bouldin Index

**Measures average similarity between clusters**

\[
DB = \frac{1}{n} \sum_{i=1}^{n} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)
\]

where:
- \(s_i\) = average distance within cluster i
- \(d_{ij}\) = distance between cluster centroids

**Range**: [0, ∞]
- **Lower is better**
- 0 = perfect clustering

```python
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

# Evaluate different K values
db_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    db_scores.append(score)
    print(f"K={k}: Davies-Bouldin Index = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), db_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index (Lower is Better)')
plt.grid(alpha=0.3)
plt.show()

best_k = np.argmin(db_scores) + 2
print(f"\nBest K: {best_k} (DB Index = {min(db_scores):.3f})")
```

### 3. Calinski-Harabasz Index (Variance Ratio)

**Ratio of between-cluster to within-cluster variance**

\[
CH = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{n - k}{k - 1}
\]

where:
- \(B_k\) = between-cluster dispersion
- \(W_k\) = within-cluster dispersion
- \(n\) = number of samples
- \(k\) = number of clusters

**Range**: [0, ∞]
- **Higher is better**

```python
from sklearn.metrics import calinski_harabasz_score

# Evaluate different K values
ch_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    ch_scores.append(score)
    print(f"K={k}: Calinski-Harabasz Index = {score:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), ch_scores, 'mo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index (Higher is Better)')
plt.grid(alpha=0.3)
plt.show()

best_k = np.argmax(ch_scores) + 2
print(f"\nBest K: {best_k} (CH Index = {max(ch_scores):.2f})")
```

### 4. Within-Cluster Sum of Squares (WCSS) / Inertia

**Sum of squared distances to cluster centers**

\[
WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
\]

**Used in Elbow Method**

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for different K
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.grid(alpha=0.3)

# Mark elbow (if at K=4)
plt.axvline(x=4, color='r', linestyle='--', label='Elbow at K=4')
plt.legend()
plt.show()

print("WCSS values:")
for k, w in zip(K_range, wcss):
    print(f"K={k}: WCSS = {w:.2f}")
```

## External Evaluation Metrics

### When to Use External Metrics

- Comparing clustering algorithms on benchmark datasets
- Validating clustering when ground truth is available
- Research and algorithm development

### 1. Adjusted Rand Index (ARI)

**Measures similarity between two clusterings**

**Range**: [-1, 1]
- **1**: Perfect match
- **0**: Random assignment
- **Negative**: Worse than random

```python
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate data with known labels
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Cluster with K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Calculate ARI
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari:.3f}")

if ari > 0.9:
    print("Excellent agreement with ground truth")
elif ari > 0.7:
    print("Good agreement")
elif ari > 0.5:
    print("Moderate agreement")
else:
    print("Poor agreement")
```

### 2. Normalized Mutual Information (NMI)

**Measures mutual information between clusterings**

**Range**: [0, 1]
- **1**: Perfect match
- **0**: No mutual information

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"Normalized Mutual Information: {nmi:.3f}")
```

### 3. V-Measure

**Harmonic mean of homogeneity and completeness**

**Homogeneity**: Each cluster contains only members of a single class

**Completeness**: All members of a class are in the same cluster

```python
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score

homogeneity = homogeneity_score(y_true, y_pred)
completeness = completeness_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)

print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-Measure: {v_measure:.3f}")
```

## Comprehensive Evaluation

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Different algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),
    'Hierarchical': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

# Evaluate
results = []

for name, algorithm in algorithms.items():
    y_pred = algorithm.fit_predict(X)
    
    # Internal metrics
    silhouette = silhouette_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)
    calinski = calinski_harabasz_score(X, y_pred)
    
    # External metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    
    results.append({
        'Algorithm': name,
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'Calinski-Harabasz': calinski,
        'ARI': ari,
        'NMI': nmi,
        'V-Measure': v_measure,
        'N_Clusters': len(np.unique(y_pred))
    })

# Results DataFrame
df_results = pd.DataFrame(results)
print("\nClustering Evaluation Summary:")
print(df_results.to_string(index=False))

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI', 'V-Measure']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    values = df_results[metric]
    colors = ['blue', 'green', 'orange']
    
    bars = ax.bar(df_results['Algorithm'], values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Best algorithm
print("\nBest Algorithm by Metric:")
for metric in ['Silhouette', 'ARI', 'NMI', 'V-Measure']:
    best = df_results.loc[df_results[metric].idxmax()]
    print(f"{metric}: {best['Algorithm']} ({best[metric]:.3f})")

# Davies-Bouldin (lower is better)
best_db = df_results.loc[df_results['Davies-Bouldin'].idxmin()]
print(f"Davies-Bouldin: {best_db['Algorithm']} ({best_db['Davies-Bouldin']:.3f})")
```

## Choosing the Right Metric

### Decision Guide

```python
import pandas as pd

guide = pd.DataFrame({
    'Metric': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI'],
    'Type': ['Internal', 'Internal', 'Internal', 'External', 'External'],
    'Range': ['[-1, 1]', '[0, ∞)', '[0, ∞)', '[-1, 1]', '[0, 1]'],
    'Best Value': ['High (>0.5)', 'Low', 'High', 'High (~1)', 'High (~1)'],
    'When to Use': [
        'General purpose, intuitive',
        'Compact clusters',
        'Well-separated clusters',
        'Ground truth available',
        'Ground truth available'
    ]
})

print("Clustering Metrics Guide:")
print(guide.to_string(index=False))
```

### Practical Recommendations

```python
print("""
Recommendations:

1. **Always use multiple metrics** - no single metric is perfect

2. **For general clustering**:
   - Primary: Silhouette Score
   - Secondary: Davies-Bouldin Index
   - Tertiary: Calinski-Harabasz Index

3. **For comparing algorithms**:
   - Use all internal metrics
   - Combine with domain knowledge
   - Visualize results

4. **If ground truth available**:
   - Use external metrics (ARI, NMI)
   - Still check internal metrics

5. **For Big Data**:
   - Sample data if needed
   - Use efficient metrics (avoid pairwise distances)

6. **Remember**:
   - High metric ≠ meaningful clusters
   - Interpret with domain knowledge
   - Visual inspection is valuable
""")
```

## Stability Analysis

**Check if clustering is stable** across different runs

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Multiple runs
n_runs = 20
ari_scores = []

for i in range(n_runs):
    # First clustering
    kmeans1 = KMeans(n_clusters=4, random_state=i, n_init=10)
    labels1 = kmeans1.fit_predict(X)
    
    # Second clustering (different initialization)
    kmeans2 = KMeans(n_clusters=4, random_state=i+100, n_init=10)
    labels2 = kmeans2.fit_predict(X)
    
    # Compare
    ari = adjusted_rand_score(labels1, labels2)
    ari_scores.append(ari)

print("Stability Analysis:")
print(f"Mean ARI across runs: {np.mean(ari_scores):.3f}")
print(f"Std ARI: {np.std(ari_scores):.3f}")

if np.mean(ari_scores) > 0.9:
    print("\nClustering is very stable")
elif np.mean(ari_scores) > 0.7:
    print("\nClustering is moderately stable")
else:
    print("\nClustering is unstable - consider different K or algorithm")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Internal metrics** don't need ground truth (Silhouette, DB, CH)
2. **External metrics** require ground truth (ARI, NMI)
3. **Silhouette score** is most intuitive (range [-1, 1])
4. **Use multiple metrics** for comprehensive evaluation
5. **Davies-Bouldin**: lower is better
6. **Calinski-Harabasz**: higher is better
7. **Visual inspection** complements metrics
8. **Stability analysis** checks robustness
9. **No perfect metric** - combine with domain knowledge
10. **For Big Data**: sample if needed for efficiency
:::

## Further Reading

- Rousseeuw, P. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- Hubert, L. & Arabie, P. (1985). "Comparing partitions" (ARI)
- Vinh, N. et al. (2010). "Information theoretic measures for clusterings comparison" (NMI)
- Scikit-learn Clustering Metrics: [sklearn.metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
