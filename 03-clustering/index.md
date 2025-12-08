# Chapter 3: Advanced Analytics - Clustering

## Introduction

Clustering is an unsupervised machine learning technique that groups similar data points together without predefined labels. It's one of the most widely used methods in Big Data analytics for customer segmentation, anomaly detection, pattern recognition, and data exploration.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand different clustering algorithms and their applications
- Implement K-Means clustering for customer segmentation
- Apply hierarchical clustering for taxonomy creation
- Use DBSCAN for density-based clustering
- Evaluate clustering quality with multiple metrics
- Scale clustering algorithms to Big Data using Spark MLlib
- Choose appropriate clustering method for different scenarios

## Chapter Overview

1. **Clustering Fundamentals** - Types, applications, and distance metrics
2. **K-Means Clustering** - Partitioning-based clustering
3. **Hierarchical Clustering** - Agglomerative and divisive methods
4. **DBSCAN** - Density-based clustering
5. **Clustering Evaluation** - Silhouette score, within-cluster sum of squares
6. **Practical Applications** - Customer segmentation, image compression, anomaly detection

## What is Clustering?

### Definition

**Clustering**: Grouping data points such that:
- Points in the **same cluster** are similar (high intra-cluster similarity)
- Points in **different clusters** are dissimilar (low inter-cluster similarity)

### Key Characteristics

- **Unsupervised**: No labeled training data
- **Exploratory**: Discover hidden patterns
- **Subjective**: "Optimal" clustering depends on objectives

### Applications

#### Customer Segmentation
```python
# Group customers by purchasing behavior
# Features: purchase_frequency, average_value, recency
# Result: High-value, Medium-value, Low-value, Churned segments
```

#### Document Clustering
```python
# Group similar news articles
# Features: TF-IDF vectors
# Result: Sports, Politics, Technology, Entertainment clusters
```

#### Image Segmentation
```python
# Group pixels by color similarity
# Features: RGB values
# Result: Distinct regions in image
```

#### Anomaly Detection
```python
# Identify unusual patterns
# Small clusters or points far from clusters = anomalies
# Application: Fraud detection, network intrusion
```

## Distance Metrics

### Euclidean Distance

**Formula**:
\[
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

```python
import numpy as np
from scipy.spatial.distance import euclidean

point1 = np.array([1, 2, 3])
point2 = np.array([4, 6, 8])

# Method 1: Manual
euc_dist = np.sqrt(np.sum((point1 - point2)**2))

# Method 2: NumPy
euc_dist = np.linalg.norm(point1 - point2)

# Method 3: SciPy
euc_dist = euclidean(point1, point2)

print(f"Euclidean distance: {euc_dist:.2f}")
```

**Use case**: Continuous numerical features with similar scales

### Manhattan Distance

**Formula**:
\[
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
\]

```python
from scipy.spatial.distance import cityblock

manh_dist = cityblock(point1, point2)
print(f"Manhattan distance: {manh_dist:.2f}")
```

**Use case**: Grid-like data, when diagonal movement isn't possible

### Cosine Similarity

**Formula**:
\[
\text{similarity}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
\]

```python
from scipy.spatial.distance import cosine

# Cosine distance = 1 - cosine similarity
cos_dist = cosine(point1, point2)
print(f"Cosine distance: {cos_dist:.4f}")

# Or calculate similarity directly
cos_sim = 1 - cos_dist
print(f"Cosine similarity: {cos_sim:.4f}")
```

**Use case**: Text analysis, high-dimensional sparse data

## Clustering Types

### Partitioning Methods

**Examples**: K-Means, K-Medoids

**Characteristics**:
- Divide data into K non-overlapping clusters
- Require number of clusters K as input
- Fast and scalable

### Hierarchical Methods

**Examples**: Agglomerative, Divisive

**Characteristics**:
- Create tree of clusters (dendrogram)
- Don't require K as input
- More computationally expensive

### Density-Based Methods

**Examples**: DBSCAN, OPTICS

**Characteristics**:
- Find clusters of arbitrary shape
- Identify outliers
- Require density parameters

### Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **K-Means** | Fast, scalable, simple | Needs K, assumes spherical clusters | Large datasets, clear separation |
| **Hierarchical** | No K needed, dendrogram | Slow O(n²), memory intensive | Small datasets, taxonomy |
| **DBSCAN** | Arbitrary shapes, finds outliers | Struggles with varying densities | Spatial data, noise |

## Real-World Example: E-Commerce Customer Segmentation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer data
df = pd.read_csv('data/customers.csv')

# Features for clustering
features = ['recency', 'frequency', 'monetary_value']
X = df[features]

# Standardize features (critical for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)

# Visualize (2D projection)
plt.figure(figsize=(10, 6))
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['frequency'], 
                cluster_data['monetary_value'],
                label=f'Cluster {cluster}',
                alpha=0.6)

plt.xlabel('Purchase Frequency')
plt.ylabel('Monetary Value')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Business interpretation
segment_names = {
    0: 'Champions',  # High RFM
    1: 'Loyal Customers',  # High F, M
    2: 'At Risk',  # Low R
    3: 'Lost'  # Low RFM
}

df['segment_name'] = df['cluster'].map(segment_names)
```

## Chapter Structure

The following sections provide detailed coverage:

1. **K-Means Clustering** - Algorithm, implementation, optimization
2. **Hierarchical Clustering** - Dendrograms, linkage methods
3. **DBSCAN** - Density-based clustering for complex shapes
4. **Evaluation Metrics** - Assessing cluster quality
5. **Practical Applications** - End-to-end clustering projects
6. **Spark MLlib Clustering** - Scaling to Big Data

## Prerequisites

- Python programming (Chapter 1)
- Descriptive statistics (Chapter 2)
- NumPy, Pandas, Matplotlib
- Scikit-learn basics

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Clustering is unsupervised** - no labeled data needed
2. **Different algorithms** suit different data characteristics
3. **Feature scaling critical** for distance-based methods
4. **Evaluation is subjective** - depends on business objectives
5. **Interpretation matters** - clusters must be actionable
6. **Spark MLlib scales** clustering to massive datasets
:::

## Next Steps

Proceed to:
- **Section 3.1**: K-Means Clustering
- **Section 3.2**: Hierarchical Clustering  
- **Section 3.3**: DBSCAN
- **Section 3.4**: Evaluation Metrics
- **Section 3.5**: Practical Applications

## Further Reading

- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 14
- Aggarwal, C. & Reddy, C. (2013). "Data Clustering: Algorithms and Applications"
- Scikit-learn Clustering Guide: [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html)
