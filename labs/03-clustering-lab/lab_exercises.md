# Lab 3: Clustering Exercises

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

## Part 1: Data Exploration (15 points)

### Exercise 1.1: Load and Examine Data (5 points)

```python
# Load dataset
url = 'https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv'
customers = pd.read_csv(url)

print("Dataset Shape:", customers.shape)
print("\nFirst 5 rows:")
print(customers.head())
print("\nDataset Info:")
print(customers.info())
print("\nDescriptive Statistics:")
print(customers.describe())
```

**Tasks:**
1. Check for missing values
2. Display data types
3. Calculate basic statistics

**Questions:**
- Q1: How many customers are in the dataset?
- Q2: What is the age range?
- Q3: Are there any missing values?

### Exercise 1.2: Exploratory Data Analysis (10 points)

**TODO:** Create the following visualizations:

```python
# 1. Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0, 0].hist(customers['Age'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution')

# Annual Income distribution
axes[0, 1].hist(customers['Annual Income (k$)'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Income Distribution')

# Spending Score distribution
axes[1, 0].hist(customers['Spending Score (1-100)'], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Spending Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Spending Score Distribution')

# Gender distribution
gender_counts = customers['Genre'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, alpha=0.7)
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Gender Distribution')

plt.tight_layout()
plt.show()

# 2. Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Income vs Spending Score
axes[0].scatter(customers['Annual Income (k$)'], customers['Spending Score (1-100)'], alpha=0.6)
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score (1-100)')
axes[0].set_title('Income vs Spending Score')
axes[0].grid(alpha=0.3)

# Age vs Spending Score
axes[1].scatter(customers['Age'], customers['Spending Score (1-100)'], alpha=0.6, color='green')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].set_title('Age vs Spending Score')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Correlation matrix
corr_data = customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
corr_matrix = corr_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q4: Do you observe any patterns in Income vs Spending Score?
- Q5: Which variables are most correlated?
- Q6: Based on the scatter plots, how many clusters might you expect?

---

## Part 2: K-Means Clustering (25 points)

### Exercise 2.1: Data Preparation (5 points)

```python
# Select features for clustering
X = customers[['Annual Income (k$)', 'Spending Score (1-100)']].values

print("Feature matrix shape:", X.shape)
print("\nFirst 5 samples:")
print(X[:5])

# TODO: Standardize the features
scaler = StandardScaler()
X_scaled = ??  # Your code here

print("\nScaled features (first 5):")
print(X_scaled[:5])
```

**Questions:**
- Q7: Why is standardization important for K-Means?
- Q8: What are the mean and standard deviation of scaled features?

### Exercise 2.2: Elbow Method (10 points)

```python
# TODO: Implement elbow method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(alpha=0.3)
plt.show()

print("Inertias:", inertias)
```

**Questions:**
- Q9: What is the optimal K based on the elbow method?
- Q10: Why does inertia decrease as K increases?

### Exercise 2.3: Silhouette Analysis (10 points)

```python
# TODO: Calculate silhouette scores
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'ro-', linewidth=2)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(alpha=0.3)
plt.show()
```

**Questions:**
- Q11: What K gives the highest silhouette score?
- Q12: Does this agree with the elbow method?
- Q13: What does a silhouette score close to 1 indicate?

---

## Part 3: Hierarchical Clustering (20 points)

### Exercise 3.1: Dendrogram (10 points)

```python
# TODO: Create dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Calculate linkage
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 7))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.axhline(y=6, color='r', linestyle='--', label='Cut at distance 6')
plt.legend()
plt.tight_layout()
plt.show()
```

**Questions:**
- Q14: How many clusters would you choose based on the dendrogram?
- Q15: What does the height of the branches represent?

### Exercise 3.2: Apply Hierarchical Clustering (10 points)

```python
# TODO: Fit hierarchical clustering
n_clusters = ??  # Choose based on dendrogram

hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)

# Evaluate
sil_score_hc = silhouette_score(X_scaled, labels_hc)
db_score_hc = davies_bouldin_score(X_scaled, labels_hc)

print(f"Hierarchical Clustering Results (K={n_clusters}):")
print(f"  Silhouette Score: {sil_score_hc:.3f}")
print(f"  Davies-Bouldin Score: {db_score_hc:.3f}")

# Visualize
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    mask = labels_hc == cluster
    plt.scatter(X[mask, 0], X[mask, 1], label=f'Cluster {cluster+1}', alpha=0.6, s=50)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Hierarchical Clustering (K={n_clusters})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q16: How do results compare to K-Means?
- Q17: What is Davies-Bouldin score and what does lower value mean?

---

## Part 4: DBSCAN (20 points)

### Exercise 4.1: Parameter Selection (10 points)

```python
# TODO: Find optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Sort and plot distances
distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Sample Index (sorted)')
plt.ylabel('4th Nearest Neighbor Distance')
plt.title('K-Distance Graph for DBSCAN eps Selection')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Suggested eps: Look for elbow in the plot")
```

**Questions:**
- Q18: What eps value would you choose?
- Q19: What does eps represent in DBSCAN?

### Exercise 4.2: Apply DBSCAN (10 points)

```python
# TODO: Apply DBSCAN
eps = ??  # Choose based on k-distance graph
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Count clusters and noise points
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"DBSCAN Results:")
print(f"  Number of clusters: {n_clusters_dbscan}")
print(f"  Number of noise points: {n_noise}")
print(f"  Cluster sizes:", np.bincount(labels_dbscan[labels_dbscan >= 0]))

# Visualize
plt.figure(figsize=(10, 6))
unique_labels = set(labels_dbscan)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black'  # Noise points
        marker = 'x'
        label_name = 'Noise'
    else:
        marker = 'o'
        label_name = f'Cluster {label+1}'
    
    mask = labels_dbscan == label
    plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name, marker=marker, alpha=0.6, s=50)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q20: How many clusters did DBSCAN find?
- Q21: Are the noise points reasonable outliers?
- Q22: How does DBSCAN compare to K-Means for this data?

---

## Part 5: Business Application (20 points)

### Exercise 5.1: Final Clustering Solution (10 points)

**TODO:** Choose the best clustering method and apply it:

```python
# Choose: K-Means, Hierarchical, or DBSCAN
# Justify your choice

optimal_k = 5  # Example
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers['Cluster'] = final_kmeans.fit_predict(X_scaled)

# Cluster centers (in original scale)
centers_scaled = final_kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

print("Cluster Centers:")
print(pd.DataFrame(centers, columns=['Income', 'Spending Score']))

# Visualize final clusters
plt.figure(figsize=(12, 8))
for cluster in range(optimal_k):
    mask = customers['Cluster'] == cluster
    plt.scatter(customers[mask]['Annual Income (k$)'], 
                customers[mask]['Spending Score (1-100)'],
                label=f'Cluster {cluster+1}', alpha=0.6, s=100)

# Plot centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=300, 
            edgecolors='black', linewidths=2, label='Centroids')

plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title(f'Customer Segmentation (K={optimal_k})', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q23: Which algorithm did you choose and why?
- Q24: Describe each cluster in business terms

### Exercise 5.2: Cluster Profiling (10 points)

**TODO:** Profile each cluster:

```python
# Calculate cluster statistics
cluster_profile = customers.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'CustomerID': 'count'
})

cluster_profile.columns = ['_'.join(col).strip() for col in cluster_profile.columns.values]
cluster_profile = cluster_profile.rename(columns={'CustomerID_count': 'Size'})

print("\nCluster Profiles:")
print(cluster_profile)

# Gender distribution per cluster
gender_dist = pd.crosstab(customers['Cluster'], customers['Genre'], normalize='index')
print("\nGender Distribution per Cluster:")
print(gender_dist)

# Visualize profiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
customers.boxplot(column='Age', by='Cluster', ax=axes[0, 0])
axes[0, 0].set_title('Age by Cluster')
axes[0, 0].set_xlabel('Cluster')

# Income distribution
customers.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[0, 1])
axes[0, 1].set_title('Income by Cluster')
axes[0, 1].set_xlabel('Cluster')

# Spending Score distribution
customers.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[1, 0])
axes[1, 0].set_title('Spending Score by Cluster')
axes[1, 0].set_xlabel('Cluster')

# Cluster sizes
cluster_sizes = customers['Cluster'].value_counts().sort_index()
axes[1, 1].bar(cluster_sizes.index, cluster_sizes.values, alpha=0.7)
axes[1, 1].set_title('Cluster Sizes')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.show()
```

**Final Analysis Questions:**
- Q25: Name each cluster (e.g., "Budget Conscious", "High Spenders")
- Q26: What marketing strategy would you recommend for each cluster?
- Q27: Which cluster is most valuable to the business?
- Q28: How would you validate these clusters?
- Q29: What additional data would improve segmentation?
- Q30: How often should clustering be updated?

---

## Bonus Challenge (10 extra points)

### Multi-Feature Clustering

**TODO:** Cluster using all features (Age, Income, Spending Score)

```python
# Your code here
# Use PCA for visualization if needed
```

## Submission Checklist

- [ ] All exercises completed
- [ ] All questions answered
- [ ] Visualizations clear and labeled
- [ ] Business recommendations provided
- [ ] Code well-commented
- [ ] Notebook runs without errors

Good luck! 📊
