# Practical Clustering Applications

## Learning Objectives

- Apply clustering to real-world business problems
- Build end-to-end clustering pipelines
- Interpret and communicate clustering results
- Implement clustering at scale with Big Data
- Create actionable insights from clusters

## Introduction

This section provides complete, production-ready clustering applications across different domains. Each example includes data preparation, algorithm selection, evaluation, visualization, and business interpretation.

## 1. Customer Segmentation for E-Commerce

### Business Problem

An e-commerce company wants to segment customers for targeted marketing campaigns based on purchasing behavior.

### Complete Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Set style
sns.set_style('whitegrid')

# 1. Load and explore data
df = pd.read_csv('data/ecommerce_customers.csv')

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# 2. Feature Engineering: RFM Analysis
# Recency: Days since last purchase
# Frequency: Number of purchases
# Monetary: Total spending

df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
reference_date = df['last_purchase_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_id').agg({
    'last_purchase_date': lambda x: (reference_date - x.max()).days,
    'order_id': 'count',
    'revenue': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

print("\nRFM Features:")
print(rfm.describe())

# 3. Handle outliers
from scipy import stats

# Remove extreme outliers (Z-score > 3)
z_scores = np.abs(stats.zscore(rfm[['recency', 'frequency', 'monetary']]))
rfm_clean = rfm[(z_scores < 3).all(axis=1)]

print(f"\nRows removed (outliers): {len(rfm) - len(rfm_clean)}")

# 4. Feature scaling
scaler = StandardScaler()
features = ['recency', 'frequency', 'monetary']
X_scaled = scaler.fit_transform(rfm_clean[features])

# 5. Determine optimal K
silhouette_scores = []
db_scores = []
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    inertias.append(kmeans.inertia_)

# Plot evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(K_range, silhouette_scores, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Analysis')
axes[0].grid(alpha=0.3)

axes[1].plot(K_range, db_scores, 'go-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1].grid(alpha=0.3)

axes[2].plot(K_range, inertias, 'ro-', linewidth=2)
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Inertia')
axes[2].set_title('Elbow Method')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_evaluation.png', dpi=300)
plt.show()

# Choose K=4 based on metrics
optimal_k = 4

# 6. Final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm_clean['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nFinal clustering with K={optimal_k}:")
print(f"Silhouette Score: {silhouette_score(X_scaled, rfm_clean['cluster']):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, rfm_clean['cluster']):.3f}")

# 7. Cluster analysis
cluster_summary = rfm_clean.groupby('cluster')[features].agg(['mean', 'median', 'count'])
print("\nCluster Summary:")
print(cluster_summary)

# 8. Cluster naming and interpretation
cluster_names = {
    0: 'Champions',       # High R, F, M
    1: 'Loyal Customers', # High F, M; Medium R
    2: 'At Risk',         # Low R; Medium F, M
    3: 'Lost'             # Low R, F, M
}

# Manual assignment based on cluster centers
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=features)
centers_df['cluster'] = range(optimal_k)

print("\nCluster Centers (Original Scale):")
print(centers_df)

# Assign names
rfm_clean['segment'] = rfm_clean['cluster'].map(cluster_names)

# 9. Visualization
# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'orange']
for cluster in range(optimal_k):
    cluster_data = rfm_clean[rfm_clean['cluster'] == cluster]
    ax.scatter(cluster_data['recency'],
               cluster_data['frequency'],
               cluster_data['monetary'],
               c=colors[cluster],
               label=cluster_names[cluster],
               s=50,
               alpha=0.6)

ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (orders)')
ax.set_zlabel('Monetary (total spending)')
ax.set_title('Customer Segments (RFM Analysis)')
ax.legend()
plt.savefig('customer_segments_3d.png', dpi=300)
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
for cluster in range(optimal_k):
    mask = rfm_clean['cluster'] == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=cluster_names[cluster],
                alpha=0.6,
                s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments (PCA Projection)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('customer_segments_2d.png', dpi=300)
plt.show()

# Heatmap
cluster_means = rfm_clean.groupby('segment')[features].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means.T, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Value'})
plt.title('Customer Segment Characteristics')
plt.xlabel('Segment')
plt.ylabel('RFM Feature')
plt.tight_layout()
plt.savefig('segment_heatmap.png', dpi=300)
plt.show()

# 10. Business insights and recommendations
print("\n" + "="*60)
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("="*60)

for cluster in range(optimal_k):
    segment_data = rfm_clean[rfm_clean['cluster'] == cluster]
    name = cluster_names[cluster]
    size = len(segment_data)
    pct = size / len(rfm_clean) * 100
    
    avg_recency = segment_data['recency'].mean()
    avg_frequency = segment_data['frequency'].mean()
    avg_monetary = segment_data['monetary'].mean()
    
    print(f"\n{name.upper()}")
    print(f"  Size: {size} customers ({pct:.1f}%)")
    print(f"  Avg Recency: {avg_recency:.0f} days")
    print(f"  Avg Frequency: {avg_frequency:.1f} orders")
    print(f"  Avg Monetary: ${avg_monetary:,.2f}")
    
    # Recommendations
    if name == 'Champions':
        print("  ✅ Strategy: VIP program, early access, loyalty rewards")
        print("  ✅ Action: Maintain engagement, upsell premium products")
    elif name == 'Loyal Customers':
        print("  ✅ Strategy: Cross-sell, bundle offers")
        print("  ✅ Action: Encourage higher frequency, exclusive deals")
    elif name == 'At Risk':
        print("  ⚠️ Strategy: Re-engagement campaign, special offers")
        print("  ⚠️ Action: Win-back emails, personalized discounts")
    elif name == 'Lost':
        print("  ❌ Strategy: Aggressive win-back or deprioritize")
        print("  ❌ Action: Survey for feedback, final offer campaign")

# 11. Export results
rfm_clean.to_csv('customer_segments.csv', index=False)
print("\nResults saved to 'customer_segments.csv'")

# Marketing allocation
total_revenue = rfm_clean['monetary'].sum()
segment_revenue = rfm_clean.groupby('segment')['monetary'].sum().sort_values(ascending=False)

print("\nRevenue by Segment:")
for segment, revenue in segment_revenue.items():
    pct = revenue / total_revenue * 100
    print(f"{segment:20s}: ${revenue:12,.2f} ({pct:5.1f}%)")
```

## 2. Anomaly Detection in Network Traffic

### Business Problem

Detect unusual network behavior for cybersecurity monitoring.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns

# Load network traffic data
# Features: bytes_sent, bytes_received, packets_sent, packets_received, duration
df = pd.read_csv('data/network_traffic.csv')

print("Network Traffic Dataset:")
print(df.head())
print(f"Shape: {df.shape}")

# Feature engineering
df['total_bytes'] = df['bytes_sent'] + df['bytes_received']
df['total_packets'] = df['packets_sent'] + df['packets_received']
df['bytes_per_packet'] = df['total_bytes'] / df['total_packets'].replace(0, 1)
df['send_receive_ratio'] = df['bytes_sent'] / df['bytes_received'].replace(0, 1)

features = ['total_bytes', 'total_packets', 'bytes_per_packet', 
            'send_receive_ratio', 'duration']

# Handle infinities and missing values
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df[features] = df[features].fillna(df[features].median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# DBSCAN for anomaly detection
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

df['cluster'] = labels

# Identify anomalies (noise points)
anomalies = df[df['cluster'] == -1]
normal = df[df['cluster'] != -1]

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_anomalies = list(labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of anomalies: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
print(f"Number of normal instances: {len(normal)}")

# Analyze anomalies
print("\nAnomaly Statistics:")
print(anomalies[features].describe())

print("\nNormal Traffic Statistics:")
print(normal[features].describe())

# Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X_pca[labels != -1, 0], X_pca[labels != -1, 1],
            c='blue', label='Normal', alpha=0.5, s=20)
plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1],
            c='red', label='Anomaly', alpha=0.8, s=50, marker='x')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Network Traffic: Normal vs Anomalies')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(122)
plt.scatter(normal['total_bytes'], normal['total_packets'],
            c='blue', label='Normal', alpha=0.5, s=20)
plt.scatter(anomalies['total_bytes'], anomalies['total_packets'],
            c='red', label='Anomaly', alpha=0.8, s=50, marker='x')
plt.xlabel('Total Bytes')
plt.ylabel('Total Packets')
plt.title('Total Bytes vs Packets')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Alert generation
print("\n" + "="*60)
print("SECURITY ALERTS")
print("="*60)

for idx, row in anomalies.head(10).iterrows():
    print(f"\nAlert #{idx}:")
    print(f"  IP: {row.get('ip_address', 'N/A')}")
    print(f"  Total Bytes: {row['total_bytes']:,.0f}")
    print(f"  Total Packets: {row['total_packets']:,.0f}")
    print(f"  Duration: {row['duration']:.2f} sec")
    print(f"  Severity: {'HIGH' if row['total_bytes'] > normal['total_bytes'].quantile(0.99) else 'MEDIUM'}")

# Export anomalies
anomalies.to_csv('network_anomalies.csv', index=False)
print("\nAnomalies exported to 'network_anomalies.csv'")
```

## 3. Image Compression with K-Means

### Application

Reduce image file size by clustering similar colors.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Load image
img = Image.open('data/sample_image.jpg')
img_array = np.array(img)

print(f"Original image shape: {img_array.shape}")
print(f"Original size: {img_array.nbytes / 1024:.2f} KB")

# Reshape to (pixels, RGB)
height, width, channels = img_array.shape
img_2d = img_array.reshape(-1, channels)

print(f"Reshaped to: {img_2d.shape}")

# Cluster colors
for n_colors in [8, 16, 32, 64]:
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(img_2d)
    
    # Replace pixels with cluster centroids
    compressed = kmeans.cluster_centers_[labels].astype(np.uint8)
    compressed_img = compressed.reshape(height, width, channels)
    
    # Calculate compression ratio
    original_colors = len(np.unique(img_2d.view([('', img_2d.dtype)] * img_2d.shape[1])))
    compression_ratio = original_colors / n_colors
    
    print(f"\nK={n_colors}: {compression_ratio:.1f}x compression")
    
    # Display
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(img_array)
    plt.title(f'Original ({original_colors:,} colors)')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(compressed_img)
    plt.title(f'Compressed ({n_colors} colors)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'compressed_{n_colors}.png', dpi=300)
    plt.show()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Customer segmentation** enables targeted marketing
2. **RFM analysis** is powerful for e-commerce clustering
3. **DBSCAN** excels at anomaly detection
4. **Feature engineering** critical for meaningful clusters
5. **Scaling** is essential before clustering
6. **Multiple visualizations** aid interpretation
7. **Business context** drives cluster naming
8. **Actionable insights** more important than technical metrics
9. **Document assumptions** and decisions
10. **Monitor cluster stability** over time
:::

## Further Reading

- Marketing Analytics with Python: Customer Segmentation
- Anomaly Detection: A Survey (Chandola et al.)
- MLOps for Clustering: Monitoring Cluster Drift
