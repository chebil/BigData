# Lab 3 Datasets - Customer Segmentation

## Dataset 1: Mall Customers Dataset

### Source
- **Origin**: Kaggle - Customer Segmentation Tutorial
- **URL**: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Size**: 200 customers
- **Format**: CSV

### Features

| Column | Type | Description | Range/Values |
|--------|------|-------------|-------------|
| CustomerID | int | Unique identifier | 1-200 |
| Gender | categorical | Customer gender | Male, Female |
| Age | int | Customer age | 18-70 years |
| Annual Income (k$) | int | Yearly income in thousands | 15-137 k$ |
| Spending Score (1-100) | int | Shopping behavior score | 1-99 |

### Data Description

**Spending Score**: Assigned by mall based on:
- Customer behavior
- Purchase patterns
- Spending amount
- Visit frequency

**Business Goal**: Segment customers to:
1. Target marketing campaigns
2. Personalize promotions
3. Optimize product placement
4. Improve customer retention

### Sample Data

```csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
0001,Male,19,15,39
0002,Male,21,15,81
0003,Female,20,16,6
0004,Female,23,16,77
0005,Female,31,17,40
```

### Loading Code

```python
import pandas as pd
import numpy as np

# Method 1: From URL
url = 'https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv'
df = pd.read_csv(url)

# Method 2: From local file
df = pd.read_csv('Mall_Customers.csv')

# Quick exploration
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
```

---

## Dataset 2: Iris Dataset (for practice)

### Source
- **Origin**: scikit-learn built-in
- **Classic dataset**: Fisher's Iris data (1936)
- **Size**: 150 samples

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| sepal length | Length of sepal | cm |
| sepal width | Width of sepal | cm |
| petal length | Length of petal | cm |
| petal width | Width of petal | cm |
| species | Flower type | setosa, versicolor, virginica |

### Loading Code

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
df_iris['species_name'] = df_iris['species'].map({
    0: 'setosa', 
    1: 'versicolor', 
    2: 'virginica'
})

print("Iris dataset loaded successfully!")
print("Shape:", df_iris.shape)
```

---

## Dataset 3: Wholesale Customers (Advanced)

### Source
- **Origin**: UCI Machine Learning Repository
- **Size**: 440 customers
- **Features**: 8 (6 product categories + 2 nominal)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Channel | Nominal | Hotel/Restaurant/Cafe (1) or Retail (2) |
| Region | Nominal | Lisbon (1), Oporto (2), Other (3) |
| Fresh | Continuous | Annual spending on fresh products |
| Milk | Continuous | Annual spending on milk products |
| Grocery | Continuous | Annual spending on grocery products |
| Frozen | Continuous | Annual spending on frozen products |
| Detergents_Paper | Continuous | Annual spending on detergents/paper |
| Delicassen | Continuous | Annual spending on delicatessen |

### Loading Code

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df_wholesale = pd.read_csv(url)

print("Wholesale customers dataset loaded!")
print("Shape:", df_wholesale.shape)
```

---

## Data Preparation Tips

### 1. Check Data Quality

```python
def check_data_quality(df):
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Missing Values:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values")
    
    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"\n⚠️  {dupes} duplicate rows found")
    else:
        print("\n✅ No duplicates")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Numeric summary
    print("\nNumeric Features Summary:")
    print(df.describe())
    
    return df

# Usage
check_data_quality(df)
```

### 2. Handle Outliers

```python
def detect_outliers(df, columns, method='iqr'):
    """
    Detect outliers using IQR or Z-score method
    """
    outliers_dict = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        else:  # z-score
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
        
        outliers_dict[col] = len(outliers)
        print(f"{col}: {len(outliers)} outliers detected")
    
    return outliers_dict

# Usage
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_counts = detect_outliers(df, numeric_cols)
```

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

# MinMaxScaler (0-1 range)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

print("Scaling complete!")
print("\nOriginal data range:")
print(df[numeric_cols].describe())
print("\nScaled data range:")
print(df_scaled.describe())
```

---

## Expected Clusters

### Mall Customers Dataset

Based on domain knowledge, expect ~5 customer segments:

1. **High Income, High Spending** (Target Group)
   - Premium customers
   - Luxury product buyers
   - VIP treatment

2. **High Income, Low Spending** (Potential Target)
   - Careful spenders
   - Need targeted promotions
   - Quality over quantity

3. **Low Income, High Spending** (Careful Group)
   - Impulse buyers
   - Risk of financial stress
   - Budget-friendly options

4. **Low Income, Low Spending** (Sensible Group)
   - Budget-conscious
   - Value seekers
   - Discount promotions

5. **Average** (Standard Group)
   - Middle income/spending
   - Regular customers
   - Standard offerings

---

## Evaluation Metrics Guide

### Within-Cluster Sum of Squares (WCSS)
- **Lower is better**
- Measures compactness
- Use elbow method to find optimal k

### Silhouette Score
- **Range**: -1 to 1
- **> 0.7**: Strong structure
- **0.5-0.7**: Reasonable structure  
- **0.25-0.5**: Weak structure
- **< 0.25**: No substantial structure

### Davies-Bouldin Index
- **Lower is better**
- Measures cluster separation
- **< 1**: Good clustering

### Calinski-Harabasz Index
- **Higher is better**
- Ratio of between/within cluster dispersion

---

## Visualization Tips

### 1. Pair Plot (Before Clustering)

```python
import seaborn as sns

sns.pairplot(df[numeric_cols])
plt.suptitle('Feature Relationships', y=1.02)
plt.show()
```

### 2. Cluster Visualization (2D)

```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 8))
for cluster in range(k):
    cluster_points = X_pca[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                label=f'Cluster {cluster}', s=50, alpha=0.6)

plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments Visualization')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 3. 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(k):
    cluster_data = df[labels == cluster]
    ax.scatter(cluster_data['Age'], 
               cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.set_title('3D Customer Segmentation')
ax.legend()
plt.show()
```

---

## Business Insights Template

For each cluster, provide:

1. **Size**: Number of customers
2. **Characteristics**: Average age, income, spending
3. **Behavior**: Shopping patterns
4. **Marketing Strategy**: Recommended approach
5. **Product Recommendations**: What to offer
6. **Retention Risk**: Low/Medium/High
7. **Lifetime Value**: Estimated value

**Example Report**:

```
Cluster 1: Premium Shoppers (n=23)
- Age: 32-40 years
- Income: $70k-$130k
- Spending: 75-99/100
- Strategy: VIP program, exclusive offers
- Products: Luxury items, premium brands
- Risk: Low
- LTV: High
```

---

## References

- Mall Customers Dataset: Kaggle
- Iris Dataset: Fisher, R.A. (1936)
- Wholesale: Margarida G.M.S. Cardoso et al., UCI ML Repository
