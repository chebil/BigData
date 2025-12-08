# Phase 2: Data Preparation

## Overview

Data Preparation is often the most time-consuming phase (60-80% of project time). It involves acquiring, cleaning, transforming, and structuring data for analysis.

## Learning Objectives

- Acquire data from multiple sources
- Clean and validate data quality
- Handle missing values and outliers
- Transform and engineer features
- Create analysis-ready datasets

## 1. Data Acquisition

### Data Sources

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example: Acquiring customer data
data_sources = {
    'CRM': 'customer_master.csv',
    'Transactions': 'transactions_db',
    'Support': 'tickets.json',
    'Web': 'clickstream_logs',
    'External': 'market_data_api'
}

print("Data Sources Inventory:")
for source, location in data_sources.items():
    print(f"  {source}: {location}")
```

### Data Extraction

```python
# Load from CSV
customers = pd.read_csv('customer_master.csv')

# Load from database
import sqlite3
conn = sqlite3.connect('transactions.db')
transactions = pd.read_sql('SELECT * FROM transactions', conn)

# Load from API
import requests
response = requests.get('https://api.example.com/data')
external_data = pd.DataFrame(response.json())

print(f"Customers: {customers.shape}")
print(f"Transactions: {transactions.shape}")
print(f"External: {external_data.shape}")
```

## 2. Data Cleaning

### Missing Values

```python
# Check missing values
print("Missing Values:")
print(customers.isnull().sum())
print(f"\nMissing percentage:")
print((customers.isnull().sum() / len(customers) * 100).round(2))

# Handling strategies
# 1. Remove rows with too many missing values
threshold = 0.5  # 50% threshold
customers_clean = customers.dropna(thresh=len(customers.columns) * threshold)

# 2. Impute numerical columns
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
customers['age'] = num_imputer.fit_transform(customers[['age']])

# 3. Impute categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
customers['region'] = cat_imputer.fit_transform(customers[['region']])

# 4. Forward fill for time series
transactions['amount'] = transactions.groupby('customer_id')['amount'].fillna(method='ffill')
```

### Duplicate Detection

```python
# Check for duplicates
duplicates = customers.duplicated()
print(f"\nDuplicate rows: {duplicates.sum()}")

# Remove duplicates
customers_dedup = customers.drop_duplicates()

# Keep specific duplicates
customers_dedup = customers.drop_duplicates(subset=['customer_id'], keep='last')

print(f"After removing duplicates: {customers_dedup.shape}")
```

### Outlier Detection and Treatment

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
axes[0].boxplot(customers['annual_spend'])
axes[0].set_title('Annual Spend - Box Plot')
axes[0].set_ylabel('Amount ($)')

# Histogram
axes[1].hist(customers['annual_spend'], bins=50, edgecolor='black')
axes[1].set_title('Annual Spend - Distribution')
axes[1].set_xlabel('Amount ($)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Z-score method
from scipy import stats

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

# Apply
customers_no_outliers = remove_outliers_iqr(customers, 'annual_spend')
print(f"After outlier removal: {customers_no_outliers.shape}")
```

## 3. Data Transformation

### Data Type Conversion

```python
# Convert to appropriate types
customers['customer_id'] = customers['customer_id'].astype(str)
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
customers['is_active'] = customers['is_active'].astype(bool)
customers['age'] = customers['age'].astype(int)

print("\nData Types:")
print(customers.dtypes)
```

### Normalization and Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (z-score normalization)
scaler = StandardScaler()
customers['age_scaled'] = scaler.fit_transform(customers[['age']])

# MinMaxScaler (0-1 range)
min_max = MinMaxScaler()
customers['spend_normalized'] = min_max.fit_transform(customers[['annual_spend']])

# RobustScaler (robust to outliers)
robust = RobustScaler()
customers['tenure_robust'] = robust.fit_transform(customers[['tenure_days']])

print("\nScaled Features:")
print(customers[['age', 'age_scaled', 'annual_spend', 'spend_normalized']].head())
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

# Label Encoding (ordinal)
le = LabelEncoder()
customers['region_encoded'] = le.fit_transform(customers['region'])

# One-Hot Encoding
customers_ohe = pd.get_dummies(customers, columns=['subscription_type'], prefix='sub')

# Target Encoding
te = ce.TargetEncoder(cols=['region'])
customers['region_target_enc'] = te.fit_transform(customers['region'], customers['churned'])

print("\nEncoded Features:")
print(customers[['region', 'region_encoded']].head())
```

## 4. Feature Engineering

### Temporal Features

```python
# Extract date features
customers['signup_year'] = customers['signup_date'].dt.year
customers['signup_month'] = customers['signup_date'].dt.month
customers['signup_day_of_week'] = customers['signup_date'].dt.dayofweek
customers['signup_quarter'] = customers['signup_date'].dt.quarter

# Calculate time differences
customers['days_since_signup'] = (datetime.now() - customers['signup_date']).dt.days
customers['tenure_years'] = customers['days_since_signup'] / 365

print("\nTemporal Features:")
print(customers[['signup_date', 'signup_year', 'signup_month', 'tenure_years']].head())
```

### Aggregation Features

```python
# Aggregate transaction data
customer_features = transactions.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'std', 'count'],
    'transaction_date': ['min', 'max']
})

customer_features.columns = ['_'.join(col) for col in customer_features.columns]
customer_features = customer_features.reset_index()

# Calculate recency, frequency, monetary (RFM)
reference_date = transactions['transaction_date'].max()

rfm = transactions.groupby('customer_id').agg({
    'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
    'transaction_id': 'count',  # Frequency
    'amount': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

print("\nRFM Features:")
print(rfm.head())
```

### Derived Features

```python
# Business logic features
customers['high_value'] = (customers['annual_spend'] > customers['annual_spend'].median()).astype(int)
customers['long_tenure'] = (customers['tenure_days'] > 365).astype(int)
customers['spend_per_day'] = customers['annual_spend'] / customers['tenure_days']

# Interaction features
customers['age_tenure_interaction'] = customers['age'] * customers['tenure_years']
customers['spend_frequency_ratio'] = customers['annual_spend'] / customers['purchase_count']

print("\nDerived Features:")
print(customers[['annual_spend', 'high_value', 'spend_per_day']].head())
```

## 5. Data Integration

### Merging Datasets

```python
# Merge customers with transactions
customer_complete = customers.merge(
    customer_features,
    on='customer_id',
    how='left'
)

# Add RFM features
customer_complete = customer_complete.merge(
    rfm,
    on='customer_id',
    how='left'
)

print(f"\nIntegrated Dataset: {customer_complete.shape}")
print(f"Columns: {customer_complete.columns.tolist()}")
```

## 6. Data Validation

### Quality Checks

```python
def validate_data(df):
    """Comprehensive data validation"""
    
    checks = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Duplicate Rows': df.duplicated().sum(),
        'Missing Values': df.isnull().sum().sum(),
        'Numeric Columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'Categorical Columns': df.select_dtypes(include=['object']).columns.tolist(),
        'Date Columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    print("Data Validation Report:")
    print("=" * 50)
    for key, value in checks.items():
        print(f"{key}: {value}")
    
    # Value range checks
    print("\nValue Range Checks:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    return checks

validation_results = validate_data(customer_complete)
```

## 7. Creating Analysis-Ready Dataset

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Define features and target
feature_cols = ['age', 'tenure_years', 'annual_spend', 'recency', 'frequency', 'monetary']
X = customer_complete[feature_cols]
y = customer_complete['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nClass distribution in train:")
print(y_train.value_counts(normalize=True))
```

### Save Prepared Data

```python
# Save to files
customer_complete.to_csv('data/processed/customer_features.csv', index=False)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("\nData saved successfully!")
```

## Data Preparation Checklist

- [ ] All data sources identified and accessed
- [ ] Missing values handled appropriately
- [ ] Duplicates removed
- [ ] Outliers detected and treated
- [ ] Data types corrected
- [ ] Features scaled/normalized
- [ ] Categorical variables encoded
- [ ] New features engineered
- [ ] Datasets merged correctly
- [ ] Data validated and quality-checked
- [ ] Train/test split created
- [ ] Prepared data saved

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Data preparation** consumes 60-80% of project time
2. **Missing values** require thoughtful handling strategies
3. **Outliers** should be investigated before removal
4. **Feature engineering** often determines model success
5. **Data validation** prevents downstream errors
6. **Documentation** of transformations is critical
7. **Reproducibility** requires saving all preprocessing steps
:::

## Next Phase

With prepared data, proceed to **Model Planning** to:
- Select appropriate algorithms
- Define evaluation metrics
- Plan model development strategy
