# Course Datasets

Comprehensive guide to all datasets used in the Big Data Analytics course.

## Built-in Datasets (scikit-learn)

These datasets are included with scikit-learn and require no download:

### 1. Iris Dataset
```python
from sklearn.datasets import load_iris
iris = load_iris()
```
- **Samples**: 150
- **Features**: 4 (sepal/petal dimensions)
- **Target**: 3 species
- **Use**: Classification, Clustering
- **Labs**: Lab 3 (Clustering)

### 2. Wine Dataset
```python
from sklearn.datasets import load_wine
wine = load_wine()
```
- **Samples**: 178
- **Features**: 13 (chemical analysis)
- **Target**: 3 wine types
- **Use**: Classification

### 3. Breast Cancer Wisconsin
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
```
- **Samples**: 569
- **Features**: 30
- **Target**: Binary (malignant/benign)
- **Use**: Binary Classification
- **Labs**: Lab 4, Lab 7

### 4. California Housing
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```
- **Samples**: 20,640
- **Features**: 8
- **Target**: Median house value
- **Use**: Regression
- **Labs**: Lab 5 (Regression)

### 5. 20 Newsgroups
```python
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')
```
- **Samples**: ~18,000
- **Type**: Text data
- **Categories**: 20
- **Use**: Text Classification, NLP
- **Labs**: Lab 9 (NLP)

---

## Download Datasets

### 1. Credit Card Fraud Detection
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (PCA transformed)
- **Imbalance**: 0.172% fraud
- **Use**: Imbalanced Classification
- **Labs**: Lab 4 (Classification)

**Download Instructions**:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

### 2. Online Retail Dataset
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Size**: 541,909 transactions
- **Period**: Dec 2010 - Dec 2011
- **Use**: Market Basket Analysis
- **Labs**: Lab 6 (Association Rules)

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)
```

### 3. Mall Customer Segmentation
- **Source**: [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Size**: 200 customers
- **Features**: Age, Income, Spending Score
- **Use**: Customer Segmentation
- **Labs**: Lab 3 (Clustering)

```python
url = 'https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv'
df = pd.read_csv(url)
```

### 4. Wine Quality Dataset
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Variants**: Red wine, White wine
- **Samples**: 1,599 (red), 4,898 (white)
- **Features**: 11 physicochemical properties
- **Target**: Quality score (0-10)
- **Use**: Regression, Classification
- **Labs**: Lab 2 (Statistics)

```python
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(red_wine_url, sep=';')
```

### 5. Airline Passengers Time Series
- **Source**: Built-in with statsmodels
- **Samples**: 144 months (1949-1960)
- **Use**: Time Series Forecasting
- **Labs**: Lab 8 (Time Series)

```python
from statsmodels.datasets import get_rdataset
airline = get_rdataset('AirPassengers')
df = airline.data
```

---

## Synthetic Datasets

Create custom datasets for specific learning objectives:

### Classification
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # Imbalanced
    random_state=42
)
```

### Regression
```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)
```

### Clustering
```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1.0,
    random_state=42
)
```

---

## Dataset Storage

Organize datasets in your project:

```
BigData/
├── data/
│   ├── raw/              # Original downloaded data
│   │   ├── creditcard.csv
│   │   ├── online_retail.xlsx
│   │   └── winequality.csv
│   └── processed/        # Cleaned data
│       ├── creditcard_clean.csv
│       └── transactions.csv
```

---

## Data Loading Template

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(name, data_dir='data/raw'):
    """
    Load dataset by name
    
    Parameters:
    -----------
    name : str
        Dataset name ('creditcard', 'online_retail', etc.)
    data_dir : str
        Directory containing datasets
    
    Returns:
    --------
    df : pd.DataFrame
    """
    data_path = Path(data_dir)
    
    datasets = {
        'creditcard': data_path / 'creditcard.csv',
        'online_retail': data_path / 'online_retail.xlsx',
        'winequality': data_path / 'winequality-red.csv',
        'mall_customers': data_path / 'Mall_Customers.csv'
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    
    file_path = datasets[name]
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df

# Usage
df = load_dataset('creditcard')
```

---

## Dataset Citations

When using these datasets in publications:

1. **Credit Card Fraud**: Machine Learning Group - ULB
2. **UCI Datasets**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
3. **Kaggle Datasets**: Individual dataset pages for citations

---

## Alternative Data Sources

- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **AWS Open Data**: https://registry.opendata.aws/
- **Data.gov**: https://data.gov/
- **GitHub Awesome Datasets**: https://github.com/awesomedata/awesome-public-datasets
- **Papers with Code**: https://paperswithcode.com/datasets

---

## Data Ethics & Privacy

**Important Considerations**:

1. **License**: Check dataset license before use
2. **Privacy**: Ensure no PII (Personally Identifiable Information)
3. **Bias**: Be aware of potential biases in data
4. **Attribution**: Always cite data sources
5. **Usage**: Respect terms of use (academic vs commercial)

---

## Troubleshooting

### Download Issues
```python
# If download fails, try:
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```

### Memory Issues
```python
# For large datasets, use chunking:
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)
```

### Missing Values
```python
# Check for missing values:
print(df.isnull().sum())

# Handle missing values:
df = df.dropna()  # Remove
df = df.fillna(df.mean())  # Impute
```
