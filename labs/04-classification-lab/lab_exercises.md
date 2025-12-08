# Lab 4: Classification - Fraud Detection

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
```

## Part 1: Data Exploration (20 points)

### Exercise 1.1: Load Credit Card Dataset

```python
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# For demo, we'll create synthetic data
np.random.seed(42)

# Create imbalanced dataset
n_samples = 10000
n_features = 28

# Generate features
X_genuine = np.random.randn(int(n_samples * 0.998), n_features)
X_fraud = np.random.randn(int(n_samples * 0.002), n_features) + 3  # Shift fraud

X = np.vstack([X_genuine, X_fraud])
y = np.hstack([np.zeros(len(X_genuine)), np.ones(len(X_fraud))])

# Create DataFrame
feature_names = [f'V{i}' for i in range(1, n_features+1)]
df = pd.DataFrame(X, columns=feature_names)
df['Amount'] = np.random.exponential(100, len(df))
df['Class'] = y

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {(df['Class']==1).mean()*100:.2f}%")
```

**Questions:**
- Q1: How imbalanced is the dataset?
- Q2: Why is imbalance a problem?
- Q3: What metrics should we use?

Good luck! 🔒
