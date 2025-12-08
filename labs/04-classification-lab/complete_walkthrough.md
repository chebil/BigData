# Lab 4: Classification - Complete Walkthrough

## Fraud Detection System

### Learning Objectives

1. Build binary classification models
2. Handle imbalanced datasets
3. Evaluate model performance
4. Deploy production fraud detection
5. Create comprehensive reports

---

## Part 1: Data Understanding

### Dataset: Credit Card Fraud

**Source**: Kaggle - Credit Card Fraud Detection
**Size**: 284,807 transactions
**Fraud Rate**: 0.172% (492 frauds out of 284,807)
**Challenge**: Highly imbalanced classification

### Features

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

# Load data
# For this lab, we'll use a simulated smaller dataset
np.random.seed(42)

# Simulate fraud detection data
n_samples = 10000
n_features = 30

# Normal transactions (99.8%)
n_normal = int(n_samples * 0.998)
X_normal = np.random.randn(n_normal, n_features)
y_normal = np.zeros(n_normal)

# Fraudulent transactions (0.2%)
n_fraud = n_samples - n_normal
X_fraud = np.random.randn(n_fraud, n_features) + 3  # Shifted distribution
y_fraud = np.ones(n_fraud)

# Combine
X = np.vstack([X_normal, X_fraud])
y = np.hstack([y_normal, y_fraud])

# Create DataFrame
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
df = pd.DataFrame(X, columns=feature_names)
df['Class'] = y

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.3f}%")
```

### Initial Exploration

```python
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Basic info
print("\n1. Dataset Info:")
print(df.info())

# Statistics
print("\n2. Descriptive Statistics:")
print(df.describe())

# Missing values
print("\n3. Missing Values:")
print(df.isnull().sum().sum())

# Class distribution
print("\n4. Class Balance:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"\nImbalance ratio: 1:{int(class_counts[0]/class_counts[1])}")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
df['Class'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Normal, 1=Fraud)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)

# Pie chart
axes[1].pie(class_counts, labels=['Normal', 'Fraud'], autopct='%1.3f%%',
            colors=['skyblue', 'salmon'], startangle=90)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Part 2: Feature Analysis

### Feature Distribution by Class

```python
# Select a few features for visualization
features_to_plot = ['V1', 'V2', 'V3', 'Amount']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(features_to_plot):
    # Normal transactions
    axes[idx].hist(df[df['Class']==0][feature], bins=50, alpha=0.7, 
                   label='Normal', color='skyblue', density=True)
    # Fraud transactions
    axes[idx].hist(df[df['Class']==1][feature], bins=50, alpha=0.7,
                   label='Fraud', color='salmon', density=True)
    axes[idx].set_title(f'{feature} Distribution by Class', fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Correlation Analysis

```python
# Correlation with target
corr_with_target = df.corr()['Class'].sort_values(ascending=False)

print("\nFeatures most correlated with Fraud:")
print(corr_with_target[1:11])  # Top 10

print("\nFeatures least correlated with Fraud:")
print(corr_with_target[-10:])  # Bottom 10

# Heatmap of top correlated features
top_features = corr_with_target[1:11].index.tolist() + ['Class']
corr_matrix = df[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix - Top Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Part 3: Data Preparation

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train set:")
print(f"  Shape: {X_train.shape}")
print(f"  Fraud rate: {y_train.mean()*100:.3f}%")

print("\nTest set:")
print(f"  Shape: {X_test.shape}")
print(f"  Fraud rate: {y_test.mean()*100:.3f}%")
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nFeatures scaled successfully!")
print("\nScaled features statistics:")
print(X_train_scaled.describe())
```

---

## Part 4: Baseline Models

### Model 1: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("LOGISTIC REGRESSION RESULTS")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Normal', 'Fraud']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix - Logistic Regression', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

[CONTINUES WITH COMPLETE WALKTHROUGH...]
