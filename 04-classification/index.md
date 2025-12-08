# Chapter 4: Classification Methods

## Introduction

Classification is a supervised learning technique that predicts categorical labels. Unlike clustering (unsupervised), classification learns from labeled training data to make predictions on new, unseen data. It's fundamental to applications like spam detection, medical diagnosis, customer churn prediction, and fraud detection.

## Learning Objectives

- Understand classification problem types and applications
- Implement logistic regression for binary classification
- Build decision trees and random forests
- Evaluate classifier performance with multiple metrics
- Handle imbalanced datasets
- Apply classification to Big Data with Spark MLlib
- Choose appropriate algorithms for different scenarios

## Chapter Overview

1. **Classification Fundamentals** - Problem types, training/testing, overfitting
2. **Logistic Regression** - Probabilistic binary classification
3. **Decision Trees** - Rule-based classification
4. **Random Forests** - Ensemble methods
5. **Evaluation Metrics** - Accuracy, precision, recall, F1-score, ROC-AUC
6. **Practical Applications** - End-to-end classification projects

## What is Classification?

### Definition

**Classification**: Predict discrete class labels from input features

\[
f: X \rightarrow Y
\]

where:
- \(X\) = feature space (inputs)
- \(Y\) = label space (outputs)

### Binary Classification

**Two classes**: Positive (1) vs. Negative (0)

**Examples**:
- Email: Spam vs. Not Spam
- Medical: Disease vs. Healthy
- Finance: Fraud vs. Legitimate
- Customer: Churn vs. Retain

### Multiclass Classification

**More than two classes**

**Examples**:
- Iris species: Setosa, Versicolor, Virginica
- News categories: Sports, Politics, Technology, Entertainment
- Image recognition: Cat, Dog, Bird, Fish

### Multilabel Classification

**Multiple labels per instance**

**Examples**:
- Movie genres: [Action, Sci-Fi, Thriller]
- Document tags: [Python, MachineLearning, Tutorial]

## Classification Workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('data/customer_churn.csv')

# 2. Prepare features and target
X = df[['age', 'income', 'tenure', 'monthly_charges']]
y = df['churned']  # Binary: 0 or 1

# 3. Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Class distribution: {y.value_counts().to_dict()}")

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# 5. Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)  # Probabilities

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Train/Test Split

### Why Split?

- **Training set**: Learn patterns
- **Test set**: Evaluate generalization
- **Avoid overfitting**: Model shouldn't "memorize" training data

### Stratified Split

```python
from sklearn.model_selection import train_test_split

# Maintain class proportions in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # Reproducibility
    stratify=y          # Preserve class distribution
)

# Verify stratification
print("Training set class distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest set class distribution:")
print(y_test.value_counts(normalize=True))
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# 5-fold cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")
```

## Real-World Applications

### Email Spam Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample emails
emails = [
    "Win free money now!!!",
    "Meeting tomorrow at 3pm",
    "Click here for prizes!!!",
    "Project update attached"
]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),
    ('classifier', MultinomialNB())
])

# Train
pipeline.fit(emails, labels)

# Predict
new_emails = ["Congratulations! You won!", "See you at the meeting"]
predictions = pipeline.predict(new_emails)
print(predictions)  # [1, 0] - spam, ham
```

### Credit Risk Assessment

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load loan data
df = pd.read_csv('data/loans.csv')

# Features: income, credit_score, debt_ratio, employment_years
# Target: default (1=defaulted, 0=repaid)

X = df[['income', 'credit_score', 'debt_ratio', 'employment_years']]
y = df['default']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

# Predict probability of default
new_applicant = [[50000, 720, 0.35, 5]]  # income, credit_score, debt_ratio, years
default_prob = rf.predict_proba(new_applicant)[0][1]
print(f"Default probability: {default_prob:.1%}")

if default_prob > 0.5:
    print("Decision: REJECT loan application")
else:
    print("Decision: APPROVE loan application")
```

### Medical Diagnosis

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Features: age, blood_pressure, cholesterol, blood_sugar
# Target: heart_disease (1=yes, 0=no)

df = pd.read_csv('data/heart_disease.csv')
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# Train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(dt, 
               feature_names=X.columns,
               class_names=['No Disease', 'Disease'],
               filled=True,
               rounded=True)
plt.show()

# Interpret: Doctors can follow decision rules
```

## Evaluation Metrics

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['Not Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Extract values
TN, FP, FN, TP = cm.ravel()
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Positives: {TP}")
```

### Accuracy, Precision, Recall, F1

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: (TP + TN) / Total
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Precision: TP / (TP + FP) - "Of predicted positives, how many are correct?"
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2%}")

# Recall (Sensitivity): TP / (TP + FN) - "Of actual positives, how many did we catch?"
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2%}")

# F1-Score: Harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.2%}")

# When to use each:
# - Accuracy: Balanced classes, equal cost of errors
# - Precision: Cost of FP is high (spam detection - don't want to mark important email as spam)
# - Recall: Cost of FN is high (disease detection - don't want to miss sick patients)
# - F1: Balance between precision and recall
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"AUC-ROC Score: {roc_auc:.3f}")
# AUC interpretation:
# 1.0 = Perfect classifier
# 0.9-1.0 = Excellent
# 0.8-0.9 = Good
# 0.7-0.8 = Fair
# 0.6-0.7 = Poor
# 0.5 = Random (no discrimination)
```

## Handling Imbalanced Data

### Problem

```python
# Highly imbalanced dataset
print(y.value_counts())
# 0 (majority): 9500
# 1 (minority): 500

# A naive classifier predicting all 0s achieves 95% accuracy!
# But completely fails to identify the minority class
```

### Solution 1: Resampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Oversample minority class (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_resampled))

# Undersample majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### Solution 2: Class Weights

```python
from sklearn.linear_model import LogisticRegression

# Automatically adjust weights inversely proportional to class frequencies
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

### Solution 3: Different Metrics

```python
# Don't use accuracy! Use:
# - Precision, Recall, F1-Score
# - ROC-AUC
# - Precision-Recall curve

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Chapter Structure

Detailed sections:

1. **Logistic Regression** - Binary and multiclass classification
2. **Decision Trees** - Interpretable rule-based models
3. **Random Forests** - Ensemble of decision trees
4. **Support Vector Machines** - Maximum margin classifiers
5. **Naive Bayes** - Probabilistic classifiers
6. **Evaluation Metrics** - Comprehensive assessment
7. **Practical Applications** - Real-world projects

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Classification predicts categorical labels** from features
2. **Always split data** into training and test sets
3. **Feature scaling** important for some algorithms
4. **Multiple evaluation metrics** needed, not just accuracy
5. **Imbalanced data requires special handling**
6. **Cross-validation** provides robust performance estimates
7. **Different algorithms** have different strengths
:::

## Next Steps

Proceed to:
- **Section 4.1**: Logistic Regression
- **Section 4.2**: Decision Trees
- **Section 4.3**: Random Forests
- **Section 4.4**: Evaluation Metrics Deep Dive
- **Section 4.5**: Practical Applications

## Further Reading

- Hastie, T. et al. (2009). "The Elements of Statistical Learning"
- James, G. et al. (2013). "An Introduction to Statistical Learning"
- Scikit-learn Classification Guide: [sklearn.org](https://scikit-learn.org/stable/supervised_learning.html)
