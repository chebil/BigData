# Classification Evaluation Metrics

## Learning Objectives

- Understand and calculate classification metrics
- Interpret confusion matrices
- Use ROC curves and AUC for model comparison
- Apply precision-recall curves for imbalanced data
- Choose appropriate metrics for different scenarios
- Implement cross-validation for robust evaluation

## Introduction

Evaluating classification models requires more than just accuracy. Different metrics reveal different aspects of model performance, and the "best" metric depends on your business objectives and data characteristics.

## Confusion Matrix

### Definition

**2×2 matrix for binary classification**:

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------|-------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data and train model
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Extract values
TN, FP, FN, TP = cm.ravel()

print(f"\nTrue Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign']).plot(ax=ax1, cmap='Blues')
ax1.set_title('Confusion Matrix (Counts)')

# Normalized
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
ConfusionMatrixDisplay(cm_normalized, display_labels=['Malignant', 'Benign']).plot(ax=ax2, cmap='Blues')
ax2.set_title('Confusion Matrix (Normalized)')

plt.tight_layout()
plt.show()
```

### Interpreting Confusion Matrix

```python
# Error analysis
total = TN + FP + FN + TP

print("\nError Analysis:")
print(f"Total predictions: {total}")
print(f"Correct predictions: {TN + TP} ({(TN + TP)/total:.1%})")
print(f"Incorrect predictions: {FP + FN} ({(FP + FN)/total:.1%})")
print(f"\nType I Error (False Positive): {FP} ({FP/total:.1%})")
print(f"Type II Error (False Negative): {FN} ({FN/total:.1%})")
```

## Core Metrics

### 1. Accuracy

**Definition**: Proportion of correct predictions

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy_manual = (TP + TN) / (TP + TN + FP + FN)

print(f"Accuracy: {accuracy:.2%}")
print(f"Manual calculation: {accuracy_manual:.2%}")

print("\nWhen to use: Balanced classes, equal cost of errors")
print("When NOT to use: Imbalanced data (misleading!)")
```

### 2. Precision (Positive Predictive Value)

**Definition**: Of predicted positives, how many are actually positive?

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Use case**: When false positives are costly

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
precision_manual = TP / (TP + FP)

print(f"Precision: {precision:.2%}")
print(f"Manual calculation: {precision_manual:.2%}")

print("\nInterpretation:")
print(f"When model predicts positive, it's correct {precision:.0%} of the time")

print("\nExample use cases:")
print("- Spam detection: Don't want to mark important emails as spam (FP)")
print("- Medical diagnosis: Don't want healthy patients to undergo unnecessary treatment (FP)")
```

### 3. Recall (Sensitivity, True Positive Rate)

**Definition**: Of actual positives, how many did we correctly identify?

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**Use case**: When false negatives are costly

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
recall_manual = TP / (TP + FN)

print(f"Recall: {recall:.2%}")
print(f"Manual calculation: {recall_manual:.2%}")

print("\nInterpretation:")
print(f"Model catches {recall:.0%} of all actual positive cases")

print("\nExample use cases:")
print("- Cancer screening: Don't want to miss sick patients (FN)")
print("- Fraud detection: Don't want to miss fraudulent transactions (FN)")
```

### 4. F1-Score

**Definition**: Harmonic mean of precision and recall

\[
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**Use case**: Balance between precision and recall

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
f1_manual = 2 * (precision * recall) / (precision + recall)

print(f"F1-Score: {f1:.2%}")
print(f"Manual calculation: {f1_manual:.2%}")

print("\nWhen to use: Imbalanced data, need balance between precision and recall")
```

### 5. Specificity (True Negative Rate)

**Definition**: Of actual negatives, how many did we correctly identify?

\[
\text{Specificity} = \frac{TN}{TN + FP}
\]

```python
specificity = TN / (TN + FP)

print(f"Specificity: {specificity:.2%}")
print(f"\nInterpretation:")
print(f"Model correctly identifies {specificity:.0%} of negative cases")
```

### Metrics Summary

```python
import pandas as pd
from sklearn.metrics import *

metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
    'Value': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        specificity
    ],
    'Formula': [
        '(TP + TN) / Total',
        'TP / (TP + FP)',
        'TP / (TP + FN)',
        '2 × (P × R) / (P + R)',
        'TN / (TN + FP)'
    ],
    'Focus': [
        'Overall correctness',
        'Minimize FP',
        'Minimize FN',
        'Balance P and R',
        'Identify negatives'
    ]
})

print("\nMetrics Summary:")
print(metrics_summary.to_string(index=False))
```

## ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

**Plots**: True Positive Rate (Recall) vs False Positive Rate

**FPR**:
\[
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
\]

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

print(f"AUC-ROC Score: {roc_auc:.3f}")

print("\nAUC Interpretation:")
if roc_auc >= 0.9:
    print("Excellent (0.9-1.0)")
elif roc_auc >= 0.8:
    print("Good (0.8-0.9)")
elif roc_auc >= 0.7:
    print("Fair (0.7-0.8)")
elif roc_auc >= 0.6:
    print("Poor (0.6-0.7)")
else:
    print("Fail (0.5-0.6)")
```

### Finding Optimal Threshold

```python
import numpy as np
from sklearn.metrics import f1_score

# Calculate F1-score for different thresholds
f1_scores = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(y_pred_thresh)) > 1:  # Avoid division by zero
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    else:
        f1_scores.append(0)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1-score at optimal threshold: {optimal_f1:.3f}")

# Compare with default threshold (0.5)
y_pred_default = (y_pred_proba >= 0.5).astype(int)
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

print(f"\nDefault threshold (0.5):")
print(f"  Precision: {precision_score(y_test, y_pred_default):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_default):.3f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_default):.3f}")

print(f"\nOptimal threshold ({optimal_threshold:.3f}):")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_optimal):.3f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_optimal):.3f}")
```

## Precision-Recall Curve

**Better for imbalanced data** than ROC curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Calculate precision-recall curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

print(f"Average Precision: {avg_precision:.3f}")
print("\nUse PR curve when:")
print("- Data is highly imbalanced")
print("- Focus is on positive class performance")
print("- FP and FN have different costs")
```

## Multiclass Classification Metrics

### Multiclass Confusion Matrix

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load Iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title('Multiclass Confusion Matrix')
plt.show()

print("Confusion Matrix:")
print(cm)
```

### Macro, Micro, and Weighted Averages

```python
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Calculate different averaging methods
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)

micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='micro'
)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted'
)

print("\nAveraging Methods:")
print(f"Macro - Precision: {macro_precision:.3f}, Recall: {macro_recall:.3f}, F1: {macro_f1:.3f}")
print(f"Micro - Precision: {micro_precision:.3f}, Recall: {micro_recall:.3f}, F1: {micro_f1:.3f}")
print(f"Weighted - Precision: {weighted_precision:.3f}, Recall: {weighted_recall:.3f}, F1: {weighted_f1:.3f}")

print("\nExplanations:")
print("- Macro: Simple average across classes (treats all classes equally)")
print("- Micro: Aggregate contributions (emphasizes larger classes)")
print("- Weighted: Weighted average by class support (accounts for imbalance)")
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Single metric
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("5-Fold Cross-Validation (Accuracy):")
print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
print(f"95% CI: [{scores.mean() - 2*scores.std():.3f}, {scores.mean() + 2*scores.std():.3f}]")

# Multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']

scores_multi = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\nMultiple Metrics:")
for metric in scoring:
    test_scores = scores_multi[f'test_{metric}']
    print(f"{metric:20s}: {test_scores.mean():.3f} (+/- {test_scores.std()*2:.3f})")
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure each fold has same class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\nStratified 5-Fold Cross-Validation:")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

print("\nUse stratified CV when:")
print("- Classes are imbalanced")
print("- Small dataset")
print("- Want to ensure representative splits")
```

## Comprehensive Model Comparison

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Metrics to evaluate
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Store results
results = []

for name, model in models.items():
    row = {'Model': name}
    
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=5, scoring=metric)
        row[metric] = scores.mean()
        row[f'{metric}_std'] = scores.std()
    
    results.append(row)

# Create DataFrame
df_results = pd.DataFrame(results)

print("\nModel Comparison (5-Fold CV):")
print(df_results[['Model'] + metrics].to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    models_list = df_results['Model']
    values = df_results[metric]
    errors = df_results[f'{metric}_std']
    
    bars = ax.bar(range(len(models_list)), values, yerr=errors, 
                  capsize=5, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(models_list, rotation=45, ha='right')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Highlight best model
    best_idx = values.idxmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# Best model for each metric
print("\nBest Model by Metric:")
for metric in metrics:
    best = df_results.loc[df_results[metric].idxmax()]
    print(f"{metric:12s}: {best['Model']:20s} ({best[metric]:.3f})")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Accuracy alone is misleading** for imbalanced data
2. **Precision**: Focus on minimizing false positives
3. **Recall**: Focus on minimizing false negatives
4. **F1-Score**: Balance between precision and recall
5. **ROC-AUC**: Overall discrimination ability
6. **PR Curve**: Better for imbalanced data
7. **Confusion matrix**: Complete picture of errors
8. **Cross-validation**: More robust than single train/test split
9. **Multiple metrics**: Comprehensive evaluation
10. **Business context**: Choose metrics aligned with objectives
:::

## Metric Selection Guide

```python
print("""
METRIC SELECTION GUIDE:

1. BALANCED DATA:
   - Primary: Accuracy, F1-Score
   - Secondary: ROC-AUC

2. IMBALANCED DATA:
   - Primary: Precision, Recall, F1-Score
   - Secondary: PR-AUC, not ROC-AUC

3. COST-SENSITIVE:
   - FP costly: Maximize Precision
   - FN costly: Maximize Recall
   - Both costly: Optimize F1 or custom metric

4. MULTICLASS:
   - Use macro/micro/weighted averages
   - Examine per-class metrics

5. PROBABILITY CALIBRATION:
   - ROC-AUC, Brier score
   - Calibration plots

6. ALWAYS:
   - Use cross-validation
   - Examine confusion matrix
   - Consider multiple metrics
   - Validate with business stakeholders
""")
```

## Further Reading

- Sokolova, M. & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks"
- Hand, D. & Till, R. (2001). "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems"
- Scikit-learn Metrics: [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
