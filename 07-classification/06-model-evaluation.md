# Model Evaluation

## Introduction

Model evaluation answers critical questions:
- How well does the model perform?
- Which model is best?
- Will it generalize to new data?
- What types of errors does it make?
- Is it ready for deployment?

Proper evaluation requires appropriate metrics, validation strategies, and understanding of the problem context.

## Train-Test Split

### Basic Approach

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducibility
    stratify=y            # Preserve class distribution
)
```

**Typical Split Ratios**:
- **80/20**: Most common
- **70/30**: Smaller datasets
- **90/10**: Large datasets
- **60/20/20**: Train/validation/test

### Stratification

**Why?** Preserve class distribution in splits

**Example without stratification**:
- Original: 90% class A, 10% class B
- Train might be: 95% A, 5% B
- Test might be: 80% A, 20% B
- Biased evaluation!

```python
# Always use stratify for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Check distribution
print("Train:", y_train.value_counts(normalize=True))
print("Test:", y_test.value_counts(normalize=True))
```

## Confusion Matrix

### Binary Classification

```
                    Predicted
                Positive  Negative
Actual  Positive    TP        FN
        Negative    FP        TN
```

**Definitions**:
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

### Example

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

**Interpretation**:
```
[[850  50]    # TN=850, FP=50
 [ 30 170]]   # FN=30, TP=170
```
- 850 correctly classified as negative
- 170 correctly classified as positive
- 50 false alarms (FP)
- 30 missed detections (FN)

### Multi-Class Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.show()
```

## Classification Metrics

### 1. Accuracy

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Interpretation**: Proportion of correct predictions

**When to use**: 
- ✅ Balanced classes
- ✅ Equal costs for all errors

**When NOT to use**:
- ❌ Imbalanced classes
- ❌ Different error costs

**Example**: 
- 95% accuracy sounds great
- But if 95% of data is class A...
- Predicting "always A" gives 95% accuracy!

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

### 2. Precision

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Interpretation**: Of positive predictions, how many are correct?

**Also called**: Positive Predictive Value (PPV)

**When to optimize**:
- False positives are costly
- Example: Spam detection (don't block legitimate emails)
- Example: Fraud detection (don't block legitimate transactions)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")
```

### 3. Recall (Sensitivity)

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**Interpretation**: Of actual positives, how many did we find?

**Also called**: True Positive Rate, Sensitivity, Hit Rate

**When to optimize**:
- False negatives are costly
- Example: Disease screening (don't miss sick patients)
- Example: Fraud detection (don't miss fraud)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")
```

### 4. F1-Score

\[
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
\]

**Interpretation**: Harmonic mean of precision and recall

**When to use**:
- Balance precision and recall
- Imbalanced classes
- Need single metric

**Properties**:
- Range: [0, 1]
- High F1: Both precision and recall high
- Low F1: At least one is low

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")
```

### 5. F-Beta Score

Generalized F1 with adjustable weight:

\[
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
\]

**Beta values**:
- \(\beta = 1\): F1-score (equal weight)
- \(\beta = 2\): F2-score (favor recall)
- \(\beta = 0.5\): F0.5-score (favor precision)

```python
from sklearn.metrics import fbeta_score

# F2: Recall is 2x more important
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-Score: {f2:.3f}")
```

### 6. Specificity

\[
\text{Specificity} = \frac{TN}{TN + FP}
\]

**Interpretation**: Of actual negatives, how many did we correctly identify?

**Also called**: True Negative Rate, Selectivity

**Important for**: 
- Medical screening (correctly identifying healthy patients)
- Quality control (correctly identifying good products)

```python
# Calculate from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.3f}")
```

### 7. Matthews Correlation Coefficient (MCC)

\[
\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

**Properties**:
- Range: [-1, 1]
- +1: Perfect prediction
- 0: Random prediction
- -1: Complete disagreement

**Advantages**:
- Balanced measure
- Works for imbalanced classes
- Considers all four confusion matrix values

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
print(f"MCC: {mcc:.3f}")
```

### Classification Report

All metrics at once:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
```

Output:
```
              precision    recall  f1-score   support

     Class 0       0.97      0.94      0.96       900
     Class 1       0.77      0.85      0.81       200

    accuracy                           0.93      1100
   macro avg       0.87      0.90      0.88      1100
weighted avg       0.94      0.93      0.93      1100
```

**Averages**:
- **Macro**: Unweighted mean (all classes equal)
- **Weighted**: Weighted by support (accounts for imbalance)

## ROC Curve and AUC

### ROC (Receiver Operating Characteristic)

**Plots**:
- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall

**Each point**: Different classification threshold

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

### AUC (Area Under Curve)

**Interpretation**: Probability that model ranks random positive higher than random negative

**Values**:
- 1.0: Perfect classifier
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.6-0.7: Poor
- 0.5: Random (coin flip)
- <0.5: Worse than random (inverted predictions)

```python
from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc_score:.3f}")
```

### Multi-Class ROC

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Binarize labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Get probabilities
y_prob = model.predict_proba(X_test)

# Compute ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend()
plt.show()
```

## Precision-Recall Curve

**Better than ROC for imbalanced data**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Calculate
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'AP = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**When to use PR curve over ROC**:
- Highly imbalanced data
- Positive class is rare
- Focus on positive class performance

## Cross-Validation

### K-Fold Cross-Validation

**Procedure**:
1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Test on remaining fold
3. Average K performance scores

```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='f1')

print(f"F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Stratified K-Fold

**Preserves class distribution** in each fold:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='f1')

print(f"Stratified F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Repeated K-Fold

**More robust estimate**:

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rskfold, scoring='f1')

print(f"Repeated F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Leave-One-Out (LOO)

**Extreme case**: K = n

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print(f"LOO Accuracy: {scores.mean():.3f}")
```

**When to use**: Very small datasets (<100 samples)
**Drawback**: Computationally expensive

### Time Series Split

**Respects temporal order**:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

## Learning Curves

**Diagnose bias/variance**:

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='f1',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, label='Validation score', marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Interpretation**:
- **High bias** (underfitting): Both curves low, plateau early, close together
- **High variance** (overfitting): Large gap between curves
- **Good fit**: Curves converge at high performance

## Validation Curves

**Analyze hyperparameter effect**:

```python
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name='C',
    param_range=param_range,
    cv=5,
    scoring='f1'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score', marker='o')
plt.plot(param_range, val_mean, label='Validation score', marker='o')
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('F1 Score')
plt.title('Validation Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Model Comparison

```python
from sklearn.model_selection import cross_val_score
import pandas as pd

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'SVM': SVC()
}

results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    results.append({
        'Model': name,
        'Mean F1': scores.mean(),
        'Std F1': scores.std()
    })

df_results = pd.DataFrame(results).sort_values('Mean F1', ascending=False)
print(df_results)
```

## Statistical Tests

### Paired T-Test

**Compare two models**:

```python
from scipy import stats

# Cross-validation scores for two models
scores_model1 = cross_val_score(model1, X, y, cv=10)
scores_model2 = cross_val_score(model2, X, y, cv=10)

# Paired t-test
t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference")
```

## Summary

### Metric Selection Guide

| Problem | Best Metric |
|---------|-------------|
| **Balanced classes, equal costs** | Accuracy |
| **Imbalanced classes** | F1, AUC, MCC |
| **False positives costly** | Precision |
| **False negatives costly** | Recall |
| **Need probability ranking** | AUC |
| **Multi-class** | Macro F1, Weighted F1 |
| **Rare positive class** | Precision-Recall AUC |

### Validation Strategy Guide

| Scenario | Strategy |
|----------|----------|
| **Small data (<1000)** | Stratified K-Fold (K=5-10) |
| **Medium data (1K-100K)** | Stratified K-Fold (K=5) |
| **Large data (>100K)** | Train-Val-Test Split |
| **Time series** | Time Series Split |
| **Very small (<100)** | Leave-One-Out |
| **Hyperparameter tuning** | Cross-validation |
| **Final evaluation** | Hold-out test set |

### Best Practices

1. **Always use test set**: Never touch until final evaluation
2. **Stratify splits**: Especially for imbalanced data
3. **Use appropriate metrics**: Not always accuracy
4. **Cross-validate**: For reliable estimates
5. **Compare multiple models**: Statistical tests
6. **Check learning curves**: Diagnose bias/variance
7. **Visualize confusion matrix**: Understand error patterns
8. **Report confidence intervals**: Not just means
9. **Consider business context**: Align metrics with costs
10. **Document everything**: Reproducibility

## Next Steps

You've completed Chapter 7! You now understand:
- Logistic regression for probabilistic classification
- Naïve Bayes for fast text classification
- Decision trees for interpretable decisions
- Random Forests and boosting for high accuracy
- SVMs for high-dimensional data
- Comprehensive evaluation strategies

**Continue to**: [Chapter 8: Time Series Analysis](../08-time-series/index.md)