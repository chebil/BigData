# Cross-Validation

## Learning Objectives

- Understand the need for cross-validation
- Implement k-fold cross-validation
- Apply stratified and group cross-validation
- Use time series cross-validation
- Compare different cross-validation strategies
- Avoid common pitfalls in model validation

## Introduction

Cross-validation is essential for **reliable model evaluation**. A single train/test split can be misleading due to:
- **Variance**: Different splits give different results
- **Data waste**: Portion of data unused for training
- **Overfitting to test set**: If used repeatedly for tuning

## Train/Test Split Limitations

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Multiple train/test splits with different random states
accuracies = []

for random_state in range(50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print("Train/Test Split Variability:")
print(f"Mean accuracy: {np.mean(accuracies):.3f}")
print(f"Std accuracy: {np.std(accuracies):.3f}")
print(f"Min accuracy: {np.min(accuracies):.3f}")
print(f"Max accuracy: {np.max(accuracies):.3f}")
print(f"Range: {np.max(accuracies) - np.min(accuracies):.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label='Mean')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Variability in Test Accuracy with Different Train/Test Splits')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\nConclusion: Single split can be misleading!")
```

## K-Fold Cross-Validation

### Concept

```python
print("""
K-FOLD CROSS-VALIDATION:

1. Split data into K equal folds
2. For each fold (i = 1 to K):
   - Train on K-1 folds
   - Validate on remaining fold
3. Average the K validation scores

Example with K=5:

Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]

Final Score = Average of 5 validation scores
""")
```

### Implementation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\n5-Fold Cross-Validation Results:")
for i, score in enumerate(scores, 1):
    print(f"Fold {i}: {score:.3f}")

print(f"\nMean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
print(f"95% CI: [{scores.mean() - 2*scores.std():.3f}, {scores.mean() + 2*scores.std():.3f}]")

# Custom KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nCustom KFold (with shuffle):")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} validation")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"  Accuracy: {score:.3f}")
```

## Stratified K-Fold

**For imbalanced datasets** - maintains class distribution in each fold

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd

# Check class distribution
print("Original class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Regular KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\nRegular KFold - Class distribution per fold:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    y_val_fold = y[val_idx]
    dist = pd.Series(y_val_fold).value_counts(normalize=True).sort_index()
    print(f"Fold {fold}: {dist.values}")

# Stratified KFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nStratified KFold - Class distribution per fold:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    y_val_fold = y[val_idx]
    dist = pd.Series(y_val_fold).value_counts(normalize=True).sort_index()
    print(f"Fold {fold}: {dist.values}")

print("\n✓ Stratified KFold maintains class proportions!")

# Compare performance
model = RandomForestClassifier(n_estimators=100, random_state=42)

scores_regular = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
scores_stratified = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

print(f"\nRegular KFold:    {scores_regular.mean():.3f} (±{scores_regular.std():.3f})")
print(f"Stratified KFold: {scores_stratified.mean():.3f} (±{scores_stratified.std():.3f})")
```

## Leave-One-Out Cross-Validation (LOOCV)

**K = n** (number of samples)

```python
from sklearn.model_selection import LeaveOneOut

# Small dataset for demonstration
X_small, y_small = X[:50], y[:50]

loo = LeaveOneOut()
n_splits = loo.get_n_splits(X_small)

print(f"\nLeave-One-Out CV: {n_splits} splits")
print("Each split uses 1 sample for validation, rest for training")

# This can be slow for large datasets!
scores_loo = cross_val_score(model, X_small, y_small, cv=loo)

print(f"\nLOOCV Accuracy: {scores_loo.mean():.3f}")
print(f"Std: {scores_loo.std():.3f}")

print("\nWhen to use LOOCV:")
print("✓ Very small datasets (< 100 samples)")
print("✗ Large datasets (computationally expensive)")
```

## Group K-Fold

**For grouped data** - ensures samples from same group don't appear in both train and validation

```python
from sklearn.model_selection import GroupKFold
import numpy as np

# Example: Patient data with multiple samples per patient
patient_ids = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5] * 10)
X_patients = np.random.randn(len(patient_ids), 5)
y_patients = np.random.randint(0, 2, len(patient_ids))

print(f"\nDataset: {len(patient_ids)} samples from {len(np.unique(patient_ids))} patients")
print(f"Samples per patient: {pd.Series(patient_ids).value_counts().values}")

# Group KFold
gkf = GroupKFold(n_splits=3)

print("\nGroup KFold - Patient distribution:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_patients, y_patients, groups=patient_ids), 1):
    train_patients = np.unique(patient_ids[train_idx])
    val_patients = np.unique(patient_ids[val_idx])
    
    print(f"\nFold {fold}:")
    print(f"  Train patients: {sorted(train_patients)}")
    print(f"  Val patients: {sorted(val_patients)}")
    print(f"  Overlap: {set(train_patients).intersection(set(val_patients))}")

print("\n✓ No patient appears in both train and validation!")

# Use case examples
print("""
WHEN TO USE GROUP K-FOLD:

✓ Medical data: Multiple measurements per patient
✓ Time series: Multiple samples per time period
✓ Images: Multiple crops from same image
✓ Text: Multiple sentences from same document
✓ Any scenario where samples are not independent
""")
```

## Time Series Cross-Validation

**Preserves temporal order**

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Time series data
n_samples = 100
X_ts = np.arange(n_samples).reshape(-1, 1)
y_ts = np.sin(X_ts.ravel() / 10) + np.random.normal(0, 0.1, n_samples)

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Cross-Validation:")
print("\nEach split uses increasing amount of past data")

fig, axes = plt.subplots(5, 1, figsize=(12, 10))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_ts), 1):
    print(f"\nFold {fold}:")
    print(f"  Train: samples {train_idx[0]} to {train_idx[-1]} ({len(train_idx)} samples)")
    print(f"  Val: samples {val_idx[0]} to {val_idx[-1]} ({len(val_idx)} samples)")
    
    # Visualize
    ax = axes[fold-1]
    ax.plot(X_ts[train_idx], y_ts[train_idx], 'b.', label='Train', alpha=0.5)
    ax.plot(X_ts[val_idx], y_ts[val_idx], 'r.', label='Validation', alpha=0.5)
    ax.set_ylabel(f'Fold {fold}')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    if fold == 5:
        ax.set_xlabel('Time')

plt.tight_layout()
plt.suptitle('Time Series Cross-Validation', y=1.02, fontsize=14)
plt.show()

print("""
\nKEY PRINCIPLE:
  ✓ Train on past data
  ✓ Validate on future data
  ✓ Never train on future to predict past!
""")
```

## Cross-Validation with Multiple Metrics

```python
from sklearn.model_selection import cross_validate
import pandas as pd

# Multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validate with multiple metrics
results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\nCross-Validation Results (Multiple Metrics):")
print("\nTest Scores:")
for metric in scoring.keys():
    test_scores = results[f'test_{metric}']
    print(f"{metric:10s}: {test_scores.mean():.3f} (±{test_scores.std():.3f})")

print("\nTrain Scores:")
for metric in scoring.keys():
    train_scores = results[f'train_{metric}']
    print(f"{metric:10s}: {train_scores.mean():.3f} (±{train_scores.std():.3f})")

# Check for overfitting
print("\nOverfitting Check (Train - Test):")
for metric in scoring.keys():
    train_mean = results[f'train_{metric}'].mean()
    test_mean = results[f'test_{metric}'].mean()
    gap = train_mean - test_mean
    print(f"{metric:10s}: {gap:.3f}{'  ⚠️ High gap' if gap > 0.1 else '  ✓ OK'}")
```

## Choosing the Right K

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

# Test different K values
K_values = [2, 3, 5, 10, 20]
results = []

for k in K_values:
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    results.append({
        'K': k,
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Time_per_fold': 1/k  # Relative time
    })

results_df = pd.DataFrame(results)

print("\nEffect of K on Cross-Validation:")
print(results_df)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(results_df['K'], results_df['Mean'], yerr=results_df['Std'],
             marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('K (number of folds)')
ax1.set_ylabel('Mean Accuracy')
ax1.set_title('Accuracy vs K')
ax1.grid(alpha=0.3)

ax2.plot(results_df['K'], results_df['Std'], 'r-o', linewidth=2)
ax2.set_xlabel('K (number of folds)')
ax2.set_ylabel('Std of Accuracy')
ax2.set_title('Variance vs K')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
CHOOSING K:

• K=5 or K=10: Most common choices
• Larger K:
  ✓ More training data per fold
  ✓ Less bias
  ✗ Higher variance
  ✗ More computational cost
• Smaller K:
  ✓ Faster computation
  ✓ Lower variance
  ✗ More bias
  ✗ Less training data per fold

RECOMMENDATION:
- Small datasets (< 1000): K=5 or K=10
- Medium datasets: K=5
- Large datasets: K=3 or use train/val/test split
- Time series: TimeSeriesSplit
- Grouped data: GroupKFold
""")
```

## Common Pitfalls

```python
print("""
COMMON CROSS-VALIDATION MISTAKES:

1. DATA LEAKAGE:
   ✗ Scaling on entire dataset before CV
   ✓ Scale within each fold separately
   
2. USING TEST SET FOR TUNING:
   ✗ Repeatedly testing on same test set
   ✓ Use CV for tuning, test set only once
   
3. IGNORING DATA STRUCTURE:
   ✗ Using regular KFold for time series
   ✓ Use TimeSeriesSplit
   ✗ Using regular KFold for grouped data
   ✓ Use GroupKFold
   
4. IMBALANCED CLASSES:
   ✗ Using regular KFold
   ✓ Use StratifiedKFold
   
5. NOT SHUFFLING:
   ✗ KFold(n_splits=5)
   ✓ KFold(n_splits=5, shuffle=True, random_state=42)
   
6. WRONG METRIC:
   ✗ Using accuracy for imbalanced data
   ✓ Use F1, precision, recall, or ROC-AUC
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Cross-validation** provides reliable performance estimates
2. **K-fold CV** is the most common approach (K=5 or 10)
3. **Stratified K-fold** for imbalanced datasets
4. **Group K-fold** when samples are grouped
5. **Time Series Split** for temporal data
6. **Larger K** → less bias, more variance, higher cost
7. **Always shuffle** (except time series)
8. **Multiple metrics** give complete picture
9. **Avoid data leakage** - preprocess within folds
10. **Use CV for tuning**, test set for final evaluation only
:::

## Further Reading

- Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"
- Varma, S. & Simon, R. (2006). "Bias in Error Estimation When Using Cross-Validation for Model Selection"
- Scikit-learn: [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
