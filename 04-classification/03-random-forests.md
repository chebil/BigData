# Random Forests

## Learning Objectives

- Understand ensemble learning and bagging
- Build and tune Random Forest classifiers
- Interpret feature importance in Random Forests
- Compare Random Forests to single decision trees
- Apply Random Forests to real-world problems
- Handle imbalanced data with Random Forests

## Introduction

Random Forests are **ensemble learning methods** that combine multiple decision trees to create a more robust and accurate classifier. They're one of the most popular machine learning algorithms due to their high performance and ease of use.

## Ensemble Learning

### Wisdom of Crowds

**Idea**: Multiple weak learners → strong learner

\[
\text{Final Prediction} = \text{Majority Vote}(\text{Tree}_1, \text{Tree}_2, \ldots, \text{Tree}_n)
\]

```python
import numpy as np

# Simulate 100 trees making predictions
np.random.seed(42)
n_trees = 100
n_samples = 1000

# Each tree is 60% accurate (weak learner)
individual_accuracy = 0.60

# Generate predictions
predictions = np.random.rand(n_trees, n_samples) < individual_accuracy
true_labels = np.ones(n_samples, dtype=bool)

# Majority vote
ensemble_predictions = np.sum(predictions, axis=0) > (n_trees / 2)

# Accuracy
individual_acc = np.mean(predictions[0] == true_labels)
ensemble_acc = np.mean(ensemble_predictions == true_labels)

print(f"Individual tree accuracy: {individual_acc:.2%}")
print(f"Ensemble accuracy: {ensemble_acc:.2%}")
print(f"Improvement: {ensemble_acc - individual_acc:.2%}")
```

## How Random Forests Work

### Algorithm

```
1. FOR each tree in forest:
   a. Bootstrap sample: Randomly sample data WITH replacement
   b. Random feature subset: At each split, consider random subset of features
   c. Build decision tree (fully grown or limited depth)

2. FOR prediction:
   a. Each tree makes a prediction
   b. Majority vote (classification) or average (regression)
```

### Key Components

**1. Bootstrap Aggregating (Bagging)**

```python
import numpy as np
import pandas as pd

# Original dataset
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'label': [0, 0, 1, 1, 1]
})

print("Original dataset:")
print(df)

# Bootstrap sample (with replacement)
np.random.seed(42)
bootstrap_sample = df.sample(n=len(df), replace=True)

print("\nBootstrap sample:")
print(bootstrap_sample)
print("\nNote: Some rows may appear multiple times, some not at all")
```

**2. Random Feature Subset**

```python
import numpy as np

# At each split, consider sqrt(n_features) random features
n_features = 10
n_features_per_split = int(np.sqrt(n_features))

print(f"Total features: {n_features}")
print(f"Features considered per split: {n_features_per_split}")

# Example random subset
all_features = list(range(n_features))
random_subset = np.random.choice(all_features, n_features_per_split, replace=False)

print(f"Random feature subset: {random_subset}")
```

## Implementation with Scikit-Learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth of each tree
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples in leaf
    max_features='sqrt',   # Features per split
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Random Forest Performance:")
print(f"Training Accuracy: {rf.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

## Random Forest vs Single Decision Tree

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Compare
print("Performance Comparison:")
print(f"Decision Tree - Train: {dt.score(X_train, y_train):.2%}, Test: {dt.score(X_test, y_test):.2%}")
print(f"Random Forest - Train: {rf.score(X_train, y_train):.2%}, Test: {rf.score(X_test, y_test):.2%}")

print(f"\nOverfitting:")
dt_overfit = dt.score(X_train, y_train) - dt.score(X_test, y_test)
rf_overfit = rf.score(X_train, y_train) - rf.score(X_test, y_test)
print(f"Decision Tree: {dt_overfit:.2%}")
print(f"Random Forest: {rf_overfit:.2%}")
print(f"\nRandom Forest reduces overfitting by {dt_overfit - rf_overfit:.2%}")
```

## Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': importances,
    'Std': std
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], 
         xerr=top_features['Std'], alpha=0.7)
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest CV score: {grid_search.best_score_:.2%}")

# Test best model
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Test accuracy: {test_score:.2%}")
```

### Random Search (Faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random search
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=50,  # Number of parameter settings to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best parameters:")
print(random_search.best_params_)
print(f"Best CV score: {random_search.best_score_:.2%}")
```

## Out-of-Bag (OOB) Error

**Out-of-Bag samples**: ~37% of samples not used in each tree's bootstrap

Can be used for validation without separate test set!

```python
from sklearn.ensemble import RandomForestClassifier

# Enable OOB score
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Calculate OOB error
    random_state=42
)

rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.2%}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.2%}")
print(f"\nOOB score approximates test performance!")
```

## Handling Imbalanced Data

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from collections import Counter

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                          n_informative=10, random_state=42)

print(f"Class distribution: {Counter(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Solution 1: Class weights
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically adjust weights
    random_state=42
)

rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)

print("\nWith Balanced Class Weights:")
print(classification_report(y_test, y_pred_weighted))

# Solution 2: Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42
)

brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

print("\nBalanced Random Forest:")
print(classification_report(y_test, y_pred_brf))
```

## Complete Example: Credit Card Fraud Detection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('data/creditcard.csv')

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nFraud cases: {df['Class'].value_counts()}")
print(f"Fraud percentage: {df['Class'].mean():.2%}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Class 0: {(y_train_balanced == 0).sum():,}")
print(f"Class 1: {(y_train_balanced == 1).sum():,}")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Random Forest...")
rf.fit(X_train_balanced, y_train_balanced)

# Cross-validation on original (imbalanced) training data
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nCV AUC-ROC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Comprehensive evaluation
print("\n" + "="*70)
print("MODEL PERFORMANCE ON TEST SET")
print("="*70)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives: {cm[1,1]:,}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0,0], cmap='Blues')
axes[0,0].set_title('Confusion Matrix')

# 2. ROC Curve
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[0,1])
axes[0,1].set_title('ROC Curve')
axes[0,1].grid(alpha=0.3)

# 3. Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=axes[1,0])
axes[1,0].set_title('Precision-Recall Curve')
axes[1,0].grid(alpha=0.3)

# 4. Feature Importance (top 15)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

axes[1,1].barh(range(15), importances[indices])
axes[1,1].set_yticks(range(15))
axes[1,1].set_yticklabels([X.columns[i] for i in indices])
axes[1,1].set_xlabel('Importance')
axes[1,1].set_title('Top 15 Feature Importances')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig('fraud_detection_analysis.png', dpi=300)
plt.show()

# Business metrics
total_fraud_amount = df[df['Class'] == 1]['Amount'].sum()
detected_fraud_indices = X_test.index[(y_test == 1) & (y_pred == 1)]
detected_fraud_amount = df.loc[detected_fraud_indices, 'Amount'].sum()

print("\n" + "="*70)
print("BUSINESS IMPACT")
print("="*70)
print(f"Total fraud amount in test set: ${total_fraud_amount:,.2f}")
print(f"Detected fraud amount: ${detected_fraud_amount:,.2f}")
print(f"Detection rate: {detected_fraud_amount/total_fraud_amount:.1%}")

false_positives = cm[0, 1]
avg_transaction = df['Amount'].mean()
investigation_cost_per_fp = 25  # Assumed cost

print(f"\nFalse positives: {false_positives:,}")
print(f"Estimated investigation cost: ${false_positives * investigation_cost_per_fp:,.2f}")

# Save model
import joblib
joblib.dump(rf, 'fraud_detection_model.pkl')
print("\nModel saved to 'fraud_detection_model.pkl'")

# Make predictions on new data
def predict_fraud(transaction_features):
    """
    Predict if transaction is fraudulent
    """
    proba = rf.predict_proba([transaction_features])[0][1]
    prediction = 'FRAUD' if proba > 0.5 else 'LEGITIMATE'
    
    return {
        'prediction': prediction,
        'fraud_probability': f"{proba:.1%}",
        'risk_level': 'HIGH' if proba > 0.7 else 'MEDIUM' if proba > 0.3 else 'LOW'
    }

# Example
sample_transaction = X_test.iloc[0].values
result = predict_fraud(sample_transaction)
print(f"\nSample prediction: {result}")
```

## Advantages and Disadvantages

### Advantages ✅

1. **High accuracy** - Often best off-the-shelf performance
2. **Reduces overfitting** compared to single trees
3. **Handles missing values** well
4. **No feature scaling needed**
5. **Feature importance** built-in
6. **Parallel training** (fast with multiple cores)
7. **Works for classification and regression**
8. **Robust to outliers**

### Disadvantages ❌

1. **Less interpretable** than single decision tree
2. **Slower prediction** than single tree
3. **Larger model size** (memory intensive)
4. **Can overfit on noisy data**
5. **Biased toward features with more categories**
6. **Not great for very high-dimensional sparse data** (e.g., text)

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Random Forests = Ensemble of decision trees**
2. **Bagging** reduces variance and overfitting
3. **Random feature selection** decorrelates trees
4. **More trees → better performance** (diminishing returns)
5. **Default hyperparameters often work well**
6. **Feature importance** helps interpretation
7. **OOB error** provides validation without test set
8. **Class weighting** handles imbalanced data
9. **Generally outperforms single decision trees**
10. **Good baseline** for many classification problems
:::

## Further Reading

- Breiman, L. (2001). "Random Forests", Machine Learning 45(1)
- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 15
- Scikit-learn Random Forests: [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
