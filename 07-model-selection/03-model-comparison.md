# Model Comparison and Selection

## Learning Objectives

- Compare multiple models systematically
- Use statistical tests for model comparison
- Create model comparison frameworks
- Interpret comparison results
- Select the best model for deployment
- Build production-ready ML pipelines

## Comparing Multiple Models

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

# Define models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=10000, random_state=42))
    ]),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
    'Naive Bayes': GaussianNB()
}

print(f"\nComparing {len(models)} models...")

# Multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validate all models
results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    results[name] = cv_results

# Compile results
comparison_data = []

for name, cv_results in results.items():
    row = {'Model': name}
    
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        row[f'{metric}_mean'] = test_scores.mean()
        row[f'{metric}_std'] = test_scores.std()
        row[f'{metric}_train'] = train_scores.mean()
        
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*90)
print("MODEL COMPARISON RESULTS")
print("="*90)
print(comparison_df[['Model', 'accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'roc_auc_mean']].to_string(index=False))

# Best model per metric
print("\n" + "="*90)
print("BEST MODEL PER METRIC")
print("="*90)
for metric in scoring.keys():
    best_idx = comparison_df[f'{metric}_mean'].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_score = comparison_df.loc[best_idx, f'{metric}_mean']
    print(f"{metric:12s}: {best_model:25s} ({best_score:.4f})")
```

### Visualization

```python
# Bar plot comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_list = list(scoring.keys())

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    
    models_sorted = comparison_df.sort_values(f'{metric}_mean', ascending=False)
    
    bars = ax.barh(models_sorted['Model'], models_sorted[f'{metric}_mean'],
                   xerr=models_sorted[f'{metric}_std'], capsize=5, alpha=0.7)
    
    # Color best model
    bars[0].set_color('green')
    bars[0].set_alpha(0.9)
    
    ax.set_xlabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.grid(alpha=0.3, axis='x')
    ax.set_xlim([0, 1])

axes[-1, -1].axis('off')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# Heatmap
plt.figure(figsize=(10, 8))
heatmap_data = comparison_df.set_index('Model')[[
    'accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'roc_auc_mean'
]]

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
plt.title('Model Performance Heatmap')
plt.tight_layout()
plt.show()

# Box plot for accuracy across folds
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

for name in comparison_df['Model']:
    box_data.append(results[name]['test_accuracy'])
    labels.append(name)

ax.boxplot(box_data, labels=labels, patch_artist=True)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Distribution Across CV Folds')
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## Statistical Significance Testing

### Paired t-test

```python
from scipy import stats

def compare_models_ttest(results1, results2, model1_name, model2_name, metric='test_accuracy'):
    """
    Compare two models using paired t-test
    """
    scores1 = results1[metric]
    scores2 = results2[metric]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    print(f"\nComparing {model1_name} vs {model2_name}:")
    print(f"  {model1_name} mean: {scores1.mean():.4f} (±{scores1.std():.4f})")
    print(f"  {model2_name} mean: {scores2.mean():.4f} (±{scores2.std():.4f})")
    print(f"  Difference: {(scores1.mean() - scores2.mean()):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        if scores1.mean() > scores2.mean():
            print(f"  ✓ {model1_name} is significantly better (p < 0.05)")
        else:
            print(f"  ✓ {model2_name} is significantly better (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")
    
    return t_stat, p_value

# Compare top two models
models_by_accuracy = comparison_df.sort_values('accuracy_mean', ascending=False)
top_model = models_by_accuracy.iloc[0]['Model']
second_model = models_by_accuracy.iloc[1]['Model']

compare_models_ttest(
    results[top_model],
    results[second_model],
    top_model,
    second_model
)
```

### McNemar's Test

```python
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import cross_val_predict

def mcnemar_test(model1, model2, X, y, model1_name, model2_name):
    """
    McNemar's test for comparing classifiers
    """
    # Get predictions
    y_pred1 = cross_val_predict(model1, X, y, cv=5)
    y_pred2 = cross_val_predict(model2, X, y, cv=5)
    
    # Create contingency table
    # both_correct, model1_only, model2_only, both_wrong
    both_correct = np.sum((y_pred1 == y) & (y_pred2 == y))
    model1_only = np.sum((y_pred1 == y) & (y_pred2 != y))
    model2_only = np.sum((y_pred1 != y) & (y_pred2 == y))
    both_wrong = np.sum((y_pred1 != y) & (y_pred2 != y))
    
    contingency = np.array([[both_correct, model1_only],
                           [model2_only, both_wrong]])
    
    print(f"\nMcNemar's Test: {model1_name} vs {model2_name}")
    print("\nContingency Table:")
    print(f"  Both correct:    {both_correct}")
    print(f"  {model1_name} only: {model1_only}")
    print(f"  {model2_name} only: {model2_only}")
    print(f"  Both wrong:      {both_wrong}")
    
    # McNemar's test
    result = mcnemar(contingency, exact=True)
    
    print(f"\n  Statistic: {result.statistic:.4f}")
    print(f"  p-value: {result.pvalue:.4f}")
    
    if result.pvalue < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")
    
    return result

# Compare models
mcnemar_test(
    models[top_model],
    models[second_model],
    X_train, y_train,
    top_model,
    second_model
)
```

## Final Model Selection

```python
print("\n" + "="*90)
print("FINAL MODEL SELECTION")
print("="*90)

# Select best model
best_model_name = comparison_df.loc[comparison_df['accuracy_mean'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\nSelected Model: {best_model_name}")

# Train on full training set
best_model.fit(X_train, y_train)

# Final test evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if y_pred_proba is not None:
    from sklearn.metrics import roc_auc_score
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save model
import joblib

model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
joblib.dump(best_model, model_filename)
print(f"\nModel saved to: {model_filename}")
```

## ML Pipelines

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Pipeline Steps:")
for name, step in pipeline.named_steps.items():
    print(f"  {name}: {step}")

# Train
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"\nPipeline Test Accuracy: {score:.4f}")
```

### Pipeline with GridSearch

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid for pipeline
param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

print("\nTuning pipeline...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.feature_indices]

# Custom pipeline
custom_pipeline = Pipeline([
    ('selector', FeatureSelector(feature_indices=[0, 1, 2, 3, 4])),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

custom_pipeline.fit(X_train, y_train)
print(f"\nCustom Pipeline Accuracy: {custom_pipeline.score(X_test, y_test):.4f}")
```

## Production Checklist

```python
print("""
MODEL DEPLOYMENT CHECKLIST:

1. MODEL SELECTION:
   ✓ Compared multiple models
   ✓ Used appropriate metrics
   ✓ Performed statistical tests
   ✓ Validated with nested CV

2. FINAL EVALUATION:
   ✓ Evaluated on hold-out test set
   ✓ Checked all relevant metrics
   ✓ Analyzed errors and edge cases
   ✓ Documented performance

3. MODEL ARTIFACTS:
   ✓ Saved trained model
   ✓ Saved preprocessing pipeline
   ✓ Saved feature names/order
   ✓ Saved hyperparameters

4. DOCUMENTATION:
   ✓ Model card with performance metrics
   ✓ Training data description
   ✓ Limitations and assumptions
   ✓ Intended use cases

5. MONITORING:
   ✓ Define performance thresholds
   ✓ Set up prediction logging
   ✓ Plan for retraining
   ✓ Monitor for data drift

6. REPRODUCIBILITY:
   ✓ Set random seeds
   ✓ Version control code
   ✓ Document dependencies
   ✓ Save data snapshots
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Compare multiple models** systematically
2. **Use multiple metrics** for comprehensive evaluation
3. **Statistical tests** determine significance of differences
4. **Paired t-test** for cross-validation scores
5. **McNemar's test** for classifier predictions
6. **Pipelines** ensure reproducible preprocessing
7. **GridSearch pipelines** tune entire workflow
8. **Custom transformers** for domain-specific preprocessing
9. **Document everything** for production deployment
10. **Test set** used only once for final evaluation
:::

## Further Reading

- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets"
- Dietterich, T. (1998). "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
- Scikit-learn: [Pipelines](https://scikit-learn.org/stable/modules/compose.html)
