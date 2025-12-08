# Lab 7: Model Selection Exercises

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, cross_validate
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
```

## Part 1: Cross-Validation (25 points)

### Exercise 1.1: K-Fold CV (10 points)

```python
# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Dataset: {X.shape}")
print(f"Classes: {np.unique(y)}")

# TODO: Implement K-Fold CV
model = LogisticRegression(max_iter=10000, random_state=42)

# Try different K values
k_values = [3, 5, 10, 20]
results = []

for k in k_values:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    results.append({
        'K': k,
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Min': scores.min(),
        'Max': scores.max()
    })
    
    print(f"\nK={k}:")
    print(f"  Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df)

# Visualize
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['K'], results_df['Mean'], yerr=results_df['Std'], 
             marker='o', capsize=5, capthick=2)
plt.xlabel('Number of Folds (K)')
plt.ylabel('Accuracy')
plt.title('K-Fold Cross-Validation')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q1: How does K affect variance in scores?
- Q2: What K would you choose?
- Q3: Why not use K=2 or K=n?

### Exercise 1.2: Stratified K-Fold (8 points)

```python
# Compare regular vs stratified
print("\nClass distribution:")
print(y.value_counts(normalize=True))

# Regular K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_regular = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stratified = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"\nRegular K-Fold:")
print(f"  Mean: {scores_regular.mean():.4f} (+/- {scores_regular.std():.4f})")

print(f"\nStratified K-Fold:")
print(f"  Mean: {scores_stratified.mean():.4f} (+/- {scores_stratified.std():.4f})")

# Check class distribution in folds
print("\nClass distribution in Stratified K-Fold:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}: {y.iloc[val_idx].value_counts(normalize=True).values}")
```

**Questions:**
- Q4: When should you use stratified K-Fold?
- Q5: What's the difference in variance?
- Q6: Does stratification matter for balanced datasets?

### Exercise 1.3: Multiple Metrics (7 points)

```python
# TODO: Evaluate multiple metrics simultaneously
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(
    model, X, y, cv=5, scoring=scoring, return_train_score=True
)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    'Train': [
        cv_results['train_accuracy'].mean(),
        cv_results['train_precision'].mean(),
        cv_results['train_recall'].mean(),
        cv_results['train_f1'].mean(),
        cv_results['train_roc_auc'].mean()
    ],
    'Test': [
        cv_results['test_accuracy'].mean(),
        cv_results['test_precision'].mean(),
        cv_results['test_recall'].mean(),
        cv_results['test_f1'].mean(),
        cv_results['test_roc_auc'].mean()
    ]
})

print("\nCross-Validation Results:")
print(metrics_df.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_df))
width = 0.35

ax.bar(x - width/2, metrics_df['Train'], width, label='Train', alpha=0.8)
ax.bar(x + width/2, metrics_df['Test'], width, label='Test', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Cross-Validation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q7: Which metric shows most overfitting?
- Q8: Are train and test scores similar?

---

## Part 2: Hyperparameter Tuning (25 points)

### Exercise 2.1: Grid Search (15 points)

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TODO: Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', 
    n_jobs=-1, verbose=1, return_train_score=True
)

print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Test set performance
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Analyze results
results = pd.DataFrame(grid_search.cv_results_)
top_10 = results.nsmallest(10, 'rank_test_score')[[
    'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
]]

print("\nTop 10 configurations:")
print(top_10.to_string(index=False))
```

**Questions:**
- Q9: What are the optimal hyperparameters?
- Q10: How many configurations were tested?
- Q11: How long did Grid Search take?

### Exercise 2.2: Random Search (10 points)

```python
from scipy.stats import randint, uniform

# TODO: Random Search with distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=100, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1, random_state=42, return_train_score=True
)

print("Starting Random Search...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Compare with Grid Search
print("\nComparison:")
print(f"Grid Search - Best Score: {grid_search.best_score_:.4f}")
print(f"Random Search - Best Score: {random_search.best_score_:.4f}")
```

**Questions:**
- Q12: Did Random Search find better parameters?
- Q13: Which was faster?
- Q14: When would you use Random Search over Grid Search?

---

## Part 3: Bayesian Optimization (20 points)

```python
# Install: pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# TODO: Bayesian Optimization
search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

bayes_search = BayesSearchCV(
    rf, search_spaces, n_iter=50, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1, random_state=42
)

print("Starting Bayesian Optimization...")
bayes_search.fit(X_train, y_train)

print(f"\nBest parameters: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_:.4f}")

# Compare all methods
print("\n" + "="*60)
print("METHOD COMPARISON")
print("="*60)
print(f"{'Method':<20} {'Best Score':<15} {'Test Accuracy':<15}")
print("-" * 60)

for name, search in [('Grid Search', grid_search), 
                      ('Random Search', random_search),
                      ('Bayesian Opt', bayes_search)]:
    cv_score = search.best_score_
    test_score = accuracy_score(y_test, search.predict(X_test))
    print(f"{name:<20} {cv_score:<15.4f} {test_score:<15.4f}")
```

**Questions:**
- Q15: Which method found the best model?
- Q16: Which was most efficient?
- Q17: What are advantages of Bayesian Optimization?

---

## Part 4: Model Comparison (15 points)

```python
# TODO: Compare multiple algorithms
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

model_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    model_results.append({
        'Model': name,
        'Mean CV Score': scores.mean(),
        'Std CV Score': scores.std()
    })
    
    print(f"{name}:")
    print(f"  CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

results_df = pd.DataFrame(model_results).sort_values('Mean CV Score', ascending=False)
print("\nModel Rankings:")
print(results_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['Mean CV Score'], 
         xerr=results_df['Std CV Score'], alpha=0.7, capsize=5)
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q18: Which model performs best?
- Q19: Are differences statistically significant?
- Q20: Which model would you choose for production?

---

## Part 5: Automated Pipeline (15 points)

```python
import joblib

# TODO: Create complete ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for pipeline
pipeline_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

pipeline_search = GridSearchCV(
    pipeline, pipeline_params, cv=5, scoring='accuracy', n_jobs=-1
)

pipeline_search.fit(X_train, y_train)

print(f"Best pipeline score: {pipeline_search.best_score_:.4f}")
print(f"Best parameters: {pipeline_search.best_params_}")

# Save best model
joblib.dump(pipeline_search.best_estimator_, 'best_model_pipeline.pkl')
print("\nModel saved!")

# Load and test
loaded_pipeline = joblib.load('best_model_pipeline.pkl')
test_accuracy = accuracy_score(y_test, loaded_pipeline.predict(X_test))
print(f"Test accuracy: {test_accuracy:.4f}")
```

**Final Questions:**
- Q21: What are benefits of using pipelines?
- Q22: How would you add feature selection to the pipeline?
- Q23: What metrics would you monitor in production?
- Q24: How often should you retrain the model?
- Q25: What's your final model recommendation?

Good luck! 🎯
