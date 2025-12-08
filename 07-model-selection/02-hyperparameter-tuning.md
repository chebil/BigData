# Hyperparameter Tuning

## Learning Objectives

- Understand hyperparameters vs parameters
- Implement grid search for hyperparameter tuning
- Apply random search for efficient exploration
- Use Bayesian optimization for advanced tuning
- Compare different tuning strategies
- Avoid overfitting during tuning

## Hyperparameters vs Parameters

```python
print("""
PARAMETERS vs HYPERPARAMETERS:

PARAMETERS:
  - Learned from data during training
  - Examples:
    • Weights in neural networks
    • Coefficients in linear regression
    • Split points in decision trees
  - Optimized by training algorithm

HYPERPARAMETERS:
  - Set BEFORE training
  - Examples:
    • Learning rate
    • Number of trees in random forest
    • Regularization strength (alpha)
    • Maximum depth of trees
  - Require manual tuning or automated search
  - Control model complexity and learning process
""")
```

## Grid Search

### Concept

**Exhaustive search** over specified parameter grid

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# Calculate total combinations
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations: {total_combinations}")
print(f"With 5-fold CV: {total_combinations * 5} model fits")

# Grid Search
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=2,
    return_train_score=True
)

print("\nPerforming Grid Search...")
grid_search.fit(X_train, y_train)

# Best parameters
print(f"\n" + "="*70)
print("GRID SEARCH RESULTS")
print("="*70)
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# Get results
results = pd.DataFrame(grid_search.cv_results_)

print(f"\nTop 5 parameter combinations:")
top_5 = results.sort_values('rank_test_score').head()[[
    'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
]]
for idx, row in top_5.iterrows():
    print(f"\nRank {int(row['rank_test_score'])}:")
    print(f"  Params: {row['params']}")
    print(f"  Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
```

### Visualizing Grid Search Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract results for visualization
results_viz = results[[
    'param_n_estimators', 'param_max_depth', 
    'mean_test_score', 'std_test_score'
]].copy()

# Convert to numeric
results_viz['param_max_depth'] = results_viz['param_max_depth'].fillna('None')

# Pivot for heatmap
pivot = results_viz.groupby(['param_n_estimators', 'param_max_depth'])['mean_test_score'].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('Grid Search: n_estimators vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.tight_layout()
plt.show()

# Line plot showing effect of each parameter
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

params_to_plot = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

for idx, param in enumerate(params_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    param_col = f'param_{param}'
    if param_col in results.columns:
        grouped = results.groupby(param_col).agg({
            'mean_test_score': 'mean',
            'std_test_score': 'mean'
        })
        
        x = grouped.index.astype(str)
        y = grouped['mean_test_score']
        yerr = grouped['std_test_score']
        
        ax.errorbar(range(len(x)), y, yerr=yerr, marker='o', capsize=5, linewidth=2)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45)
        ax.set_xlabel(param)
        ax.set_ylabel('Mean CV Score')
        ax.set_title(f'Effect of {param}')
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Random Search

**Samples random combinations** from parameter distributions

### Advantages over Grid Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

print("""
RANDOM SEARCH vs GRID SEARCH:

GRID SEARCH:
  ✓ Exhaustive
  ✓ Finds exact optimum in grid
  ✗ Expensive for many parameters
  ✗ Wastes time on unimportant parameters

RANDOM SEARCH:
  ✓ Explores wider range
  ✓ Better for high dimensions
  ✓ Can specify continuous distributions
  ✓ Finds good solutions faster
  ✗ May miss exact optimum
  ✗ Non-deterministic (use random_state)
""")

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),  # Integer from 50 to 299
    'max_depth': [None] + list(randint(5, 50).rvs(10)),  # None or 5-50
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)  # Continuous from 0.1 to 1.0
}

print("\nParameter Distributions:")
for param, dist in param_distributions.items():
    print(f"  {param}: {dist}")

# Random Search
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

print(f"\nPerforming Random Search ({random_search.n_iter} iterations)...")
random_search.fit(X_train, y_train)

# Results
print(f"\n" + "="*70)
print("RANDOM SEARCH RESULTS")
print("="*70)
print(f"\nBest parameters:")
for param, value in random_search.best_params_.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.4f}")
    else:
        print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {random_search.best_score_:.4f}")
print(f"Test score: {random_search.score(X_test, y_test):.4f}")

# Compare with Grid Search
print(f"\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Grid Search CV score:   {grid_search.best_score_:.4f}")
print(f"Random Search CV score: {random_search.best_score_:.4f}")
print(f"\nGrid Search iterations: {len(grid_search.cv_results_['params'])}")
print(f"Random Search iterations: {len(random_search.cv_results_['params'])}")
```

### Random Search Visualization

```python
# Visualize explored parameter space
results_random = pd.DataFrame(random_search.cv_results_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: n_estimators vs max_depth
ax = axes[0]
scatter = ax.scatter(
    results_random['param_n_estimators'],
    results_random['param_max_depth'].fillna(50),  # Replace None with 50 for plotting
    c=results_random['mean_test_score'],
    s=100,
    cmap='viridis',
    alpha=0.6
)
ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_title('Random Search: Parameter Space Exploration')
plt.colorbar(scatter, ax=ax, label='CV Score')
ax.grid(alpha=0.3)

# Best parameters highlighted
best_idx = results_random['rank_test_score'] == 1
ax.scatter(
    results_random.loc[best_idx, 'param_n_estimators'],
    results_random.loc[best_idx, 'param_max_depth'].fillna(50),
    s=300,
    marker='*',
    c='red',
    edgecolors='black',
    label='Best'
)
ax.legend()

# Score distribution
ax = axes[1]
ax.hist(results_random['mean_test_score'], bins=20, edgecolor='black', alpha=0.7)
ax.axvline(random_search.best_score_, color='red', linestyle='--', linewidth=2, label='Best')
ax.set_xlabel('CV Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Scores')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Bayesian Optimization

**Smart search** using probabilistic models

```python
# Install: pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search space
search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

print("Bayesian Optimization Search Space:")
for param, space in search_spaces.items():
    print(f"  {param}: {space}")

# Bayesian Search
rf = RandomForestClassifier(random_state=42)

bayes_search = BayesSearchCV(
    estimator=rf,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\nPerforming Bayesian Optimization...")
bayes_search.fit(X_train, y_train)

# Results
print(f"\n" + "="*70)
print("BAYESIAN OPTIMIZATION RESULTS")
print("="*70)
print(f"\nBest parameters:")
for param, value in bayes_search.best_params_.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.4f}")
    else:
        print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {bayes_search.best_score_:.4f}")
print(f"Test score: {bayes_search.score(X_test, y_test):.4f}")

# Compare all methods
print(f"\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
print(f"Grid Search:    {grid_search.best_score_:.4f}")
print(f"Random Search:  {random_search.best_score_:.4f}")
print(f"Bayesian Opt:   {bayes_search.best_score_:.4f}")
```

### Convergence Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract iteration history
fig, ax = plt.subplots(figsize=(12, 6))

# Random search
random_scores = results_random.sort_values('mean_fit_time')['mean_test_score'].values
random_best = np.maximum.accumulate(random_scores)

# Bayesian
bayes_results = pd.DataFrame(bayes_search.cv_results_)
bayes_scores = bayes_results.sort_values('mean_fit_time')['mean_test_score'].values
bayes_best = np.maximum.accumulate(bayes_scores)

# Plot
ax.plot(range(1, len(random_best)+1), random_best, 'b-o', label='Random Search', linewidth=2)
ax.plot(range(1, len(bayes_best)+1), bayes_best, 'r-s', label='Bayesian Optimization', linewidth=2)

ax.set_xlabel('Iteration')
ax.set_ylabel('Best Score Found')
ax.set_title('Convergence: Random Search vs Bayesian Optimization')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("""
CONVERGENCE INSIGHTS:

• Bayesian Optimization typically converges faster
• Uses past results to guide future sampling
• More efficient for expensive model training
• Random Search is simpler and often good enough
""")
```

## Nested Cross-Validation

**Avoid overfitting to validation set** during hyperparameter tuning

```python
from sklearn.model_selection import cross_val_score

print("""
NESTED CROSS-VALIDATION:

OUTER LOOP (Model Evaluation):
  For each of 5 outer folds:
    
    INNER LOOP (Hyperparameter Tuning):
      Use GridSearch/RandomSearch with CV
      Find best hyperparameters
    
    Train with best hyperparameters on outer train
    Evaluate on outer validation

RESULT: Unbiased estimate of model performance
""")

# Simple nested CV example
from sklearn.model_selection import KFold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Use grid search as the estimator
nested_scores = cross_val_score(
    grid_search, X, y, cv=outer_cv, scoring='accuracy'
)

print("\nNested Cross-Validation Scores:")
for i, score in enumerate(nested_scores, 1):
    print(f"Outer Fold {i}: {score:.4f}")

print(f"\nMean: {nested_scores.mean():.4f}")
print(f"Std:  {nested_scores.std():.4f}")

print("\nThis is the true generalization performance estimate!")
```

## Best Practices

```python
print("""
HYPERPARAMETER TUNING BEST PRACTICES:

1. START SIMPLE:
   ✓ Begin with default parameters
   ✓ Identify most important hyperparameters
   ✓ Tune one at a time initially

2. SEARCH STRATEGY:
   ✓ Small grids: Use Grid Search
   ✓ Many parameters: Use Random Search
   ✓ Expensive models: Use Bayesian Optimization

3. SEARCH SPACE:
   ✓ Start with wide ranges
   ✓ Refine based on results
   ✓ Use log scale for learning rates, regularization

4. VALIDATION:
   ✓ Use stratified K-fold for classification
   ✓ Use nested CV for unbiased estimates
   ✓ Keep separate test set

5. AVOID OVERFITTING:
   ✗ Don't tune too many hyperparameters
   ✗ Don't use test set for tuning
   ✓ Use nested CV
   ✓ Regularize

6. COMPUTATIONAL EFFICIENCY:
   ✓ Use n_jobs=-1 for parallel processing
   ✓ Start with fewer CV folds (3)
   ✓ Sample data for initial exploration
   ✓ Use early stopping when available

7. DOCUMENTATION:
   ✓ Save best parameters
   ✓ Record all experiments
   ✓ Version control configurations
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Hyperparameters** control model complexity and learning
2. **Grid Search** exhaustively searches parameter grid
3. **Random Search** samples randomly from distributions
4. **Bayesian Optimization** uses past results to guide search
5. **Random Search** often as good as Grid Search with less cost
6. **Bayesian** best for expensive models
7. **Nested CV** gives unbiased performance estimates
8. **Don't overfit** to validation set
9. **Start simple** then refine
10. **Document everything** for reproducibility
:::

## Further Reading

- Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization"
- Snoek, J. et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"
- Scikit-learn: [Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
