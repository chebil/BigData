# Polynomial Regression

## Learning Objectives

- Understand polynomial regression and feature transformation
- Implement polynomial regression in Python
- Choose appropriate polynomial degree
- Avoid overfitting with regularization
- Apply polynomial regression to real-world problems
- Combine polynomials with interaction terms

## Introduction

Polynomial regression extends linear regression by adding polynomial features, allowing the model to capture **non-linear relationships** between features and target while still using linear regression techniques.

## Mathematical Foundation

### Simple Polynomial Regression

**Degree 1 (Linear)**:
\[
y = \beta_0 + \beta_1 x
\]

**Degree 2 (Quadratic)**:
\[
y = \beta_0 + \beta_1 x + \beta_2 x^2
\]

**Degree 3 (Cubic)**:
\[
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
\]

**Degree d (General)**:
\[
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d
\]

### Multiple Features with Polynomials

For features \(x_1, x_2\) with degree 2:
\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_1 x_2 + \beta_5 x_2^2
\]

**Includes interaction terms** (\(x_1 x_2\))

## Basic Implementation

### From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def create_polynomial_features(X, degree):
    """
    Create polynomial features up to specified degree
    """
    n_samples = X.shape[0]
    features = [np.ones(n_samples)]  # Intercept
    
    for d in range(1, degree + 1):
        features.append(X ** d)
    
    return np.column_stack(features)

# Generate data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, 100)

# Create polynomial features (degree 2)
X_poly = create_polynomial_features(X, degree=2)

print(f"Original shape: {X.shape}")
print(f"Polynomial features shape: {X_poly.shape}")
print(f"\nFirst few rows:")
print(X_poly[:5])
print("\nColumns: [1, x, x^2]")

# Fit using normal equation
beta = np.linalg.lstsq(X_poly, y, rcond=None)[0]

print(f"\nCoefficients: {beta}")
print(f"Model: y = {beta[0]:.2f} + {beta[1]:.2f}x + {beta[2]:.2f}x^2")

# Predictions
y_pred = X_poly @ beta

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Polynomial fit (degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Calculate R-squared
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r2:.3f}")
```

### Using Scikit-Learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + X.ravel() + 2 + np.random.normal(0, 1, 100)

# Compare different polynomial degrees
degrees = [1, 2, 3, 5, 10]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    # Create polynomial features and fit
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predictions
    y_pred = model.predict(X_poly)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Plot
    ax = axes[idx]
    ax.scatter(X, y, alpha=0.5, label='Data')
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}\nR²={r2:.3f}, RMSE={rmse:.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([y.min()-2, y.max()+2])

# Hide extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

print("Observations:")
print("- Degree 1: Underfits (linear can't capture parabola)")
print("- Degree 2: Perfect fit (matches true relationship)")
print("- Degree 10: Overfits (too flexible)")
```

## Using Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Pipeline (Degree 3 + Ridge):")
print(f"  Train R²: {train_score:.3f}")
print(f"  Test R²: {test_score:.3f}")

print("\nPipeline steps:")
for name, step in pipeline.named_steps.items():
    print(f"  {name}: {step}")
```

## Choosing Optimal Degree

### Validation Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve

# Create pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression())
])

# Test different degrees
degrees = range(1, 15)

train_scores, val_scores = validation_curve(
    pipeline, X, y,
    param_name='poly__degree',
    param_range=degrees,
    cv=5,
    scoring='r2'
)

# Calculate means and stds
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(degrees, train_mean - train_std, train_mean + train_std, 
                 alpha=0.2, color='blue')

plt.plot(degrees, val_mean, 'o-', color='red', label='Validation score')
plt.fill_between(degrees, val_mean - val_std, val_mean + val_std, 
                 alpha=0.2, color='red')

plt.xlabel('Polynomial Degree')
plt.ylabel('R-squared')
plt.title('Validation Curve: Choosing Optimal Polynomial Degree')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.axvline(x=2, color='green', linestyle='--', label='Optimal degree')
plt.show()

# Find optimal degree
optimal_degree = degrees[np.argmax(val_mean)]
print(f"\nOptimal degree: {optimal_degree}")
print(f"Validation R²: {val_mean[optimal_degree-1]:.3f}")
```

### Cross-Validation for Degree Selection

```python
from sklearn.model_selection import cross_val_score
import pandas as pd

# Test different degrees
results = []

for degree in range(1, 11):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    results.append({
        'Degree': degree,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Min Score': cv_scores.min(),
        'Max Score': cv_scores.max()
    })

results_df = pd.DataFrame(results)

print("\nCross-Validation Results:")
print(results_df.to_string(index=False))

# Best degree
best = results_df.loc[results_df['CV Mean'].idxmax()]
print(f"\nBest degree: {int(best['Degree'])} (CV R² = {best['CV Mean']:.3f})")
```

## Multivariate Polynomial Regression

### With Interaction Terms

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate data with two features
np.random.seed(42)
n_samples = 200

X1 = np.random.uniform(-3, 3, n_samples)
X2 = np.random.uniform(-3, 3, n_samples)

# True relationship: y = 2*X1 + 3*X2 + 1.5*X1^2 - 0.5*X1*X2
y = 2*X1 + 3*X2 + 1.5*X1**2 - 0.5*X1*X2 + np.random.normal(0, 1, n_samples)

X = np.column_stack([X1, X2])

print(f"Original features: {X.shape}")

# Create polynomial features (degree 2, with interactions)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

print(f"Polynomial features: {X_poly.shape}")
print(f"\nFeature names: {poly.get_feature_names_out(['X1', 'X2'])}")
print("\nFeatures include: 1, X1, X2, X1^2, X1*X2, X2^2")

# Fit model
model = LinearRegression()
model.fit(X_poly, y)

# Coefficients
coef_df = pd.DataFrame({
    'Feature': poly.get_feature_names_out(['X1', 'X2']),
    'Coefficient': np.concatenate([[model.intercept_], model.coef_[1:]])
})

print("\nLearned Coefficients:")
print(coef_df)

print("\nNote: Coefficients should be close to:")
print("  X1: 2.0")
print("  X2: 3.0")
print("  X1^2: 1.5")
print("  X1*X2: -0.5")

# Evaluate
y_pred = model.predict(X_poly)
print(f"\nR-squared: {r2_score(y, y_pred):.3f}")
```

### Without Interaction Terms

```python
# Polynomial features WITHOUT interactions
poly_no_interaction = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)

# Actually, to exclude interactions completely, we need custom transformation
from sklearn.preprocessing import FunctionTransformer

def add_polynomial_no_interaction(X, degree=2):
    """
    Add polynomial features without interaction terms
    """
    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)
    return np.hstack(features)

X_poly_no_int = add_polynomial_no_interaction(X, degree=2)

print(f"\nPolynomial features (no interactions): {X_poly_no_int.shape}")
print("Features: X1, X2, X1^2, X2^2 (no X1*X2)")

model_no_int = LinearRegression()
model_no_int.fit(X_poly_no_int, y)

print(f"R-squared (with interactions): {r2_score(y, y_pred):.3f}")
print(f"R-squared (without interactions): {model_no_int.score(X_poly_no_int, y):.3f}")

print("\nConclusion: Interaction terms improve fit when present in true relationship")
```

## Complete Example: California Housing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("California Housing Dataset:")
print(f"Features: {list(X.columns)}")
print(f"Shape: {X.shape}")
print(f"Target: Median house value")

# Use subset of features for demonstration
feature_subset = ['MedInc', 'HouseAge', 'AveRooms']
X_subset = X[feature_subset]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y, test_size=0.2, random_state=42
)

# Compare linear vs polynomial
models = {
    'Linear': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'Polynomial (degree 2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'Polynomial (degree 3)': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10.0))  # More regularization for degree 3
    ])
}

results = []

for name, pipeline in models.items():
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    # Train and test
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    results.append({
        'Model': name,
        'CV R²': f"{cv_scores.mean():.3f} (±{cv_scores.std():.3f})",
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test)
    })

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("MODEL COMPARISON: Linear vs Polynomial")
print("="*80)
print(results_df.to_string(index=False))

# Feature importance for degree 2 model
poly_model = models['Polynomial (degree 2)']
poly = poly_model.named_steps['poly']
ridge = poly_model.named_steps['model']

feature_names = poly.get_feature_names_out(feature_subset)
coefficients = ridge.coef_

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n" + "="*80)
print("POLYNOMIAL FEATURES (Degree 2) - Top 10 by Absolute Coefficient")
print("="*80)
print(coef_df.head(10).to_string(index=False))

# Visualize predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, pipeline) in enumerate(models.items()):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.3f}')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_regression_comparison.png', dpi=300)
plt.show()
```

## Regularization with Polynomial Features

**Critical**: High-degree polynomials need regularization!

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Create pipeline with polynomial features
base_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# Grid search for best alpha
param_grid = {
    'model__alpha': np.logspace(-2, 3, 20)
}

grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['model__alpha']:.3f}")
print(f"Best CV R²: {grid_search.best_score_:.3f}")
print(f"Test R²: {grid_search.score(X_test, y_test):.3f}")

# Compare Ridge, Lasso, ElasticNet
regularizers = {
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

for name, model in regularizers.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    n_features = len(pipeline.named_steps['poly'].get_feature_names_out(feature_subset))
    n_nonzero = np.sum(np.abs(pipeline.named_steps['model'].coef_) > 1e-5)
    
    print(f"\n{name}:")
    print(f"  Test R²: {test_score:.3f}")
    print(f"  Active features: {n_nonzero}/{n_features}")
```

## Advantages and Disadvantages

### Advantages ✅

1. **Captures non-linear relationships** with linear methods
2. **Easy to implement** using existing tools
3. **Interpretable** (with low degrees)
4. **Flexible** - can model complex patterns

### Disadvantages ❌

1. **Overfitting risk** with high degrees
2. **Feature explosion** with multiple variables
3. **Extrapolation problems** outside training range
4. **Multicollinearity** among polynomial terms
5. **Requires regularization** for stability

## Best Practices

```python
print("""
POLYNOMIAL REGRESSION BEST PRACTICES:

1. FEATURE SCALING:
   ✓ Always standardize features before creating polynomials
   ✓ Prevents numerical instability

2. DEGREE SELECTION:
   ✓ Start with degree 2 or 3
   ✓ Use cross-validation to choose optimal degree
   ✓ Higher degrees → more regularization needed

3. REGULARIZATION:
   ✓ Always use Ridge/Lasso/ElasticNet with polynomials
   ✓ Especially critical for degree > 2
   ✓ Prevents overfitting

4. FEATURE ENGINEERING:
   ✓ Consider domain knowledge for interactions
   ✓ Not all interactions may be meaningful
   ✓ Use Lasso for automatic feature selection

5. VALIDATION:
   ✓ Check for overfitting (train vs test performance)
   ✓ Visualize predictions
   ✓ Be cautious about extrapolation

6. ALTERNATIVES:
   ✓ For very non-linear data, consider:
     - Splines (more flexible)
     - Tree-based models (Random Forest, XGBoost)
     - Neural networks
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Polynomial regression models non-linear relationships** using linear methods
2. **Feature transformation**: \(x \rightarrow [1, x, x^2, \ldots, x^d]\)
3. **Interaction terms** (\(x_1 x_2\)) often improve fit
4. **Degree selection** via cross-validation critical
5. **Always use regularization** with polynomials
6. **Feature scaling** essential before creating polynomials
7. **Trade-off**: Flexibility vs overfitting
8. **Use pipelines** for clean, reproducible workflow
9. **Low degrees (2-3)** usually sufficient
10. **Beware extrapolation** - predictions unreliable outside training range
:::

## Further Reading

- James, G. et al. (2013). "An Introduction to Statistical Learning", Chapter 7
- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 5
- Scikit-learn Polynomial Features: [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
