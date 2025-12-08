# Regularization in Regression

## Learning Objectives

- Understand the problem of overfitting in regression
- Implement Ridge Regression (L2 regularization)
- Implement Lasso Regression (L1 regularization)
- Apply Elastic Net (L1 + L2 combination)
- Compare regularization techniques
- Tune regularization parameters
- Interpret regularized models

## Introduction

Regularization adds a penalty term to the loss function to prevent overfitting by constraining coefficient magnitudes. It's especially useful when:

- Number of features is large
- Features are highly correlated (multicollinearity)
- Model is overfitting training data

## The Overfitting Problem

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate data with noise
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2*X.ravel() + 3*np.sin(X.ravel()) + np.random.normal(0, 1, 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit polynomial models of increasing degree
degrees = [1, 3, 10, 20]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, degree in enumerate(degrees):
    ax = axes[idx // 2, idx % 2]
    
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
    test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
    
    # Plot
    ax.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
    ax.scatter(X_test, y_test, color='red', alpha=0.5, label='Test')
    ax.plot(X_plot, y_plot, 'g-', linewidth=2, label='Model')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([y.min()-5, y.max()+5])

plt.tight_layout()
plt.show()

print("Notice: High degree polynomials overfit!")
print("Train error decreases, but test error increases")
```

## Ridge Regression (L2 Regularization)

### Mathematical Foundation

**Objective Function**:
\[
\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \right\}
\]

**Simplified**:
\[
\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|^2 \right\}
\]

where \(\alpha \geq 0\) is the regularization parameter

**Closed-Form Solution**:
\[
\boldsymbol{\beta}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
\]

### Properties

- **Shrinks coefficients** toward zero (but never exactly to zero)
- **Handles multicollinearity** well
- **All features retained**
- **Biased estimates** but lower variance

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_scores = []
test_scores = []
coefficients = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    train_scores.append(ridge.score(X_train_scaled, y_train))
    test_scores.append(ridge.score(X_test_scaled, y_test))
    coefficients.append(ridge.coef_)

# Plot R-squared vs alpha
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(alphas, train_scores, 'b-o', label='Train R²', linewidth=2)
ax1.plot(alphas, test_scores, 'r-o', label='Test R²', linewidth=2)
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (regularization strength)')
ax1.set_ylabel('R-squared')
ax1.set_title('Ridge Regression: R² vs Alpha')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot coefficients vs alpha
coefficients = np.array(coefficients)
for i in range(coefficients.shape[1]):
    ax2.plot(alphas, coefficients[:, i], label=data.feature_names[i])

ax2.set_xscale('log')
ax2.set_xlabel('Alpha (regularization strength)')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Ridge Coefficients vs Alpha')
ax2.legend(loc='best', fontsize=8)
ax2.grid(alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()

print("Observations:")
print("- As alpha increases, coefficients shrink toward zero")
print("- Too high alpha → underfitting (both scores decrease)")
print("- Too low alpha → like ordinary least squares")
```

### Choosing Optimal Alpha

```python
from sklearn.linear_model import RidgeCV

# Cross-validation to find best alpha
alphas = np.logspace(-3, 3, 50)

ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha: {ridge_cv.alpha_:.3f}")
print(f"Train R²: {ridge_cv.score(X_train_scaled, y_train):.3f}")
print(f"Test R²: {ridge_cv.score(X_test_scaled, y_test):.3f}")

# Compare with OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

print(f"\nOrdinary Least Squares:")
print(f"Train R²: {ols.score(X_train_scaled, y_train):.3f}")
print(f"Test R²: {ols.score(X_test_scaled, y_test):.3f}")

print(f"\nImprovement: {ridge_cv.score(X_test_scaled, y_test) - ols.score(X_test_scaled, y_test):.3f}")
```

## Lasso Regression (L1 Regularization)

### Mathematical Foundation

**Objective Function**:
\[
\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \alpha \sum_{j=1}^{p} |\beta_j| \right\}
\]

**Simplified**:
\[
\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|_1 \right\}
\]

### Properties

- **Performs feature selection** (sets some coefficients exactly to zero)
- **Sparse solutions**
- **No closed-form solution** (requires iterative optimization)
- **Useful when many features are irrelevant**

### Implementation

```python
from sklearn.linear_model import Lasso, LassoCV
import numpy as np
import matplotlib.pyplot as plt

# Try different alpha values
alphas_lasso = [0.001, 0.01, 0.1, 1, 10]

coefficients_lasso = []
n_nonzero = []

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    coefficients_lasso.append(lasso.coef_)
    n_nonzero.append(np.sum(lasso.coef_ != 0))
    
    print(f"\nAlpha = {alpha}:")
    print(f"  Non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")
    print(f"  Test R²: {lasso.score(X_test_scaled, y_test):.3f}")

# Plot coefficients
coefficients_lasso = np.array(coefficients_lasso)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Coefficient paths
for i in range(coefficients_lasso.shape[1]):
    ax1.plot(alphas_lasso, coefficients_lasso[:, i], 
             marker='o', label=data.feature_names[i])

ax1.set_xscale('log')
ax1.set_xlabel('Alpha (regularization strength)')
ax1.set_ylabel('Coefficient Value')
ax1.set_title('Lasso Coefficients vs Alpha (Feature Selection)')
ax1.legend(loc='best', fontsize=8)
ax1.grid(alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

# Number of non-zero coefficients
ax2.plot(alphas_lasso, n_nonzero, 'bo-', linewidth=2, markersize=8)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (regularization strength)')
ax2.set_ylabel('Number of Non-Zero Coefficients')
ax2.set_title('Feature Selection: Active Features vs Alpha')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nObservations:")
print("- Lasso sets coefficients exactly to zero")
print("- Performs automatic feature selection")
print("- High alpha → fewer features retained")
```

### Optimal Alpha with Cross-Validation

```python
# LassoCV with automatic alpha selection
alphas = np.logspace(-4, 1, 50)

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nLasso Cross-Validation Results:")
print(f"Best alpha: {lasso_cv.alpha_:.4f}")
print(f"Train R²: {lasso_cv.score(X_train_scaled, y_train):.3f}")
print(f"Test R²: {lasso_cv.score(X_test_scaled, y_test):.3f}")
print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}/{len(lasso_cv.coef_)}")

# Selected features
selected_features = np.array(data.feature_names)[lasso_cv.coef_ != 0]
print(f"\nSelected features: {list(selected_features)}")

# Coefficient values
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': lasso_cv.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nLasso Coefficients:")
print(coef_df)
```

## Elastic Net (L1 + L2 Combination)

### Mathematical Foundation

**Objective Function**:
\[
\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \left( \rho \|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\beta}\|^2 \right) \right\}
\]

where:
- \(\alpha\) = overall regularization strength
- \(\rho\) = mixing parameter (0 = Ridge, 1 = Lasso)

### Properties

- **Combines Ridge and Lasso**
- **Performs feature selection** like Lasso
- **Handles correlated features** better than Lasso
- **More stable** than Lasso with correlated features

### Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pandas as pd

# ElasticNet with different l1_ratio values
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

results = []

for l1_ratio in l1_ratios:
    enet = ElasticNetCV(l1_ratio=l1_ratio, alphas=np.logspace(-4, 1, 50),
                        cv=5, max_iter=10000, random_state=42)
    enet.fit(X_train_scaled, y_train)
    
    results.append({
        'l1_ratio': l1_ratio,
        'best_alpha': enet.alpha_,
        'train_r2': enet.score(X_train_scaled, y_train),
        'test_r2': enet.score(X_test_scaled, y_test),
        'n_nonzero': np.sum(enet.coef_ != 0)
    })

results_df = pd.DataFrame(results)

print("\nElastic Net: Different L1 Ratios")
print(results_df)

print("\nInterpretation:")
print("- l1_ratio = 0.0 → Pure Ridge (all features, shrinkage only)")
print("- l1_ratio = 1.0 → Pure Lasso (feature selection)")
print("- 0 < l1_ratio < 1 → Mix of both")

# Best model
best_idx = results_df['test_r2'].idxmax()
best_params = results_df.loc[best_idx]

print(f"\nBest ElasticNet:")
print(f"  L1 ratio: {best_params['l1_ratio']}")
print(f"  Alpha: {best_params['best_alpha']:.4f}")
print(f"  Test R²: {best_params['test_r2']:.3f}")
print(f"  Active features: {best_params['n_nonzero']}/{len(data.feature_names)}")
```

## Comprehensive Comparison

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Train all models
models = {
    'OLS': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    'Lasso': LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000),
    'ElasticNet': ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-4, 1, 50), 
                               cv=5, max_iter=10000)
}

comparison_results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Test performance
    y_pred = model.predict(X_test_scaled)
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
    else:
        n_nonzero = len(data.feature_names)
    
    comparison_results.append({
        'Model': name,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std(),
        'Test R²': r2_score(y_test, y_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Active Features': n_nonzero
    })

comparison_df = pd.DataFrame(comparison_results)

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R-squared comparison
axes[0].bar(comparison_df['Model'], comparison_df['Test R²'], 
           yerr=comparison_df['CV R² Std'], capsize=5, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('R-squared')
axes[0].set_title('Test R² Comparison')
axes[0].grid(alpha=0.3, axis='y')
axes[0].set_ylim([0, 1])

# RMSE comparison
axes[1].bar(comparison_df['Model'], comparison_df['Test RMSE'], 
           alpha=0.7, edgecolor='black', color='orange')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Test RMSE Comparison (Lower is Better)')
axes[1].grid(alpha=0.3, axis='y')

# Active features
axes[2].bar(comparison_df['Model'], comparison_df['Active Features'], 
           alpha=0.7, edgecolor='black', color='green')
axes[2].set_ylabel('Number of Features')
axes[2].set_title('Active Features (Sparsity)')
axes[2].grid(alpha=0.3, axis='y')
axes[2].axhline(y=len(data.feature_names), color='red', 
               linestyle='--', label='Total features')
axes[2].legend()

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300)
plt.show()
```

## When to Use Which?

```python
print("""
REGULARIZATION SELECTION GUIDE:

1. RIDGE REGRESSION:
   ✓ Many features, most are relevant
   ✓ Multicollinearity present
   ✓ Want to keep all features
   ✓ Prediction accuracy is priority
   
2. LASSO REGRESSION:
   ✓ Many features, many are irrelevant
   ✓ Want automatic feature selection
   ✓ Interpretability important
   ✓ Sparse solution desired
   
3. ELASTIC NET:
   ✓ Many correlated features
   ✓ Want feature selection + stability
   ✓ Best of both Ridge and Lasso
   ✓ When unsure, try this first!
   
4. ORDINARY LEAST SQUARES:
   ✓ Few features (p << n)
   ✓ No multicollinearity
   ✓ Model assumptions met
   ✓ Interpretability critical
""")
```

## Complete Example: High-Dimensional Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Generate high-dimensional data (p > n scenario)
n_samples = 100
n_features = 200
n_informative = 20  # Only 20 features are actually useful

X, y, true_coef = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=10,
    coef=True,
    random_state=42
)

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Informative features: {n_informative}")
print(f"True non-zero coefficients: {np.sum(true_coef != 0)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    results[name] = {
        'model': model,
        'r2': r2_score(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'n_nonzero': np.sum(np.abs(model.coef_) > 1e-5)
    }
    
    print(f"\n{name}:")
    print(f"  Test R²: {results[name]['r2']:.3f}")
    print(f"  Test RMSE: {results[name]['rmse']:.2f}")
    print(f"  Non-zero coefficients: {results[name]['n_nonzero']}/{n_features}")

# Compare coefficients
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    coef = result['model'].coef_
    
    ax.stem(range(len(coef)), coef, basefmt=' ', use_line_collection=True)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{name}\n({result["n_nonzero"]} non-zero coefficients)')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print("\nConclusion:")
print("- Lasso performs best feature selection (most sparse)")
print("- Ridge keeps all features but shrinks them")
print("- ElasticNet provides balance")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Regularization prevents overfitting** by penalizing large coefficients
2. **Ridge (L2)**: Shrinks coefficients, keeps all features
3. **Lasso (L1)**: Feature selection, sets coefficients to zero
4. **Elastic Net**: Combination of Ridge and Lasso
5. **Alpha (\(\alpha\))**: Controls regularization strength
6. **Higher \(\alpha\)** → more regularization → simpler model
7. **Cross-validation** essential for choosing \(\alpha\)
8. **Feature scaling** required for fair regularization
9. **Lasso for interpretability**, Ridge for prediction
10. **ElasticNet** often best default choice
:::

## Further Reading

- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 3.4
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
- Zou, H. & Hastie, T. (2005). "Regularization and Variable Selection via the Elastic Net"
- Scikit-learn: [Linear Models with Regularization](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
