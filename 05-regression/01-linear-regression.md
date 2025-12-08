# Linear Regression

## Learning Objectives

- Understand the theory behind linear regression
- Implement simple and multiple linear regression
- Interpret regression coefficients and statistics
- Assess model assumptions and diagnostics
- Handle multicollinearity
- Build and evaluate regression models in Python

## Introduction

Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation. It's one of the most fundamental and widely used statistical techniques.

## Simple Linear Regression

### Mathematical Foundation

**Model**:
\[
y = \beta_0 + \beta_1 x + \epsilon
\]

where:
- \(y\) = dependent variable (target)
- \(x\) = independent variable (feature)
- \(\beta_0\) = intercept
- \(\beta_1\) = slope (coefficient)
- \(\epsilon\) = error term

**Objective**: Minimize sum of squared residuals

\[
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2
\]

### Closed-Form Solution

\[
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
\]

\[
\beta_0 = \bar{y} - \beta_1 \bar{x}
\]

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope
    
    def fit(self, X, y):
        """
        Fit simple linear regression
        X: 1D array of features
        y: 1D array of targets
        """
        n = len(X)
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (beta_1)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.beta_1 = numerator / denominator
        
        # Calculate intercept (beta_0)
        self.beta_0 = y_mean - self.beta_1 * x_mean
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.beta_0 + self.beta_1 * X
    
    def score(self, X, y):
        """
        Calculate R-squared
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example: Predicting house prices based on size
np.random.seed(42)

# Generate synthetic data
house_size = np.random.uniform(500, 3500, 100)  # Square feet
price = 50000 + 100 * house_size + np.random.normal(0, 50000, 100)  # Price

# Train model
model = SimpleLinearRegression()
model.fit(house_size, price)

print(f"Intercept (β₀): ${model.beta_0:,.2f}")
print(f"Slope (β₁): ${model.beta_1:.2f} per square foot")
print(f"R-squared: {model.score(house_size, price):.3f}")

# Predictions
test_sizes = np.array([1000, 2000, 3000])
predictions = model.predict(test_sizes)

print("\nPredictions:")
for size, pred in zip(test_sizes, predictions):
    print(f"  {size:,} sq ft → ${pred:,.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(house_size, price, alpha=0.5, label='Actual data')
plt.plot(house_size, model.predict(house_size), 'r-', linewidth=2, label='Regression line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Simple Linear Regression: House Price vs Size')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Using Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Reshape for sklearn (needs 2D array)
X = house_size.reshape(-1, 1)
y = price

# Train model
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

print(f"Intercept: ${model_sklearn.intercept_:,.2f}")
print(f"Coefficient: ${model_sklearn.coef_[0]:.2f} per sq ft")

# Predictions
y_pred = model_sklearn.predict(X)

# Metrics
print(f"\nR-squared: {r2_score(y, y_pred):.3f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y, y_pred)):,.2f}")
print(f"MAE: ${np.mean(np.abs(y - y_pred)):,.2f}")
```

## Multiple Linear Regression

### Mathematical Foundation

**Model**:
\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon
\]

**Matrix Form**:
\[
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
\]

**Normal Equation**:
\[
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\]

### Implementation from Scratch

```python
import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Fit multiple linear regression using Normal Equation
        """
        # Add intercept column (column of ones)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Normal equation: β = (X^T X)^(-1) X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        
        # Solve for beta
        beta = np.linalg.solve(XtX, Xty)
        
        # Separate intercept and coefficients
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.intercept + X @ self.coefficients
    
    def score(self, X, y):
        """
        Calculate R-squared
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example: Predicting house prices with multiple features
np.random.seed(42)
n_samples = 200

# Features: size, bedrooms, age
size = np.random.uniform(500, 3500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

# Target: price
price = (50000 + 100*size + 20000*bedrooms - 1000*age + 
         np.random.normal(0, 30000, n_samples))

X = np.column_stack([size, bedrooms, age])
y = price

# Train model
model = MultipleLinearRegression()
model.fit(X, y)

print("Multiple Linear Regression Results:")
print(f"Intercept: ${model.intercept:,.2f}")
print(f"\nCoefficients:")
print(f"  Size: ${model.coefficients[0]:.2f} per sq ft")
print(f"  Bedrooms: ${model.coefficients[1]:,.2f} per bedroom")
print(f"  Age: ${model.coefficients[2]:,.2f} per year")
print(f"\nR-squared: {model.score(X, y):.3f}")

# Interpretation
print("\nInterpretation:")
print(f"- Each additional sq ft adds ${model.coefficients[0]:.2f} to price")
print(f"- Each additional bedroom adds ${model.coefficients[1]:,.2f} to price")
print(f"- Each year of age reduces price by ${abs(model.coefficients[2]):,.2f}")
```

## Complete Example: Boston Housing Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nFeatures: {list(data.feature_names)}")
print(f"\nBasic statistics:\n{df.describe()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Correlation analysis
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nCorrelation with Price:")
print(corr_matrix['PRICE'].sort_values(ascending=False))

# Prepare data
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluation
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"\nTraining Set:")
print(f"  R-squared: {r2_score(y_train, y_train_pred):.3f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")
print(f"  MAE: ${mean_absolute_error(y_train, y_train_pred):.3f}")

print(f"\nTest Set:")
print(f"  R-squared: {r2_score(y_test, y_test_pred):.3f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")
print(f"  MAE: ${mean_absolute_error(y_test, y_test_pred):.3f}")

# Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*60)
print(coef_df)

# Visualize coefficients
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients (Standardized)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

# Residual analysis
residuals = y_test - y_test_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title('Actual vs Predicted')
axes[0, 0].grid(alpha=0.3)

# 2. Residuals vs Predicted
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Predicted (Should be random)')
axes[0, 1].grid(alpha=0.3)

# 3. Residual distribution
axes[1, 0].hist(residuals, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution (Should be normal)')
axes[1, 0].grid(alpha=0.3)

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Should be linear)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=300)
plt.show()
```

## Model Assumptions

Linear regression makes several key assumptions:

### 1. Linearity

**Assumption**: Relationship between X and y is linear

```python
# Check linearity with scatter plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, col in enumerate(X.columns):
    ax = axes[idx // 4, idx % 4]
    ax.scatter(X[col], y, alpha=0.3)
    ax.set_xlabel(col)
    ax.set_ylabel('Price')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. Independence

**Assumption**: Observations are independent

**Check**: Durbin-Watson test for autocorrelation

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw:.3f}")
print("Interpretation: 2.0 = no autocorrelation, <2 = positive, >2 = negative")
```

### 3. Homoscedasticity

**Assumption**: Constant variance of residuals

**Check**: Plot residuals vs predicted values (should show random scatter)

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity Check')
plt.grid(alpha=0.3)
plt.show()

print("Look for:")
print("✓ Random scatter around zero (good)")
print("✗ Funnel shape (heteroscedasticity - bad)")
```

### 4. Normality of Residuals

**Assumption**: Residuals are normally distributed

```python
from scipy import stats

# Shapiro-Wilk test
statistic, p_value = stats.shapiro(residuals[:5000])  # Sample for large datasets

print(f"Shapiro-Wilk test: p-value = {p_value:.4f}")
if p_value > 0.05:
    print("✓ Residuals are normally distributed (p > 0.05)")
else:
    print("✗ Residuals are NOT normally distributed (p < 0.05)")

# Visual check
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(residuals, bins=50, edgecolor='black', density=True)
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Density')
ax1.set_title('Residual Distribution')
ax1.grid(alpha=0.3)

# Add normal curve
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal curve')
ax1.legend()

stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Multicollinearity

**Problem**: High correlation between independent variables

### Detection

#### 1. Correlation Matrix

```python
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

print("High correlations (|r| > 0.7):")
corr_matrix = X.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
```

#### 2. Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\nVariance Inflation Factors:")
print(vif_data)

print("\nInterpretation:")
print("VIF < 5: Low multicollinearity ✓")
print("5 < VIF < 10: Moderate multicollinearity ⚠️")
print("VIF > 10: High multicollinearity (problematic) ✗")

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(vif_data['Feature'], vif_data['VIF'])
plt.xlabel('VIF')
plt.title('Variance Inflation Factors')
plt.axvline(x=5, color='orange', linestyle='--', label='Moderate threshold')
plt.axvline(x=10, color='red', linestyle='--', label='High threshold')
plt.legend()
plt.tight_layout()
plt.show()
```

### Solutions

1. **Remove highly correlated features**
2. **Use regularization** (Ridge, Lasso)
3. **Principal Component Analysis (PCA)**
4. **Domain knowledge** to select features

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Linear regression models linear relationships** between features and target
2. **Normal equation**: \(\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}\)
3. **R-squared**: Proportion of variance explained (0-1, higher is better)
4. **Assumptions**: Linearity, independence, homoscedasticity, normality
5. **Residual analysis**: Check model assumptions visually
6. **Multicollinearity**: Check with VIF, address if VIF > 10
7. **Interpretation**: Coefficients show feature impact on target
8. **Scaling**: Standardize features for fair coefficient comparison
9. **RMSE and MAE**: Error metrics in original units
10. **Simple but powerful**: Often good baseline model
:::

## Further Reading

- James, G. et al. (2013). "An Introduction to Statistical Learning", Chapter 3
- Montgomery, D. et al. (2012). "Introduction to Linear Regression Analysis"
- Scikit-learn Linear Models: [sklearn.linear_model](https://scikit-learn.org/stable/modules/linear_model.html)
