# Chapter 5: Regression Analysis

## Introduction

Regression analysis predicts continuous numerical values rather than discrete categories. It's essential for forecasting sales, estimating prices, predicting demand, and understanding relationships between variables. This chapter covers linear regression, regularization techniques, and advanced regression methods.

## Learning Objectives

- Understand regression problem formulation
- Implement linear regression from scratch and with libraries
- Apply regularization (Ridge, Lasso) to prevent overfitting
- Evaluate regression models with appropriate metrics
- Handle multicollinearity and feature engineering
- Build polynomial and non-linear regression models
- Scale regression to Big Data with Spark MLlib

## Chapter Overview

1. **Regression Fundamentals** - Problem types, assumptions, least squares
2. **Linear Regression** - Simple and multiple regression
3. **Regularization** - Ridge, Lasso, Elastic Net
4. **Polynomial Regression** - Non-linear relationships
5. **Evaluation Metrics** - R², RMSE, MAE
6. **Practical Applications** - Price prediction, demand forecasting

## What is Regression?

### Definition

**Regression**: Predict continuous target variable from input features

\[
f: X \rightarrow \mathbb{R}
\]

where:
- \(X\) = feature space (inputs)
- \(\mathbb{R}\) = real numbers (continuous output)

### Simple vs. Multiple Regression

**Simple Linear Regression**: One predictor
\[
y = \beta_0 + \beta_1 x + \epsilon
\]

**Multiple Linear Regression**: Multiple predictors
\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
\]

where:
- \(y\) = target variable
- \(x_i\) = predictor variables
- \(\beta_i\) = coefficients
- \(\epsilon\) = error term

## Real-World Applications

### House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load housing data
df = pd.read_csv('data/housing.csv')

# Features: sqft, bedrooms, bathrooms, age, distance_to_city
# Target: price

X = df[['sqft', 'bedrooms', 'bathrooms', 'age', 'distance_to_city']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:,.2f}")

# Interpretation:
# "For each additional sqft, price increases by $X"
# "For each additional bedroom, price increases by $Y"

# Predictions
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# Predict new house
new_house = [[2000, 3, 2, 10, 5]]  # 2000 sqft, 3 bed, 2 bath, 10 years old, 5 miles
predicted_price = model.predict(new_house)[0]
print(f"\nPredicted price: ${predicted_price:,.2f}")
```

### Sales Forecasting

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Time series data
df = pd.read_csv('data/monthly_sales.csv')
df['month_num'] = range(len(df))  # Convert to numeric

X = df[['month_num']].values
y = df['sales'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_months = np.array([[len(df) + i] for i in range(6)])
forecast = model.predict(future_months)

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(df['month_num'], df['sales'], 'bo-', label='Historical Sales')
plt.plot(future_months, forecast, 'ro--', label='Forecast')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.title('Sales Forecast')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("6-Month Forecast:")
for i, sale in enumerate(forecast, 1):
    print(f"Month +{i}: ${sale:,.2f}")
```

### Salary Prediction

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Employee data
df = pd.read_csv('data/employees.csv')

# Features: years_experience, education_level, department
# Target: salary

# Encode categorical variables
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education_level'])
df['dept_encoded'] = le.fit_transform(df['department'])

X = df[['years_experience', 'education_encoded', 'dept_encoded']]
y = df['salary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Feature importance
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:,.2f}/unit")

# Predict for new employee
new_employee = [[5, 2, 1]]  # 5 years exp, Master's degree, Dept 1
predicted_salary = model.predict(new_employee)[0]
print(f"\nPredicted salary: ${predicted_salary:,.2f}")
```

## Linear Regression Mathematics

### Ordinary Least Squares (OLS)

**Goal**: Minimize sum of squared residuals

\[
\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_{\beta} \sum_{i=1}^{n} (y_i - \beta^T x_i)^2
\]

**Closed-form solution**:
\[
\beta = (X^T X)^{-1} X^T y
\]

### Implementation from Scratch

```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Add intercept term (column of 1s)
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        # Closed-form solution: β = (X^T X)^-1 X^T y
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_

# Test
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegressionScratch()
model.fit(X, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

y_pred = model.predict(X)
print(f"Predictions: {y_pred}")
```

### Gradient Descent Alternative

```python
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.intercept_ + X @ self.coef_
            
            # Calculate loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # Gradients
            d_intercept = -2 * np.mean(y - y_pred)
            d_coef = -2 * X.T @ (y - y_pred) / n_samples
            
            # Update parameters
            self.intercept_ -= self.lr * d_intercept
            self.coef_ -= self.lr * d_coef
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_

# Test
model_gd = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model_gd.fit(X, y)

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(model_gd.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent Convergence')
plt.grid(alpha=0.3)
plt.show()
```

## Evaluation Metrics

### R² (Coefficient of Determination)

\[
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]

**Interpretation**: Proportion of variance explained by model
- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- R² < 0: Model worse than mean

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
print(f"Model explains {r2*100:.1f}% of variance")
```

### RMSE (Root Mean Squared Error)

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

**Interpretation**: Average prediction error in original units

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
print(f"On average, predictions are off by ${rmse:,.2f}")
```

### MAE (Mean Absolute Error)

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

**Less sensitive to outliers** than RMSE

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
```

## Model Assumptions

### Checking Assumptions

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Linearity
plt.figure(figsize=(15, 4))

plt.subplot(131)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')

# 2. Residuals normally distributed
residuals = y_test - y_pred

plt.subplot(132)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

# 3. Homoscedasticity (constant variance)
plt.subplot(133)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

## Chapter Structure

1. **Linear Regression** - Simple and multiple regression
2. **Regularization** - Ridge, Lasso, Elastic Net
3. **Polynomial Regression** - Non-linear relationships
4. **Feature Engineering** - Transformations and interactions
5. **Evaluation Metrics** - Comprehensive assessment
6. **Practical Applications** - Real-world projects

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Regression predicts continuous values**
2. **Linear regression assumes linear relationship**
3. **Feature scaling helps gradient descent** (not needed for OLS)
4. **Check assumptions** with residual plots
5. **Multiple metrics** provide complete picture
6. **Regularization prevents overfitting**
7. **Feature engineering often more important** than algorithm choice
:::

## Next Steps

- **Section 5.1**: Linear Regression Deep Dive
- **Section 5.2**: Regularization Techniques
- **Section 5.3**: Polynomial Regression
- **Section 5.4**: Feature Engineering
- **Section 5.5**: Practical Applications

## Further Reading

- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 3
- James, G. et al. (2013). "An Introduction to Statistical Learning", Chapter 3
