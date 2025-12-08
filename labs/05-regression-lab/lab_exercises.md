# Lab 5: Regression Exercises

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

## Part 1: Data Exploration (20 points)

### Exercise 1.1: Load Data (5 points)

```python
# Load California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')

df = pd.concat([X, y], axis=1)

print(f"Dataset shape: {df.shape}")
print("\nFeatures:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
```

**Questions:**
- Q1: How many features are there?
- Q2: What is the price range?
- Q3: Are there missing values?

### Exercise 1.2: Feature Analysis (10 points)

```python
# Correlation analysis
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature correlations with target
target_corr = corr_matrix['MedHouseValue'].sort_values(ascending=False)
print("\nFeature correlations with price:")
print(target_corr)

# Distribution plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(df.columns[:-1]):
    ax = axes[i//3, i%3]
    df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q4: Which feature correlates most with price?
- Q5: Do you observe multicollinearity?
- Q6: Are any features skewed?

### Exercise 1.3: Feature Engineering (5 points)

```python
# TODO: Create new features
# 1. Rooms per household
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']

# 2. Bedrooms ratio
df['BedroomsRatio'] = df['AveBedrms'] / df['AveRooms']

# 3. People per household
df['PeoplePerHousehold'] = df['Population'] / df['HouseAge']

print("\nNew features created:")
print(df[['RoomsPerHousehold', 'BedroomsRatio', 'PeoplePerHousehold']].head())

# Check correlations
new_corr = df[['RoomsPerHousehold', 'BedroomsRatio', 'PeoplePerHousehold', 'MedHouseValue']].corr()
print("\nNew feature correlations:")
print(new_corr['MedHouseValue'])
```

**Questions:**
- Q7: Do the new features improve correlation with price?
- Q8: What other features could you engineer?

---

## Part 2: Linear Regression (20 points)

### Exercise 2.1: Train/Test Split (5 points)

```python
# Prepare features and target
feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                'AveOccup', 'Latitude', 'Longitude']
X = df[feature_cols]
y = df['MedHouseValue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Exercise 2.2: Fit Linear Regression (10 points)

```python
# Train model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = lr.predict(X_train_scaled)
y_pred_test = lr.predict(X_test_scaled)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Linear Regression Results:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Feature Importance (Linear Regression)')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q9: What is the RMSE?
- Q10: Is there overfitting?
- Q11: Which features are most important?

### Exercise 2.3: Residual Analysis (5 points)

```python
# Calculate residuals
residuals = y_test - y_pred_test

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual plot
axes[0, 0].scatter(y_pred_test, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residual Plot')

# Histogram of residuals
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Actual vs Predicted
axes[1, 1].scatter(y_test, y_pred_test, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title('Actual vs Predicted')

plt.tight_layout()
plt.show()
```

**Questions:**
- Q12: Are residuals normally distributed?
- Q13: Is there heteroscedasticity?
- Q14: Are there outliers?

---

## Part 3: Regularization (25 points)

### Exercise 3.1: Ridge Regression (10 points)

```python
# TODO: Try different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred = ridge.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    ridge_results.append({
        'alpha': alpha,
        'rmse': rmse,
        'r2': r2
    })

ridge_df = pd.DataFrame(ridge_results)
print("\nRidge Regression Results:")
print(ridge_df)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ridge_df['alpha'], ridge_df['rmse'], 'o-')
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Ridge: Alpha vs RMSE')
axes[0].grid(alpha=0.3)

axes[1].plot(ridge_df['alpha'], ridge_df['r2'], 'o-')
axes[1].set_xscale('log')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('R²')
axes[1].set_title('Ridge: Alpha vs R²')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Questions:**
- Q15: What is the optimal alpha?
- Q16: How does Ridge compare to Linear Regression?

### Exercise 3.2: Lasso Regression (10 points)

```python
# TODO: Try Lasso
lasso_results = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    y_pred = lasso.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n_features = np.sum(lasso.coef_ != 0)
    
    lasso_results.append({
        'alpha': alpha,
        'rmse': rmse,
        'r2': r2,
        'n_features': n_features
    })

lasso_df = pd.DataFrame(lasso_results)
print("\nLasso Regression Results:")
print(lasso_df)

# Feature selection
best_alpha = lasso_df.loc[lasso_df['rmse'].idxmin(), 'alpha']
best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train_scaled, y_train)

feature_coef = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': best_lasso.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nBest Lasso (alpha={best_alpha}):")
print(feature_coef)
```

**Questions:**
- Q17: Which features did Lasso eliminate?
- Q18: How does feature selection help?
- Q19: Ridge vs Lasso - which is better?

### Exercise 3.3: Elastic Net (5 points)

```python
# TODO: Try Elastic Net
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

enet = ElasticNet(max_iter=10000)
grid = GridSearchCV(enet, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV score (RMSE): {np.sqrt(-grid.best_score_):.4f}")

y_pred_enet = grid.predict(X_test_scaled)
test_rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_enet))
print(f"Test RMSE: {test_rmse_enet:.4f}")
```

**Questions:**
- Q20: What are optimal hyperparameters?
- Q21: Does Elastic Net outperform Ridge and Lasso?

---

## Part 4: Polynomial Regression (20 points)

### Exercise 4.1: Polynomial Features (15 points)

```python
# TODO: Try polynomial degrees
degrees = [1, 2, 3, 4]
poly_results = []

for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Train Ridge (to avoid overfitting)
    ridge_poly = Ridge(alpha=1.0)
    ridge_poly.fit(X_train_poly, y_train)
    
    # Evaluate
    y_pred_train = ridge_poly.predict(X_train_poly)
    y_pred_test = ridge_poly.predict(X_test_poly)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    poly_results.append({
        'degree': degree,
        'n_features': X_train_poly.shape[1],
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    })

poly_df = pd.DataFrame(poly_results)
print("\nPolynomial Regression Results:")
print(poly_df)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(poly_df['degree'], poly_df['train_rmse'], 'o-', label='Train')
axes[0].plot(poly_df['degree'], poly_df['test_rmse'], 's-', label='Test')
axes[0].set_xlabel('Polynomial Degree')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Polynomial Degree vs RMSE')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(poly_df['degree'], poly_df['train_r2'], 'o-', label='Train')
axes[1].plot(poly_df['degree'], poly_df['test_r2'], 's-', label='Test')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('R²')
axes[1].set_title('Polynomial Degree vs R²')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Questions:**
- Q22: What degree is optimal?
- Q23: Do you observe overfitting?
- Q24: How many features does degree 3 create?

### Exercise 4.2: Bias-Variance Tradeoff (5 points)

```python
# Visualize bias-variance tradeoff
print("\nBias-Variance Analysis:")
print(poly_df[['degree', 'train_rmse', 'test_rmse']])

gap = poly_df['test_rmse'] - poly_df['train_rmse']
print("\nTrain-Test Gap (indication of overfitting):")
for i, row in poly_df.iterrows():
    print(f"Degree {row['degree']}: {gap[i]:.4f}")
```

**Questions:**
- Q25: At what degree does overfitting begin?

---

## Part 5: Final Model (15 points)

### Exercise 5.1: Model Comparison (10 points)

```python
# Compare all models
comparison = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso', 'Elastic Net', 'Poly (deg=2)'],
    'RMSE': [
        test_rmse,
        ridge_df.loc[ridge_df['rmse'].idxmin(), 'rmse'],
        lasso_df.loc[lasso_df['rmse'].idxmin(), 'rmse'],
        test_rmse_enet,
        poly_df[poly_df['degree']==2]['test_rmse'].values[0]
    ]
})

comparison = comparison.sort_values('RMSE')
print("\nModel Comparison:")
print(comparison)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(comparison['Model'], comparison['RMSE'], alpha=0.7)
plt.xlabel('RMSE')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q26: Which model performs best?
- Q27: What is the best RMSE?
- Q28: Would you recommend this model for production?

### Exercise 5.2: Production Pipeline (5 points)

```python
import joblib

# Create production pipeline
from sklearn.pipeline import Pipeline

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))  # Use best model
])

# Train on all data
best_pipeline.fit(X, y)

# Save
joblib.dump(best_pipeline, 'house_price_model.pkl')
print("Model saved!")

# Test loading
loaded_model = joblib.load('house_price_model.pkl')

# Make prediction
sample = X.iloc[0:1]
prediction = loaded_model.predict(sample)
print(f"\nSample prediction:")
print(f"  Features: {sample.values[0]}")
print(f"  Predicted price: ${prediction[0]*100000:.2f}")
print(f"  Actual price: ${y.iloc[0]*100000:.2f}")
```

**Final Questions:**
- Q29: How would you deploy this model?
- Q30: What metrics would you monitor in production?

Good luck! 🏠
