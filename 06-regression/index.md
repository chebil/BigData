# Chapter 6: Regression Analysis

## Overview

Regression analysis is one of the most fundamental and widely used techniques in data science. This chapter covers linear regression for predicting continuous outcomes, logistic regression for binary classification, model evaluation techniques, and regularization methods. You'll learn both the mathematical foundations and practical implementation using Python and scikit-learn.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Build and interpret** linear regression models
2. **Assess assumptions** of linear regression (linearity, normality, homoscedasticity, independence)
3. **Evaluate model performance** using R², RMSE, MAE, and residual analysis
4. **Handle multiple predictors** and understand multicollinearity
5. **Apply transformations** to meet regression assumptions
6. **Implement logistic regression** for binary classification
7. **Use regularization** (Ridge, Lasso, Elastic Net) to prevent overfitting
8. **Select appropriate regression models** for different problem types

## Topics Covered

### 1. Linear Regression
- Simple linear regression
- Multiple linear regression
- Ordinary least squares (OLS)
- Assumptions and diagnostics
- R² and adjusted R²
- Residual analysis

### 2. Multiple Regression
- Multiple predictors
- Interaction terms
- Polynomial regression
- Multicollinearity (VIF)
- Feature selection

### 3. Model Evaluation
- Training and test sets
- Cross-validation
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² and adjusted R²

### 4. Regularization
- Ridge regression (L2)
- Lasso regression (L1)
- Elastic Net
- Hyperparameter tuning
- Feature selection with Lasso

## Chapter Sections

```{tableofcontents}
```

## Linear Regression

### Simple Linear Regression

Modeling relationship between one predictor and one outcome:

\[
y = \beta_0 + \beta_1 x + \epsilon
\]

Where:
- \(y\): Dependent variable (what we predict)
- \(x\): Independent variable (predictor)
- \(\beta_0\): Intercept
- \(\beta_1\): Slope (coefficient)
- \(\epsilon\): Error term

### Ordinary Least Squares (OLS)

**Objective**: Minimize sum of squared residuals

\[
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2
\]

**Closed-form solution**:

\[
\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\]

\[
\beta_0 = \bar{y} - \beta_1 \bar{x}
\]

### Multiple Linear Regression

Multiple predictors:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon
\]

**Matrix form**:

\[
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
\]

**OLS solution**:

\[
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\]

## Regression Assumptions

### 1. Linearity

Relationship between X and Y is linear.

**Check**: Scatter plots, residual plots
**Fix**: Transform variables (log, sqrt, polynomial)

### 2. Independence

Observations are independent of each other.

**Check**: Domain knowledge, Durbin-Watson test
**Fix**: Use appropriate models (time series, hierarchical)

### 3. Homoscedasticity

Constant variance of residuals across all levels of X.

**Check**: Residual plot (should show random scatter)
**Fix**: Transform Y (log, sqrt), use weighted regression

### 4. Normality

Residuals are normally distributed.

**Check**: Q-Q plot, histogram of residuals, Shapiro-Wilk test
**Fix**: Transform Y, use robust regression

### 5. No Perfect Multicollinearity

Predictors are not perfectly correlated with each other.

**Check**: Variance Inflation Factor (VIF)
**Fix**: Remove redundant features, use regularization

## Model Evaluation Metrics

### R² (Coefficient of Determination)

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- Range: 0 to 1 (higher is better)
- Proportion of variance explained
- **Problem**: Always increases with more predictors

### Adjusted R²

\[
R^2_{adj} = 1 - \frac{(1 - R^2)(n-1)}{n - p - 1}
\]

- Penalizes for number of predictors
- Can decrease if adding irrelevant predictors
- Better for comparing models

### Root Mean Squared Error (RMSE)

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

- Same units as Y
- Sensitive to outliers
- Lower is better

### Mean Absolute Error (MAE)

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

- Same units as Y
- Less sensitive to outliers than RMSE
- Lower is better

## Logistic Regression

### For Binary Classification

Predicting probability of binary outcome:

\[
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p)}}
\]

### Logit (Log-Odds)

\[
\log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
\]

### Interpretation

- **Coefficients**: Change in log-odds for unit increase in X
- **Odds Ratio**: \(e^{\beta}\) = multiplicative effect on odds
- **Probability**: Use sigmoid function

### Evaluation Metrics

- **Accuracy**: Proportion correctly classified
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: 2x2 table of predictions vs. actuals

## Regularization

### Why Regularize?

- Prevent overfitting
- Handle multicollinearity
- Feature selection (Lasso)
- Better generalization to new data

### Ridge Regression (L2)

**Objective**: Minimize RSS + penalty on coefficient size

\[
\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}
\]

**Effect**: Shrinks coefficients toward zero (but not exactly to zero)

### Lasso Regression (L1)

\[
\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
\]

**Effect**: Can shrink coefficients exactly to zero (automatic feature selection)

### Elastic Net

Combines L1 and L2 penalties:

\[
\min_{\boldsymbol{\beta}} \left\{ \text{RSS} + \lambda \left( \alpha \sum_{j=1}^{p} |\beta_j| + (1-\alpha) \sum_{j=1}^{p} \beta_j^2 \right) \right\}
\]

**Parameters**:
- \(\lambda\): Overall regularization strength
- \(\alpha\): Mix between L1 (α=1) and L2 (α=0)

### Choosing λ (Hyperparameter Tuning)

- **Cross-validation**: Split data, try different λ, select best
- **Grid search**: Try predefined λ values
- **Random search**: Randomly sample λ values
- **Use scikit-learn**: `RidgeCV`, `LassoCV`, `ElasticNetCV`

## Hands-On Practice

### Associated Lab
- **[Lab 6: Regression](../labs/lab-06-regression/README.md)** - Build and evaluate regression models

### Jupyter Notebooks
1. [Simple Linear Regression](notebooks/01-simple-linear-regression.ipynb) - One predictor
2. [Multiple Regression](notebooks/02-multiple-regression.ipynb) - Multiple predictors
3. [Polynomial Regression](notebooks/03-polynomial-regression.ipynb) - Non-linear relationships
4. [Regularization Techniques](notebooks/04-regularization-techniques.ipynb) - Ridge, Lasso, Elastic Net

## Use Cases

### Linear Regression
- **Sales forecasting**: Predict sales from advertising spend
- **Real estate**: House price prediction
- **Finance**: Stock price modeling
- **Healthcare**: Medical cost prediction

### Logistic Regression
- **Credit scoring**: Loan default prediction
- **Medical diagnosis**: Disease presence/absence
- **Marketing**: Customer churn prediction
- **Quality control**: Defect detection

## Practical Considerations

### Feature Engineering

- **Polynomial terms**: \(x^2, x^3\) for non-linear relationships
- **Interaction terms**: \(x_1 \times x_2\) for combined effects
- **Log transforms**: Handle skewed distributions
- **Standardization**: Scale features (especially important for regularization)

### Model Selection

**Simple Linear Regression**: 
- One predictor
- Clear interpretation needed

**Multiple Regression**:
- Multiple predictors
- Linear relationships
- Need coefficient interpretability

**Polynomial Regression**:
- Non-linear relationships
- Still interpretable

**Regularized Regression**:
- Many predictors
- Multicollinearity present
- Feature selection needed (Lasso)

## Common Pitfalls

- ❌ Not checking regression assumptions
- ❌ Interpreting correlation as causation
- ❌ Using R² alone to evaluate models
- ❌ Not using train/test split
- ❌ Extrapolating beyond data range
- ❌ Not scaling features before regularization
- ❌ Ignoring outliers' influence

## Additional Resources

### Required Reading
- Textbook Chapter 6: "Advanced Analytical Theory and Methods: Regression"
- EMC Education Services, pp. 161-190

### Recommended Reading
- "Introduction to Statistical Learning" Chapters 3 and 6
- "Applied Predictive Modeling" by Kuhn and Johnson

### Videos
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [StatQuest: Regularization](https://www.youtube.com/watch?v=Q81RR3yKn30)

### Online Resources
- [Scikit-learn Regression Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## Summary

Regression is fundamental to predictive modeling:

- **Linear Regression**: Predicts continuous outcomes
- **Logistic Regression**: Predicts binary outcomes (classification)
- **Regularization**: Prevents overfitting, handles multicollinearity

Key practices:
- Always check assumptions
- Use train/test splits
- Validate with proper metrics
- Consider regularization for complex models
- Interpret coefficients in context

## Next Steps

1. Work through all four Jupyter notebooks
2. Complete [Lab 6: Regression](../labs/lab-06-regression/README.md)
3. Practice on different datasets
4. Prepare for Quiz 2 (Chapters 4-6) in Week 7
5. Move on to [Chapter 7: Classification Methods](../07-classification/index.md)

---

**Remember**: A model that fits training data perfectly may perform poorly on new data (overfitting)!
