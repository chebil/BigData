# Logistic Regression

## Introduction

Logistic regression is a statistical method for binary classification that models the probability of an instance belonging to a particular class. Despite its name containing "regression," it is primarily used for classification tasks. The method extends linear regression to handle categorical outcomes by using the logistic (sigmoid) function to constrain outputs between 0 and 1.

## Mathematical Foundation

### The Logistic Function

The logistic (sigmoid) function transforms any real-valued number into a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
\]

Where \(z\) is the linear combination of features:

\[
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n = \beta^T x
\]

### Properties of Sigmoid Function

1. **Range**: Output is always between 0 and 1
2. **Monotonic**: Always increasing
3. **Symmetric**: \(\sigma(-z) = 1 - \sigma(z)\)
4. **Interpretable**: Can be interpreted as probability
5. **Smooth**: Differentiable everywhere (useful for optimization)

### Odds and Odds Ratio

**Odds**: The ratio of the probability of success to the probability of failure.

\[
\text{Odds} = \frac{P(Y=1)}{P(Y=0)} = \frac{P(Y=1)}{1 - P(Y=1)}
\]

For example, if \(P(Y=1) = 0.8\), then:

\[
\text{Odds} = \frac{0.8}{0.2} = 4
\]

This means the event is 4 times more likely to occur than not occur.

**Log-Odds (Logit)**:

\[
\text{logit}(P) = \log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n
\]

The log-odds transformation:
- Converts probabilities (0, 1) to real numbers (-∞, ∞)
- Creates linear relationship with predictors
- Makes model interpretation straightforward

### Interpreting Coefficients

For a one-unit increase in \(x_j\):

\[
\text{Odds Ratio} = e^{\beta_j}
\]

**Example**: If \(\beta_1 = 0.5\), then \(e^{0.5} \approx 1.65\)
- One-unit increase in \(x_1\) multiplies odds by 1.65
- Equivalently: 65% increase in odds

If \(\beta_1 = -0.5\), then \(e^{-0.5} \approx 0.61\)
- One-unit increase in \(x_1\) multiplies odds by 0.61
- Equivalently: 39% decrease in odds

## Parameter Estimation

### Maximum Likelihood Estimation (MLE)

Unlike linear regression (which uses least squares), logistic regression uses maximum likelihood estimation.

**Likelihood Function**:

For binary outcomes, the likelihood of observing the data is:

\[
L(\beta) = \prod_{i=1}^{n} P(y_i | x_i, \beta) = \prod_{i=1}^{n} \sigma(\beta^T x_i)^{y_i} (1 - \sigma(\beta^T x_i))^{1-y_i}
\]

**Log-Likelihood**:

For numerical stability and easier optimization:

\[
\ell(\beta) = \sum_{i=1}^{n} [y_i \log(\sigma(\beta^T x_i)) + (1-y_i) \log(1 - \sigma(\beta^T x_i))]
\]

**Optimization**: Use iterative algorithms:
- Gradient descent
- Newton-Raphson method
- L-BFGS (Limited-memory BFGS)

### Gradient Descent for Logistic Regression

**Update Rule**:

\[
\beta := \beta - \alpha \nabla_{\beta} J(\beta)
\]

Where the gradient is:

\[
\frac{\partial J}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (\sigma(\beta^T x_i) - y_i) x_{ij}
\]

## Regularization

Regularization prevents overfitting by penalizing large coefficient values.

### L2 Regularization (Ridge)

**Objective Function**:

\[
J(\beta) = -\ell(\beta) + \lambda \sum_{j=1}^{n} \beta_j^2
\]

- Shrinks coefficients toward zero
- Doesn't set coefficients to exactly zero
- Good when all features are potentially relevant

### L1 Regularization (Lasso)

**Objective Function**:

\[
J(\beta) = -\ell(\beta) + \lambda \sum_{j=1}^{n} |\beta_j|
\]

- Can set coefficients to exactly zero
- Performs feature selection
- Produces sparse models

### Elastic Net

Combines L1 and L2 regularization:

\[
J(\beta) = -\ell(\beta) + \lambda_1 \sum_{j=1}^{n} |\beta_j| + \lambda_2 \sum_{j=1}^{n} \beta_j^2
\]

### Choosing Regularization Parameter

- Use cross-validation to select \(\lambda\)
- Larger \(\lambda\): More regularization, simpler model
- Smaller \(\lambda\): Less regularization, more complex model

## Making Predictions

### Probability Prediction

\[
P(Y=1|x) = \sigma(\beta^T x) = \frac{1}{1 + e^{-(\beta^T x)}}
\]

### Class Prediction

Using threshold \(t\) (typically 0.5):

\[
\hat{y} = \begin{cases}
1 & \text{if } P(Y=1|x) \geq t \\
0 & \text{if } P(Y=1|x) < t
\end{cases}
\]

### Threshold Selection

**Cost-Sensitive Classification**:

Choose threshold based on business costs:

\[
t^* = \arg\min_t \left[C_{FP} \cdot FP(t) + C_{FN} \cdot FN(t)\right]
\]

Where:
- \(C_{FP}\): Cost of false positive
- \(C_{FN}\): Cost of false negative

## Multiclass Logistic Regression

### One-vs-Rest (OvR)

For \(K\) classes:
1. Train \(K\) binary classifiers
2. Each classifier: one class vs. all others
3. Predict class with highest probability

### Multinomial (Softmax) Regression

**Softmax Function**:

\[
P(Y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}
\]

Properties:
- Probabilities sum to 1
- Generalizes logistic regression
- Trains single model for all classes

## Model Evaluation

### Deviance

Measures model fit (analogous to residual sum of squares):

\[
D = -2 \ell(\beta)
\]

**Null Deviance**: Deviance of intercept-only model
**Residual Deviance**: Deviance of fitted model

**Pseudo R²**:

\[
R^2_{\text{McFadden}} = 1 - \frac{D_{\text{model}}}{D_{\text{null}}}
\]

### Likelihood Ratio Test

Compare nested models:

\[
LR = -2(\ell_{\text{reduced}} - \ell_{\text{full}}) \sim \chi^2_{\text{df}}
\]

Where df = difference in number of parameters.

### ROC Curve

**Receiver Operating Characteristic** curve plots:
- TPR (True Positive Rate) vs. FPR (False Positive Rate)
- Shows performance across all thresholds

**Area Under Curve (AUC)**:
- Range: [0.5, 1.0]
- Interpretation: Probability that model ranks random positive higher than random negative
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.5-0.7: Poor

### Calibration

Calibrated model: Predicted probabilities match true frequencies.

**Calibration Plot**: 
- X-axis: Predicted probability bins
- Y-axis: Observed frequency
- Well-calibrated: Points lie on diagonal

## Assumptions

### Key Assumptions

1. **Binary outcome**: Target variable is binary
2. **Independence**: Observations are independent
3. **Linearity**: Log-odds are linear in parameters
4. **No multicollinearity**: Predictors are not highly correlated
5. **Large sample size**: MLE requires adequate sample size

### Diagnostic Checks

**Multicollinearity**:
- Calculate Variance Inflation Factor (VIF)
- VIF > 10 indicates problem
- Solution: Remove or combine correlated variables

**Influential Points**:
- Cook's distance
- Leverage values
- Standardized residuals

**Model Fit**:
- Hosmer-Lemeshow test
- Deviance residuals
- Calibration plots

## Implementation in Python

### Basic Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
print(f'AUC: {roc_auc_score(y_test, y_prob):.3f}')
```

### With Regularization

```python
# L2 regularization (default)
model_l2 = LogisticRegression(penalty='l2', C=1.0, random_state=42)

# L1 regularization
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)

# Elastic Net
model_elastic = LogisticRegression(penalty='elasticnet', solver='saga',
                                    l1_ratio=0.5, C=1.0, random_state=42)
```

**Note**: Smaller `C` = more regularization (inverse of λ)

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best AUC: {grid_search.best_score_:.3f}')
```

## Advantages and Limitations

### Advantages

✅ **Probabilistic interpretation**: Outputs probabilities
✅ **Efficient**: Fast training and prediction
✅ **Interpretable**: Coefficients have clear meaning
✅ **No hyperparameters**: (without regularization)
✅ **Works well**: For linearly separable classes
✅ **Extensible**: Handles multi-class easily
✅ **Less prone to overfitting**: With regularization

### Limitations

❌ **Linear decision boundary**: Can't capture complex relationships
❌ **Assumes independence**: Among predictors
❌ **Sensitive to outliers**: In predictor space
❌ **Requires feature engineering**: For non-linear relationships
❌ **Multicollinearity issues**: Affects coefficient stability
❌ **Large sample size needed**: For reliable estimates

## Best Practices

1. **Feature Scaling**: Always scale features to same range
2. **Handle Missing Values**: Impute before modeling
3. **Encode Categorical Variables**: Use one-hot or target encoding
4. **Check Multicollinearity**: Remove highly correlated features
5. **Use Regularization**: Especially with many features
6. **Cross-Validation**: For hyperparameter tuning
7. **Stratified Sampling**: Preserve class distribution
8. **Feature Selection**: Use L1 or domain knowledge
9. **Threshold Tuning**: Based on business requirements
10. **Monitor Calibration**: Ensure probabilities are meaningful

## Real-World Applications

### Medical Diagnosis
- Disease presence/absence based on symptoms
- Risk prediction (heart disease, diabetes)
- Treatment response prediction

### Finance
- Credit default prediction
- Fraud detection
- Customer churn

### Marketing
- Customer conversion prediction
- Email click-through
- Ad response modeling

### Technology
- Spam detection
- Recommendation systems
- User behavior prediction

## Example: Customer Churn Prediction

### Problem Statement
Predict which customers will churn based on:
- Account tenure
- Monthly charges
- Total charges
- Contract type
- Payment method
- Customer service calls

### Solution Approach

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('telecom_churn.csv')

# Feature engineering
df['tenure_years'] = df['tenure'] / 12
df['avg_monthly_charges'] = df['TotalCharges'] / df['tenure']

# Encode categorical variables
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'], drop_first=True)

# Prepare features
features = ['tenure_years', 'MonthlyCharges', 'TotalCharges', 
            'CustomerServiceCalls', 'Contract_One year', 
            'Contract_Two year', 'PaymentMethod_Credit card',
            'PaymentMethod_Electronic check']

X = df[features]
y = df['Churn']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with regularization
model = LogisticRegression(penalty='l2', C=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Select threshold based on business requirement
# Cost of losing customer: $500
# Cost of retention offer: $50
# Optimal threshold balances these costs
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

print(classification_report(y_test, y_pred))

# Feature importance
coefs = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
}).sort_values('coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(coefs)
```

### Interpretation

**Most important predictors** (hypothetical results):
1. **Contract_Two year**: OR = 0.15 (85% reduction in churn odds)
2. **tenure_years**: OR = 0.80 (20% reduction per year)
3. **CustomerServiceCalls**: OR = 1.35 (35% increase per call)

**Business Actions**:
- Encourage customers to sign 2-year contracts
- Improve customer service to reduce calls
- Target retention efforts at customers with <1 year tenure

## Summary

Logistic regression is a fundamental classification algorithm that:
- Models probability of binary outcomes
- Uses logistic function to transform linear combination
- Estimates parameters via maximum likelihood
- Provides interpretable coefficients as odds ratios
- Extends to multiclass via softmax
- Benefits from regularization for complex problems
- Serves as baseline for more complex models

## Further Reading

- "An Introduction to Statistical Learning" - Chapter 4
- "The Elements of Statistical Learning" - Chapter 4
- "Applied Logistic Regression" by Hosmer, Lemeshow, and Sturdivant

## Next Section

Continue to [Naïve Bayes Classifier](02-naive-bayes.md) to learn about probabilistic classification using Bayes' theorem.