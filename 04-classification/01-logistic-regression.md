# Logistic Regression

## Learning Objectives

- Understand logistic regression for binary and multiclass classification
- Implement logistic regression from scratch and with scikit-learn
- Interpret coefficients and probabilities
- Apply regularization to prevent overfitting
- Evaluate models with appropriate metrics

## Introduction

Despite its name, logistic regression is a **classification** algorithm, not regression. It predicts the probability that an instance belongs to a particular class, making it ideal for binary classification problems.

## Mathematical Foundation

### Sigmoid Function

**Transforms linear combination to probability** [0, 1]

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \(z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n\)

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot sigmoid function
z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary (0.5)')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
plt.xlabel('z (linear combination)')
plt.ylabel('σ(z) (probability)')
plt.title('Sigmoid (Logistic) Function')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

print("Sigmoid properties:")
print(f"σ(0) = {sigmoid(0)}")  # 0.5
print(f"σ(-∞) → {sigmoid(-100):.10f}")  # ~0
print(f"σ(+∞) → {sigmoid(100):.10f}")  # ~1
```

### Logistic Regression Model

**Probability of class 1**:
\[
P(y=1|x) = \sigma(\beta^T x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}
\]

**Probability of class 0**:
\[
P(y=0|x) = 1 - P(y=1|x)
\]

**Decision rule**:
\[
\hat{y} = \begin{cases}
1 & \text{if } P(y=1|x) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

### Log-Likelihood and Cost Function

**Likelihood** for single observation:
\[
L(\beta) = P(y|x) = P(y=1|x)^y \cdot P(y=0|x)^{1-y}
\]

**Log-Likelihood** (entire dataset):
\[
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]
\]

**Cost Function** (minimize):
\[
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]
\]

This is **binary cross-entropy** or **log loss**.

## Implementation from Scratch

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear model
            z = X @ self.weights + self.bias
            
            # Predictions
            y_pred = self.sigmoid(z)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.losses.append(loss)
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Test on synthetic data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train custom model
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nAccuracy: {accuracy:.2%}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (Binary Cross-Entropy)')
plt.title('Training Loss Over Time')
plt.grid(alpha=0.3)
plt.show()
```

## Using Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load data
df = pd.read_csv('data/heart_disease.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']  # 0 = no disease, 1 = disease

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for logistic regression!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    penalty='l2',        # Regularization type
    C=1.0,              # Inverse of regularization strength
    solver='lbfgs',     # Optimization algorithm
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Coefficients
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:20s}: {coef:8.4f}")
print(f"{'Intercept':20s}: {model.intercept_[0]:8.4f}")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Interpreting Coefficients

### Odds and Odds Ratio

**Odds**: Ratio of probability of event to probability of non-event
\[
\text{Odds} = \frac{P(y=1)}{P(y=0)} = \frac{P(y=1)}{1 - P(y=1)}
\]

**Logistic regression in terms of log-odds**:
\[
\log\left(\frac{P(y=1|x)}{P(y=0|x)}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n
\]

**Odds Ratio**: \(e^{\beta_i}\) is the multiplicative change in odds for one-unit increase in \(x_i\)

```python
import numpy as np
import pandas as pd

# Example: Predicting loan default
from sklearn.linear_model import LogisticRegression

# Features: income (thousands), credit_score, debt_ratio
X = pd.DataFrame({
    'income': [30, 50, 70, 40, 60, 80],
    'credit_score': [600, 700, 750, 650, 720, 780],
    'debt_ratio': [0.5, 0.3, 0.2, 0.4, 0.25, 0.15]
})
y = np.array([1, 0, 0, 1, 0, 0])  # 1=default, 0=no default

model = LogisticRegression()
model.fit(X, y)

# Interpret coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})

print(coef_df)
print("\nInterpretation:")
for _, row in coef_df.iterrows():
    if row['Odds Ratio'] > 1:
        print(f"{row['Feature']}: For each unit increase, odds of default multiply by {row['Odds Ratio']:.2f}")
    else:
        print(f"{row['Feature']}: For each unit increase, odds of default multiply by {row['Odds Ratio']:.2f} (decrease)")
```

## Regularization

### L2 Regularization (Ridge)

**Cost function**:
\[
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right] + \frac{\lambda}{2} \sum_{j=1}^{p} \beta_j^2
\]

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Try different regularization strengths
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_values:
    model = LogisticRegression(penalty='l2', C=C, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"C={C:6.3f}: Train={train_score:.3f}, Test={test_score:.3f}, "
          f"Coef norm={np.linalg.norm(model.coef_):.3f}")

# Note: C = 1/λ (inverse of regularization strength)
# Smaller C = stronger regularization
```

### L1 Regularization (Lasso)

**Feature selection** - drives some coefficients to exactly zero

```python
model_l1 = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=1000)
model_l1.fit(X_train_scaled, y_train)

print("L1 Regularization - Feature Selection:")
for feature, coef in zip(X.columns, model_l1.coef_[0]):
    if coef != 0:
        print(f"{feature:20s}: {coef:8.4f} (kept)")
    else:
        print(f"{feature:20s}: {coef:8.4f} (removed)")
```

## Multiclass Classification

### One-vs-Rest (OvR)

**Strategy**: Train K binary classifiers (one per class)

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiclass logistic regression (OvR by default)
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2%}")

# Probabilities for all classes
y_pred_proba = model.predict_proba(X_test)
print("\nProbabilities for first 5 samples:")
print(pd.DataFrame(y_pred_proba[:5], columns=iris.target_names))
```

### Softmax (Multinomial)

**Single model** that outputs probabilities for all classes

\[
P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}
\]

```python
# Softmax regression
model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_softmax.fit(X_train, y_train)

print(f"Softmax Accuracy: {model_softmax.score(X_test, y_test):.2%}")
```

## Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

# ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)

# Find optimal threshold (maximize F1-score)
from sklearn.metrics import f1_score

f1_scores = []
for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

optimal_threshold = np.arange(0.1, 0.9, 0.01)[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Max F1-score: {max(f1_scores):.3f}")

# Use optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

print("\nWith default threshold (0.5):")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")

print(f"\nWith optimal threshold ({optimal_threshold:.2f}):")
print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.3f}")
```

## Complete Example: Customer Churn Prediction

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

# Load data
df = pd.read_csv('data/telecom_churn.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['Churn'].value_counts())

# Feature engineering
# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']

for col in categorical_cols:
    df[f'{col}_encoded'] = le.fit_transform(df[col])

# Select features
feature_cols = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_encoded', 'Contract_encoded', 
    'PaymentMethod_encoded', 'InternetService_encoded'
]

X = df[feature_cols]
y = (df['Churn'] == 'Yes').astype(int)  # Convert to binary

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\nCV AUC-ROC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Comprehensive evaluation
print("\n=== Model Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.3f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Confusion Matrix')

# ROC Curve
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[0, 1])
axes[0, 1].set_title('ROC Curve')
axes[0, 1].grid(alpha=0.3)

# Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=axes[1, 0])
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].grid(alpha=0.3)

# Feature Importance
axes[1, 1].barh(feature_importance['Feature'], feature_importance['Coefficient'])
axes[1, 1].set_xlabel('Coefficient')
axes[1, 1].set_title('Feature Importance (Coefficients)')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('churn_prediction_analysis.png', dpi=300)
plt.show()

# Business insights
print("\n=== Business Recommendations ===")
print(f"Expected churn rate: {y_test.mean():.1%}")
print(f"Customers at high risk (prob > 0.7): {(y_pred_proba > 0.7).sum()}")
print(f"Customers at medium risk (0.4 < prob < 0.7): {((y_pred_proba > 0.4) & (y_pred_proba <= 0.7)).sum()}")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Logistic regression predicts probabilities** using sigmoid function
2. **Outputs interpretable coefficients** (odds ratios)
3. **Works well for linearly separable data**
4. **Regularization (L1/L2) prevents overfitting**
5. **Feature scaling critical** for proper convergence
6. **Threshold tuning** can optimize for specific metrics
7. **Multiclass via OvR or softmax**
8. **Fast training**, suitable for Big Data
:::

## Further Reading

- Hosmer, D. & Lemeshow, S. (2000). "Applied Logistic Regression"
- Hastie, T. et al. (2009). "The Elements of Statistical Learning", Chapter 4
