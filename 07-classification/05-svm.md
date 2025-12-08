# Support Vector Machines (SVM)

## Introduction

Support Vector Machines are powerful supervised learning algorithms for classification and regression. SVMs find the optimal hyperplane that maximally separates classes in high-dimensional space. The key innovation is the **kernel trick**, which allows SVMs to efficiently handle non-linear decision boundaries.

## Intuition

### Linear Separation

For linearly separable data, infinite hyperplanes can separate classes:

```
Class A: ●●●●●
Class B:           ○○○○○
```

**Question**: Which separating line is best?

**SVM Answer**: The one with **maximum margin** from both classes.

### Maximum Margin

**Margin**: Distance from hyperplane to nearest data point

**Maximum Margin Hyperplane**: 
- Maximizes distance to closest points
- Most robust to new data
- Better generalization

**Support Vectors**: Data points closest to hyperplane
- Determine the boundary
- Remove non-support vectors → same hyperplane
- Typically small subset of training data

## Mathematical Formulation

### Linear SVM

**Hyperplane** in \(p\)-dimensional space:

\[
w^T x + b = 0
\]

Where:
- \(w\): Normal vector (perpendicular to hyperplane)
- \(b\): Bias term (intercept)
- \(x\): Feature vector

**Decision Function**:

\[
f(x) = \text{sign}(w^T x + b)
\]

- \(f(x) = +1\): Class positive
- \(f(x) = -1\): Class negative

### Margin

**Distance** from point \(x_i\) to hyperplane:

\[
\text{distance} = \frac{|w^T x_i + b|}{||w||}
\]

**Margin**: Minimum distance among all points:

\[
\text{margin} = \min_i \frac{|w^T x_i + b|}{||w||}
\]

For correctly classified points: \(y_i(w^T x_i + b) > 0\)

### Optimization Problem

**Goal**: Maximize margin

\[
\max_{w, b} \frac{1}{||w||}
\]

Subject to: \(y_i(w^T x_i + b) \geq 1\) for all \(i\)

**Equivalent** (easier to solve):

\[
\min_{w, b} \frac{1}{2} ||w||^2
\]

Subject to: \(y_i(w^T x_i + b) \geq 1\) for all \(i\)

This is a **convex quadratic programming** problem.

### Dual Formulation

Using Lagrange multipliers \(\alpha_i \geq 0\):

\[
\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j
\]

Subject to:
- \(\sum_{i=1}^{n} \alpha_i y_i = 0\)
- \(\alpha_i \geq 0\) for all \(i\)

**Key Insight**: Only depends on \(x_i^T x_j\) (dot products)

**Solution**:

\[
w = \sum_{i=1}^{n} \alpha_i y_i x_i
\]

**Support Vectors**: Points where \(\alpha_i > 0\)

## Soft Margin SVM

### The Problem

Real data is rarely perfectly separable:
- Outliers
- Noise
- Overlapping classes

**Hard margin** SVM: No misclassification allowed → May not have solution

### Solution: Slack Variables

Introduce slack variables \(\xi_i \geq 0\) to allow violations:

\[
y_i(w^T x_i + b) \geq 1 - \xi_i
\]

- \(\xi_i = 0\): Correct side, outside margin
- \(0 < \xi_i < 1\): Correct side, inside margin  
- \(\xi_i \geq 1\): Wrong side (misclassified)

### Optimization with Penalty

\[
\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
\]

Subject to:
- \(y_i(w^T x_i + b) \geq 1 - \xi_i\)
- \(\xi_i \geq 0\)

**C Parameter** (Cost/Regularization):
- **Large C**: Small margin, few violations (low bias, high variance)
- **Small C**: Large margin, more violations (high bias, low variance)
- Tradeoff: Margin size vs. training error

### Tuning C

**C = 0.01**: Very soft margin
- Wide margin
- Many support vectors
- Underfitting risk

**C = 1**: Moderate (default in sklearn)
- Balanced
- Good starting point

**C = 100**: Nearly hard margin
- Narrow margin
- Few violations
- Overfitting risk

**Best practice**: Tune via cross-validation

## Kernel Trick

### Non-Linear Decision Boundaries

**Problem**: Real data often not linearly separable

**Solution**: Map to higher-dimensional space where linear separation possible

**Example**: 2D → 3D

Original features: \((x_1, x_2)\)

Map to: \(\phi(x) = (x_1, x_2, x_1^2, x_2^2, x_1 x_2, ...)\)

Now linearly separable in higher dimension!

### The Trick

**Problem**: High-dimensional \(\phi(x)\) is expensive to compute

**Kernel Trick**: Never explicitly compute \(\phi(x)\)

Recall dual formulation depends only on \(x_i^T x_j\)

Replace with: \(K(x_i, x_j) = \phi(x_i)^T \phi(x_j)\)

**Kernel function** computes dot product in high-dimensional space efficiently!

### Common Kernels

#### 1. Linear Kernel

\[
K(x, x') = x^T x'
\]

- No transformation
- Fastest
- Use when data is linearly separable

#### 2. Polynomial Kernel

\[
K(x, x') = (x^T x' + c)^d
\]

Where:
- \(d\): Degree (2, 3, 4, ...)
- \(c\): Coefficient (often 1)

**Degree 2**: \((x^T x' + 1)^2\)
- Implicitly computes all pairwise feature products
- Good for image data

**Higher degree**:
- More flexible
- Higher risk of overfitting
- More expensive

#### 3. RBF Kernel (Radial Basis Function)

\[
K(x, x') = \exp\left(-\gamma ||x - x'||^2\right)
\]

Also called **Gaussian kernel**.

Where:
- \(\gamma = \frac{1}{2\sigma^2}\): Width parameter
- \(||x - x'||^2\): Squared Euclidean distance

**Intuition**: Similarity decreases with distance

**Gamma (\(\gamma\))**:
- **Large \(\gamma\)**: Narrow influence, complex boundary, overfitting
- **Small \(\gamma\)**: Wide influence, smooth boundary, underfitting
- **Default**: \(1/p\) (p = number of features)

**Most popular kernel**: Works well for many problems

#### 4. Sigmoid Kernel

\[
K(x, x') = \tanh(\alpha x^T x' + c)
\]

- Similar to neural network
- Rarely used
- Can be unstable

### Choosing a Kernel

**Decision Tree**:

1. **Try linear first**: Fast, interpretable
   - If good: Done!
   - If poor: Try non-linear

2. **Try RBF**: Default non-linear choice
   - Works for most problems
   - Tune C and gamma

3. **Try polynomial**: If RBF slow
   - Faster than RBF for some data
   - Start with degree 2-3

4. **Custom kernel**: Domain-specific
   - Text: String kernels
   - Graphs: Graph kernels
   - Requires kernel to be positive semi-definite

## Multi-Class SVM

SVM is inherently binary. For multi-class:

### 1. One-vs-Rest (OvR)

- Train \(K\) binary classifiers
- Class \(k\) vs. all others
- Predict: Class with highest decision function value

**Pros**: Simple, efficient
**Cons**: Imbalanced training data

### 2. One-vs-One (OvO)

- Train \(K(K-1)/2\) classifiers
- Each pair of classes
- Predict: Majority voting

**Pros**: Balanced data per classifier
**Cons**: Many classifiers (\(O(K^2)\))

**Default in scikit-learn**: OvO

## Implementation

### Linear SVM

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANT: Always scale features for SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Coefficients
w = pipeline.named_steps['svm'].coef_
b = pipeline.named_steps['svm'].intercept_
```

### Non-Linear SVM

```python
from sklearn.svm import SVC

# RBF kernel (default)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
svm_poly.fit(X_train_scaled, y_train)

# Number of support vectors
print(f"Support vectors: {len(svm_rbf.support_vectors_)}")
print(f"Per class: {svm_rbf.n_support_}")
```

### Probability Estimates

```python
# Enable probability estimates (slower)
svm_prob = SVC(kernel='rbf', probability=True, random_state=42)
svm_prob.fit(X_train_scaled, y_train)

y_prob = svm_prob.predict_proba(X_test_scaled)
```

**Note**: Uses Platt scaling (internal cross-validation), computationally expensive.

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

# Grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

best_svm = grid.best_estimator_
```

### Randomized Search (Faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# Distributions
param_dist = {
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e1),
    'kernel': ['rbf']
}

random_search = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
```

## Scaling: Critical for SVM

**Why?** SVM is distance-based

**Example**:
- Feature 1: Income ($20K-$200K)
- Feature 2: Age (20-70)

Income dominates distance calculation!

**Solution**: Standardize

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

**Alternative**: MinMaxScaler (scales to [0, 1])

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Computational Complexity

**Training**:
- Best case: \(O(n^2 p)\)
- Worst case: \(O(n^3 p)\)
- Where \(n\) = samples, \(p\) = features

**Memory**: \(O(n^2)\) for kernel matrix

**Prediction**: \(O(s \times p)\)
- \(s\) = number of support vectors

**Implication**: 
- Slow for large \(n\) (>10K samples)
- Use LinearSVC or SGDClassifier for large data
- Consider sampling or approximate methods

## Advantages

✅ **Effective in high dimensions**: Even \(p > n\)
✅ **Memory efficient**: Only stores support vectors
✅ **Versatile**: Different kernels for different data
✅ **Robust**: Less prone to overfitting (with right C)
✅ **Well-theorized**: Strong mathematical foundation
✅ **Good for small-medium datasets**: Especially high-dimensional

## Disadvantages

❌ **Slow training**: For large \(n\) (>10K)
❌ **Memory intensive**: \(O(n^2)\) for kernel matrix
❌ **Requires scaling**: Sensitive to feature scales
❌ **Black box**: Hard to interpret (especially with RBF)
❌ **Hyperparameter tuning**: C and gamma need careful tuning
❌ **No probability estimates**: Without extra computation
❌ **Multiclass**: Not natural (requires extensions)

## When to Use SVM

### Good Choice:

✅ **High-dimensional data**: Text, genomics (\(p\) >> \(n\))
✅ **Clear margin**: Well-separated classes
✅ **Medium-sized data**: 1K-10K samples
✅ **Non-linear patterns**: With RBF kernel
✅ **Binary classification**: Natural fit

### Consider Alternatives:

❌ **Large data**: Use logistic regression or neural networks
❌ **Very high \(n\)**: Use linear models or tree ensembles
❌ **Interpretability needed**: Use logistic regression or trees
❌ **Probability calibration**: Use logistic regression
❌ **Categorical features**: Use tree-based methods

## Example: Text Classification

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load text data
df = pd.read_csv('news_articles.csv')
X = df['text']
y = df['category']  # politics, sports, technology

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: TF-IDF + Linear SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('svm', LinearSVC(C=1.0, max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Most important words per class
feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
coefs = pipeline.named_steps['svm'].coef_

for i, category in enumerate(pipeline.named_steps['svm'].classes_):
    top_indices = coefs[i].argsort()[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"\nTop words for {category}:")
    print(', '.join(top_words))
```

## SVM vs. Other Classifiers

| Aspect | SVM | Logistic Regression | Random Forest |
|--------|-----|---------------------|---------------|
| **Training Speed** | Slow | Fast | Moderate |
| **Prediction Speed** | Fast | Very Fast | Fast |
| **High Dimensions** | Excellent | Good | Poor |
| **Large Data** | Poor | Good | Good |
| **Interpretability** | Low | High | Medium |
| **Hyperparameters** | C, gamma | C, penalty | Many |
| **Probability Output** | Extra cost | Native | Native |
| **Feature Scaling** | Required | Recommended | Not needed |
| **Handles Outliers** | Moderate | Poor | Good |
| **Non-linearity** | Kernel | No | Yes |

## Tips and Best Practices

1. **Always scale features**: StandardScaler before SVM

2. **Start with linear kernel**: If good, done. If not, try RBF.

3. **Use LinearSVC for large data**: Faster than SVC(kernel='linear')

4. **Tune C first, then gamma**: Coarse grid, then fine grid

5. **Use cross-validation**: Essential for hyperparameter tuning

6. **Check number of support vectors**: 
   - High percentage (>50%): Consider simpler model
   - Low percentage (<10%): Model working well

7. **For probability estimates**: Consider logistic regression instead

8. **Imbalanced data**: Use `class_weight='balanced'`

9. **Large datasets**: 
   - Use LinearSVC with SGD solver
   - Or sample data for RBF kernel

10. **Monitor training time**: If > 5 minutes, consider alternatives

## Advanced Topics

### 1. Nu-SVM

Alternative formulation using \(\nu\) instead of C:
- \(\nu \in (0, 1]\): Upper bound on fraction of margin errors
- Easier to interpret than C
- Same performance as C-SVM

```python
from sklearn.svm import NuSVC

nu_svm = NuSVC(nu=0.1, kernel='rbf', random_state=42)
nu_svm.fit(X_train_scaled, y_train)
```

### 2. One-Class SVM

Anomal detection / novelty detection:
- Learn boundary around normal data
- Identify outliers

```python
from sklearn.svm import OneClassSVM

oclf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
oclf.fit(X_train_scaled)  # Only normal data

y_pred = oclf.predict(X_test_scaled)  # +1: normal, -1: anomaly
```

### 3. Custom Kernels

Define your own kernel function:

```python
def custom_kernel(X, Y):
    # Must return Gram matrix
    return np.dot(X, Y.T) ** 2  # Example: squared dot product

svm = SVC(kernel=custom_kernel)
svm.fit(X_train, y_train)
```

## Summary

Support Vector Machines are powerful classifiers that:

- Find maximum margin hyperplane separating classes
- Use support vectors (data points on margin)
- Handle non-linearity via kernel trick
- Work well in high-dimensional spaces
- Require careful feature scaling
- Need hyperparameter tuning (C, gamma)
- Excel for text classification and small-medium datasets
- Slow for large datasets (>10K samples)

**Key Parameters**:
- **C**: Regularization (smaller = more regularization)
- **gamma**: RBF kernel width (larger = more complex)
- **kernel**: linear, rbf, poly, sigmoid

**Best Practices**:
- Scale features
- Try linear first, then RBF
- Tune hyperparameters
- Use LinearSVC for large data
- Consider alternatives for very large datasets

SVMs remain relevant despite deep learning, especially for:
- High-dimensional data
- Small-medium datasets
- Text classification
- Scenarios requiring theoretical guarantees

## Further Reading

- "The Elements of Statistical Learning" - Chapter 12
- "Pattern Recognition and Machine Learning" - Chapter 7
- Original paper: Cortes & Vapnik (1995)
- "A Tutorial on Support Vector Machines" by Burges (1998)

## Next Section

Continue to [Model Evaluation](06-model-evaluation.md) for comprehensive evaluation strategies.