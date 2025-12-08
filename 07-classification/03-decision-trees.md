# Decision Trees

## Introduction

Decision trees are intuitive, interpretable classification (and regression) algorithms that make predictions by learning simple decision rules from data features. They recursively partition the feature space into regions and assign class labels based on majority voting within each region.

## Tree Structure

### Components

**Node Types**:
- **Root Node**: Top of tree, contains all data
- **Internal Nodes**: Decision points that split data
- **Leaf Nodes**: Terminal nodes with class predictions
- **Branches**: Connections representing decision outcomes

**Node Properties**:
- **Depth**: Distance from root (root = 0)
- **Split**: Test condition at internal node
- **Samples**: Number of training instances at node
- **Class Distribution**: Proportion of each class

### Example Tree Structure

```
                    [Root: Income]
                   /              \
            Income ≤ 50K       Income > 50K
               /                      \
        [Age Node]              [Education Node]
         /      \                /            \
    Age≤30   Age>30      Secondary      Tertiary
      /         \            |              |
   [No]       [Yes]       [No]          [Yes]
```

## Decision Tree Algorithms

### 1. ID3 (Iterative Dichotomiser 3)

**Developed by**: Ross Quinlan (1986)

**Splitting Criterion**: Information Gain (based on Entropy)

**Entropy**:

\[
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

Where:
- \(S\): Dataset at node
- \(c\): Number of classes
- \(p_i\): Proportion of class \(i\)

**Properties of Entropy**:
- Minimum (0): All samples same class (pure)
- Maximum (1): Classes equally distributed (binary classification)
- Maximum (\(\log_2 c\)): For \(c\) classes

**Information Gain**:

\[
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
\]

Where:
- \(A\): Attribute to split on
- \(S_v\): Subset where \(A = v\)

**Algorithm**:
1. Calculate entropy of target at current node
2. For each attribute, calculate information gain
3. Select attribute with highest information gain
4. Create child nodes for each attribute value
5. Recursively repeat for each child
6. Stop when:
   - All samples same class
   - No more attributes
   - Samples below threshold

**Example Calculation**:

Dataset: 9 Yes, 5 No

\[
H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = 0.940
\]

Attribute "Age" splits:
- Age ≤ 30: 2 Yes, 3 No → \(H = 0.971\)
- Age > 30: 7 Yes, 2 No → \(H = 0.764\)

\[
\text{IG}(S, \text{Age}) = 0.940 - \left(\frac{5}{14} \times 0.971 + \frac{9}{14} \times 0.764\right) = 0.048
\]

**Limitations**:
- Only handles categorical features
- Biased toward multi-valued attributes
- No pruning
- Overfitting tendency

### 2. C4.5 (Successor to ID3)

**Improvements over ID3**:

**1. Gain Ratio** (addresses multi-valued attribute bias):

\[
\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
\]

\[
\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}
\]

**2. Continuous Attributes**: 
- Sort values
- Consider midpoints as split candidates
- Choose split maximizing gain ratio

**3. Missing Values**:
- Distribute samples proportionally
- Use surrogate splits

**4. Pruning**:
- Post-pruning using error-based method
- Subtree replacement
- Subtree raising

**5. Rule Extraction**:
- Convert tree to rules
- Simplify rules

### 3. CART (Classification and Regression Trees)

**Developed by**: Breiman et al. (1984)

**Key Differences from ID3/C4.5**:
- Always **binary splits**
- Uses **Gini impurity** (not entropy)
- Handles **regression** and classification
- Cost-complexity pruning

**Gini Impurity**:

\[
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

**Properties**:
- Range: [0, 0.5] for binary classification
- 0: Pure node (all same class)
- 0.5: Maximum impurity (50/50 split)
- Faster to compute than entropy

**Gini Gain**:

\[
\text{GiniGain}(S, A) = \text{Gini}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Gini}(S_v)
\]

**Example**:

Node: 6 Yes, 4 No

\[
\text{Gini} = 1 - \left(\frac{6}{10}\right)^2 - \left(\frac{4}{10}\right)^2 = 1 - 0.36 - 0.16 = 0.48
\]

After split:
- Left: 5 Yes, 1 No → Gini = 0.278
- Right: 1 Yes, 3 No → Gini = 0.375

\[
\text{GiniGain} = 0.48 - \left(\frac{6}{10} \times 0.278 + \frac{4}{10} \times 0.375\right) = 0.163
\]

### Comparison: Entropy vs. Gini

| Aspect | Entropy | Gini |
|--------|---------|------|
| **Computation** | Slower (log) | Faster |
| **Range** | [0, log₂c] | [0, 0.5] |
| **Sensitivity** | More | Less |
| **Tree depth** | Deeper | Shallower |
| **Performance** | Similar | Similar |
| **Preference** | Balanced splits | Pure splits |

In practice, both produce similar trees.

## Handling Continuous Features

### Splitting Strategy

1. **Sort** feature values: \(v_1, v_2, ..., v_n\)
2. **Generate candidates**: Midpoints \(\frac{v_i + v_{i+1}}{2}\)
3. **Evaluate** each split: Calculate information gain
4. **Select** best split point

**Example**:

Age values: [22, 25, 30, 35, 40, 45, 50]

Candidate splits: [23.5, 27.5, 32.5, 37.5, 42.5, 47.5]

Evaluate:
- Age ≤ 23.5 vs. > 23.5
- Age ≤ 27.5 vs. > 27.5
- ...
- Age ≤ 47.5 vs. > 47.5

Choose split with highest gain.

### Optimization

**Naive approach**: O(n²) per feature

**Optimized**: 
- Pre-sort data: O(n log n)
- Linear scan for splits: O(n)
- Total: O(n log n) per feature

## Pruning Techniques

### Why Prune?

**Overfitting**: Full trees memorize training data
- High training accuracy
- Poor test accuracy
- Captures noise, not signal

**Solution**: Prune tree to reduce complexity

### Pre-Pruning (Early Stopping)

Stop growing tree based on criteria:

**1. Maximum Depth**:
```python
max_depth=10  # Limit tree depth
```

**2. Minimum Samples Split**:
```python
min_samples_split=20  # Don't split if < 20 samples
```

**3. Minimum Samples Leaf**:
```python
min_samples_leaf=10  # Each leaf must have ≥ 10 samples
```

**4. Maximum Leaf Nodes**:
```python
max_leaf_nodes=50  # Limit total leaf nodes
```

**5. Minimum Impurity Decrease**:
```python
min_impurity_decrease=0.01  # Split must reduce impurity by ≥ 0.01
```

**Advantages**:
- Fast
- Simple
- Memory efficient

**Disadvantages**:
- May stop too early
- Horizon effect (useful split later)

### Post-Pruning

Grow full tree, then prune back:

**Reduced Error Pruning**:
1. Grow full tree
2. For each internal node:
   - Convert to leaf (assign majority class)
   - Measure validation accuracy
   - Keep conversion if accuracy improves
3. Repeat until no improvement

**Cost-Complexity Pruning (CART)**:

Balance tree complexity and error:

\[
\text{Cost}(T) = \text{Error}(T) + \alpha |T_{leaves}|
\]

Where:
- \(\text{Error}(T)\): Misclassification rate
- \(|T_{leaves}|\): Number of leaf nodes
- \(\alpha\): Complexity parameter (tuned via CV)

**Procedure**:
1. Grow full tree
2. For \(\alpha\) from 0 to \(\infty\):
   - Prune subtrees that minimize cost
   - Create sequence of trees \(T_0, T_1, ..., T_k\)
3. Use cross-validation to select best \(\alpha\)

**In scikit-learn**:
```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf.fit(X_train, y_train)
```

## Feature Importance

Decision trees provide natural feature importance:

\[
\text{Importance}(f) = \sum_{\text{splits using } f} \frac{n_{\text{samples}}}{n_{\text{total}}} \times \text{ImpurityDecrease}
\]

**Interpretation**:
- Higher value = more important
- Normalized to sum to 1
- Top features appear near root

**Python Example**:
```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))
```

## Advantages

✅ **Interpretable**: Easy to understand and visualize
✅ **No preprocessing**: No scaling or encoding required (for tree-based methods)
✅ **Handles mixed data**: Categorical and numerical
✅ **Non-linear**: Captures complex relationships
✅ **Feature interactions**: Automatically detected
✅ **Missing values**: Can handle (some implementations)
✅ **Feature importance**: Built-in ranking
✅ **Fast prediction**: O(log n) depth

## Disadvantages

❌ **High variance**: Small data changes → different tree
❌ **Overfitting**: Without pruning
❌ **Biased**: Toward dominant classes
❌ **Greedy**: Locally optimal splits
❌ **Axis-aligned**: Only perpendicular splits
❌ **Extrapolation**: Poor outside training range
❌ **Instability**: Sensitive to data perturbations

## Implementation in Python

### Basic Usage

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train tree
clf = DecisionTreeClassifier(
    criterion='gini',           # or 'entropy'
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Export as Text Rules

```python
from sklearn.tree import export_text

# Text representation
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
```

Output:
```
|--- feature_2 <= 0.50
|   |--- feature_0 <= 0.75
|   |   |--- class: 0
|   |--- feature_0 >  0.75
|   |   |--- class: 1
|--- feature_2 >  0.50
|   |--- class: 1
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.3f}")

best_clf = grid_search.best_estimator_
```

### Cost-Complexity Pruning Path

```python
import numpy as np
import matplotlib.pyplot as plt

# Get pruning path
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Plot
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("alpha")
ax.set_ylabel("Total impurity")
ax.set_title("Total Impurity vs alpha for training set")

# Train trees with different alphas
clfs = []
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Plot accuracy vs alpha
fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs alpha")
ax.legend()
plt.show()
```

## Decision Tree for Regression

**Prediction**: Mean of target values in leaf

**Splitting Criterion**: Variance reduction

\[
\text{VarReduction}(S, A) = \text{Var}(S) - \sum_{v} \frac{|S_v|}{|S|} \text{Var}(S_v)
\]

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

## Real-World Example: Loan Approval

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('loan_data.csv')

# Features: income, age, employment_length, credit_score, debt_ratio
# Target: approved (1) or denied (0)

X = df[['income', 'age', 'employment_length', 'credit_score', 'debt_ratio']]
y = df['approved']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train decision tree
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,  # Keep interpretable
    min_samples_leaf=50,  # Ensure statistical significance
    random_state=42
)

clf.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns,
          class_names=['Denied', 'Approved'],
          filled=True,
          rounded=True,
          proportion=True,
          precision=2)
plt.savefig('loan_approval_tree.png', dpi=300, bbox_inches='tight')

# Extract rules
from sklearn.tree import export_text
rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Rules:")
print(rules)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importances)

# Evaluate
y_pred = clf.predict(X_test)
print("\nPerformance:")
print(classification_report(y_test, y_pred, target_names=['Denied', 'Approved']))
```

**Insights from tree**:
- Most important: credit_score
- Second: debt_ratio
- Third: income
- Rules are transparent and auditable
- Can explain decisions to loan applicants

## When to Use Decision Trees

### Good Choice When:

✅ **Interpretability required**: Explain decisions
✅ **Mixed data types**: Categorical + numerical
✅ **Non-linear relationships**: Complex interactions
✅ **No preprocessing time**: Quick prototyping
✅ **Feature interactions**: Important to capture
✅ **Baseline model**: First model to try

### Consider Alternatives When:

❌ **Stability needed**: Use ensemble methods
❌ **Linear relationships**: Use logistic regression
❌ **High-dimensional sparse data**: Use linear models
❌ **Small dataset**: Risk of overfitting
❌ **Extrapolation needed**: Use regression models

## Summary

Decision trees are powerful, interpretable classifiers that:

- Recursively partition feature space
- Use greedy algorithms (ID3, C4.5, CART)
- Split based on information gain or Gini impurity
- Handle both categorical and continuous features
- Provide natural feature importance
- Require pruning to prevent overfitting
- Serve as basis for ensemble methods
- Offer transparent, auditable decisions

While single trees are unstable and prone to overfitting, they form the foundation for powerful ensemble methods like Random Forests and Gradient Boosting.

## Further Reading

- "The Elements of Statistical Learning" - Chapter 9
- "Introduction to Data Mining" by Tan et al. - Chapter 4
- Original CART book by Breiman et al. (1984)
- C4.5 by Quinlan (1993)

## Next Section

Continue to [Ensemble Methods](04-random-forests.md) to learn how combining multiple trees improves performance.