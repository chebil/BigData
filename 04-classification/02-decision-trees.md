# Decision Trees

## Learning Objectives

- Understand decision tree structure and algorithms
- Build and visualize decision trees
- Interpret decision rules and feature importance
- Handle overfitting with pruning
- Apply decision trees to real-world problems
- Understand ID3, C4.5, and CART algorithms

## Introduction

Decision trees are intuitive, interpretable models that make decisions by learning simple decision rules from data. They're widely used in business for their transparency and ease of explanation to non-technical stakeholders.

## Decision Tree Structure

### Components

**Root Node**: Top node (entire dataset)

**Internal Nodes**: Decision points (test on attribute)

**Branches**: Outcomes of tests

**Leaf Nodes**: Final predictions (class labels)

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train simple tree
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X, y)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Decision Tree for Iris Classification', fontsize=16)
plt.show()

print("Tree structure:")
print(f"Number of nodes: {tree.tree_.node_count}")
print(f"Number of leaves: {tree.tree_.n_leaves}")
print(f"Maximum depth: {tree.tree_.max_depth}")
```

## How Decision Trees Learn

### Splitting Criteria

#### 1. Gini Impurity (CART)

**Measures probability of incorrect classification**

\[
\text{Gini}(t) = 1 - \sum_{i=1}^{C} p_i^2
\]

where \(p_i\) = proportion of class \(i\) at node \(t\)

**Range**: [0, 0.5] for binary classification
- **0**: Pure node (all same class)
- **0.5**: Maximum impurity (equal classes)

```python
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini impurity
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

# Example: Pure node
pure = np.array([0, 0, 0, 0, 0])
print(f"Pure node Gini: {gini_impurity(pure):.3f}")  # 0.0

# Example: Mixed node
mixed = np.array([0, 0, 0, 1, 1])
print(f"Mixed node Gini: {gini_impurity(mixed):.3f}")  # 0.48

# Example: Maximum impurity (50-50 split)
max_impure = np.array([0, 0, 1, 1])
print(f"Max impurity Gini: {gini_impurity(max_impure):.3f}")  # 0.5
```

#### 2. Entropy (ID3, C4.5)

**Measures information gain**

\[
\text{Entropy}(t) = -\sum_{i=1}^{C} p_i \log_2(p_i)
\]

**Information Gain**:
\[
\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
\]

```python
import numpy as np

def entropy(y):
    """
    Calculate entropy
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(y, X_feature, threshold):
    """
    Calculate information gain for a split
    """
    # Parent entropy
    parent_entropy = entropy(y)
    
    # Split
    left_mask = X_feature <= threshold
    right_mask = ~left_mask
    
    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0
    
    # Weighted child entropy
    n = len(y)
    n_left, n_right = len(y[left_mask]), len(y[right_mask])
    child_entropy = (n_left / n * entropy(y[left_mask]) +
                     n_right / n * entropy(y[right_mask]))
    
    return parent_entropy - child_entropy

# Example
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
print(f"Dataset entropy: {entropy(y):.3f}")

X_feature = np.array([1, 2, 3, 4, 5, 6, 7, 8])
threshold = 3.5
ig = information_gain(y, X_feature, threshold)
print(f"Information gain (threshold={threshold}): {ig:.3f}")
```

### Gini vs Entropy

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare Gini and Entropy
p = np.linspace(0.001, 0.999, 100)
gini = 2 * p * (1 - p)  # For binary classification
entropy_vals = -(p * np.log2(p) + (1-p) * np.log2(1-p))

plt.figure(figsize=(10, 6))
plt.plot(p, gini, label='Gini Impurity', linewidth=2)
plt.plot(p, entropy_vals, label='Entropy', linewidth=2, linestyle='--')
plt.xlabel('Proportion of Class 1')
plt.ylabel('Impurity')
plt.title('Gini Impurity vs Entropy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("Gini and Entropy are similar in practice")
print("Gini is slightly faster to compute")
```

## Building a Decision Tree from Scratch

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value
        self.left = left           # Left child
        self.right = right         # Right child
        self.value = value         # Class label (for leaf nodes)

class DecisionTreeScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _gini(self, y):
        counter = Counter(y)
        probabilities = np.array(list(counter.values())) / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        parent_gini = self._gini(y)
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate weighted Gini
                n = len(y)
                gini = (len(y_left) / n * self._gini(y_left) +
                       len(y_right) / n * self._gini(y_right))
                
                gain = parent_gini - gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create leaf node
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return Node(value=most_common)
        
        # Find best split
        feature, threshold, gain = self._best_split(X, y)
        
        if feature is None:  # No valid split found
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return Node(value=most_common)
        
        # Split data
        X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
        
        # Recursively build children
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return Node(feature=feature, threshold=threshold, 
                   left=left_child, right=right_child)
    
    def _traverse_tree(self, x, node):
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom tree
tree_custom = DecisionTreeScratch(max_depth=3)
tree_custom.fit(X_train, y_train)
y_pred = tree_custom.predict(X_test)

print("Custom Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Compare with sklearn
from sklearn.tree import DecisionTreeClassifier
tree_sklearn = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_sklearn.fit(X_train, y_train)
y_pred_sklearn = tree_sklearn.predict(X_test)

print(f"\nScikit-learn Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.2%}")
```

## Using Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train decision tree
tree = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=5,          # Limit depth to prevent overfitting
    min_samples_split=20,  # Minimum samples to split
    min_samples_leaf=10,   # Minimum samples in leaf
    random_state=42
)

tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)

# Evaluation
print("Decision Tree Performance:")
print(f"Training Accuracy: {tree.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {tree.score(X_test, y_test):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

# Feature importance
importances = tree.feature_importances_
feature_names = data.feature_names

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'][:10], 
         feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## Overfitting and Pruning

### Demonstrating Overfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate noisy data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = (np.sin(X).ravel() > 0).astype(int)
y[::10] = 1 - y[::10]  # Add noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train trees with different max_depth
depths = [2, 5, 10, None]  # None = unlimited depth

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, depth in enumerate(depths):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    # Plot decision boundary
    X_grid = np.linspace(0, 5, 300).reshape(-1, 1)
    y_grid = tree.predict(X_grid)
    
    ax = axes[idx // 2, idx % 2]
    ax.scatter(X_train, y_train, c=y_train, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.plot(X_grid, y_grid, 'r-', linewidth=2, label='Decision Boundary')
    
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    
    depth_str = depth if depth else 'Unlimited'
    ax.set_title(f'Max Depth: {depth_str}\nTrain Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

# Grid search
tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    tree, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.2%}")

# Test best model
best_tree = grid_search.best_estimator_
test_score = best_tree.score(X_test, y_test)
print(f"Test accuracy: {test_score:.2%}")
```

## Complete Example: Loan Approval

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/loan_data.csv')

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nTarget distribution:\n{df['loan_status'].value_counts()}")

# Feature engineering
le = LabelEncoder()
categorical_cols = ['gender', 'married', 'education', 'self_employed', 'property_area']

for col in categorical_cols:
    df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Missing'))

# Select features
feature_cols = [
    'income', 'coapplicant_income', 'loan_amount', 'loan_term',
    'credit_history', 'gender_encoded', 'married_encoded',
    'education_encoded', 'self_employed_encoded', 'property_area_encoded'
]

X = df[feature_cols].fillna(df[feature_cols].median())
y = (df['loan_status'] == 'Approved').astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train decision tree
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

tree.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# Predictions
y_pred = tree.predict(X_test)
y_pred_proba = tree.predict_proba(X_test)[:, 1]

# Evaluation
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training Accuracy: {tree.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {tree.score(X_test, y_test):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2%}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Visualize tree
plt.figure(figsize=(25, 15))
plot_tree(tree,
          feature_names=feature_cols,
          class_names=['Rejected', 'Approved'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Loan Approval Decision Tree', fontsize=20)
plt.savefig('loan_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': tree.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Loan Approval')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Extract decision rules
from sklearn.tree import _tree

def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child
            left_rule = rule + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_rule)
            
            # Right child
            right_rule = rule + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_rule)
        else:
            # Leaf node
            class_idx = np.argmax(tree_.value[node])
            rules.append({
                'rule': ' AND '.join(rule),
                'class': class_idx,
                'samples': tree_.n_node_samples[node]
            })
    
    recurse(0, [])
    return rules

rules = extract_rules(tree, feature_cols)

print("\n" + "="*60)
print("DECISION RULES (Sample)")
print("="*60)
for i, rule in enumerate(rules[:5], 1):
    outcome = 'APPROVED' if rule['class'] == 1 else 'REJECTED'
    print(f"\nRule {i} → {outcome} ({rule['samples']} samples)")
    print(f"  IF {rule['rule']}")
```

## Advantages and Disadvantages

### Advantages ✅

1. **Interpretable**: Easy to understand and explain
2. **No feature scaling needed**
3. **Handles mixed data types** (numerical and categorical)
4. **Non-parametric**: No assumptions about data distribution
5. **Feature importance** automatically calculated
6. **Fast prediction**

### Disadvantages ❌

1. **Prone to overfitting** (especially deep trees)
2. **Unstable**: Small data changes → different tree
3. **Biased toward dominant classes**
4. **Not optimal for XOR-type problems**
5. **Can create overly complex trees**

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Decision trees split data** based on feature thresholds
2. **Gini impurity and entropy** measure split quality
3. **Max depth** is critical hyperparameter to prevent overfitting
4. **Feature importance** shows which features matter most
5. **Pruning** (max_depth, min_samples) prevents overfitting
6. **Highly interpretable** - can extract decision rules
7. **No scaling required** unlike many algorithms
8. **Base for ensemble methods** (Random Forests, Gradient Boosting)
:::

## Further Reading

- Quinlan, J.R. (1986). "Induction of Decision Trees"
- Breiman, L. et al. (1984). "Classification and Regression Trees (CART)"
- Scikit-learn Decision Trees: [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html)
