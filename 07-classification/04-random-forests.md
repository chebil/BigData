# Ensemble Methods: Random Forests and Boosting

## Introduction

Ensemble methods combine multiple models to create a more powerful predictor. The key insight: many weak learners can form a strong learner. Two main approaches are **bagging** (Random Forests) and **boosting** (AdaBoost, Gradient Boosting, XGBoost).

## Why Ensemble Methods?

### The Wisdom of Crowds

**Principle**: Aggregate predictions from multiple models typically outperform any single model.

**Condorcet's Jury Theorem**: If each voter has probability \(p > 0.5\) of being correct, majority vote accuracy increases toward 1 as number of voters increases.

**Example**: 
- 1 model with 60% accuracy
- 100 models with 60% accuracy each (independent errors)
- Majority vote: ~95% accuracy

### Bias-Variance Tradeoff

**Total Error** = Bias² + Variance + Irreducible Error

- **High Bias**: Underfitting (too simple)
- **High Variance**: Overfitting (too complex)

**How Ensembles Help**:
- **Bagging**: Reduces variance (parallel ensemble)
- **Boosting**: Reduces bias (sequential ensemble)

## Bagging (Bootstrap Aggregating)

### Algorithm

1. **Bootstrap**: Create \(B\) bootstrap samples from training data
   - Sample with replacement
   - Each sample size \(n\) (same as original)
   
2. **Train**: Build model on each bootstrap sample
   - Models train independently (parallel)
   
3. **Aggregate**:
   - **Classification**: Majority vote
   - **Regression**: Average predictions

### Mathematical Foundation

For \(B\) models with predictions \(f_1(x), f_2(x), ..., f_B(x)\):

**Classification**:
\[
\hat{y} = \text{mode}(f_1(x), f_2(x), ..., f_B(x))
\]

**Regression**:
\[
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)
\]

### Variance Reduction

If models are independent with variance \(\sigma^2\):

\[
\text{Var}(\bar{f}) = \frac{\sigma^2}{B}
\]

Variance decreases as \(B\) increases!

But models aren't fully independent (trained on similar data), so:

\[
\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
\]

Where \(\rho\) is correlation between models.

**Strategy**: Reduce \(\rho\) by decorrelating models → **Random Forests**

## Random Forests

### Algorithm

**Extension of Bagging with Added Randomness**:

1. **Bootstrap**: Create \(B\) bootstrap samples

2. **For each tree**:
   - At each split, randomly select \(m\) features from \(p\) total
   - Choose best split from these \(m\) features only
   - Grow tree to maximum depth (no pruning)

3. **Aggregate**: 
   - Classification: Majority vote
   - Regression: Average

### Key Hyperparameters

**n_estimators** (\(B\)): Number of trees
- Default: 100
- More trees: Better performance, diminishing returns
- Typical: 100-500
- More trees never hurts (unlike boosting)

**max_features** (\(m\)): Features per split
- Classification default: \(\sqrt{p}\)
- Regression default: \(p/3\)
- Lower \(m\): More decorrelation, more bias
- Higher \(m\): Less decorrelation, less bias

**max_depth**: Tree depth
- Default: None (full depth)
- Limit for faster training, less overfitting

**min_samples_split**: Minimum samples to split node
- Default: 2
- Higher: Prevents overfitting

**min_samples_leaf**: Minimum samples in leaf
- Default: 1
- Higher: Smoother decision boundaries

**bootstrap**: Use bootstrap samples
- Default: True
- False: Use all data (loses variance reduction benefit)

### Out-of-Bag (OOB) Error

**Key Insight**: Each bootstrap sample excludes ~37% of data

\[
P(\text{sample } i \text{ not selected}) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.37
\]

**OOB Prediction**: For each instance, average predictions from trees where it was OOB

**OOB Error**: Error rate on OOB predictions
- Free cross-validation estimate
- No need for separate validation set
- Slightly conservative (fewer trees per instance)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.3f}")
```

### Feature Importance

Two methods:

**1. Mean Decrease in Impurity (MDI)**:
- Average impurity decrease across all trees
- Biased toward high-cardinality features
- Fast to compute (during training)

**2. Mean Decrease in Accuracy (MDA)** (Permutation Importance):
- Shuffle feature values
- Measure decrease in OOB accuracy
- More reliable
- Computationally expensive

```python
# MDI (built-in)
importances = rf.feature_importances_

# Permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
importances_perm = result.importances_mean
```

### Advantages

✅ **High accuracy**: Often best off-the-shelf classifier
✅ **Robust to overfitting**: Unlike single trees
✅ **Handles high dimensions**: Thousands of features
✅ **Feature importance**: Built-in ranking
✅ **Missing values**: Can handle (with extra work)
✅ **Parallel training**: Fast with multiple cores
✅ **OOB error**: Free validation estimate
✅ **Works out-of-box**: Minimal tuning needed

### Disadvantages

❌ **Less interpretable**: Than single tree
❌ **Slower prediction**: Than single tree
❌ **Memory intensive**: Stores many trees
❌ **Extrapolation**: Poor outside training range
❌ **Imbalanced data**: Can favor majority class

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

# Basic Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

## Boosting

### Core Idea

**Sequential learning**: Each model corrects errors of previous models

1. Train weak learner on data
2. Identify mistakes
3. Focus next learner on mistakes
4. Repeat
5. Combine models (weighted by performance)

### AdaBoost (Adaptive Boosting)

**Algorithm**:

1. **Initialize**: Uniform sample weights \(w_i = 1/n\)

2. **For** \(m = 1\) to \(M\):
   - Train classifier \(f_m\) on weighted data
   - Calculate error: \(\epsilon_m = \sum_{i: y_i \neq f_m(x_i)} w_i\)
   - Calculate weight: \(\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}\)
   - Update weights: 
     \[
     w_i \leftarrow w_i \times \begin{cases}
     e^{\alpha_m} & \text{if misclassified} \\
     e^{-\alpha_m} & \text{if correct}
     \end{cases}
     \]
   - Normalize weights: \(w_i \leftarrow w_i / \sum_j w_j\)

3. **Final prediction**:
   \[
   F(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m f_m(x)\right)
   \]

**Key Points**:
- Focuses on hard examples (high weights)
- Better models get higher vote (\(\alpha_m\))
- Sensitive to outliers
- Can overfit with too many iterations

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Use shallow trees (weak learners)
base_clf = DecisionTreeClassifier(max_depth=1)  # Decision stump

ada = AdaBoostClassifier(
    base_estimator=base_clf,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
```

### Gradient Boosting

**Core Idea**: Fit each new model to **residuals** (errors) of ensemble so far

**Algorithm**:

1. **Initialize**: \(F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c)\)
   - For squared error: \(F_0(x) = \bar{y}\)

2. **For** \(m = 1\) to \(M\):
   - Compute residuals: \(r_{im} = y_i - F_{m-1}(x_i)\)
   - Fit tree \(h_m(x)\) to residuals \(r_{im}\)
   - Update: \(F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)\)
     - \(\nu\): Learning rate

3. **Final model**: \(F(x) = F_0(x) + \nu \sum_{m=1}^{M} h_m(x)\)

**Loss Functions**:
- **Regression**: Squared error, absolute error, Huber
- **Classification**: Logistic loss, exponential loss

**Regularization**:
- **Learning rate** (\(\nu\)): Shrinkage, typically 0.01-0.3
- **Subsample**: Use random subset per tree (stochastic GB)
- **Max depth**: Shallow trees (3-8)
- **Min samples leaf**: Higher values prevent overfitting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

gb.fit(X_train, y_train)
```

**Early Stopping**:
```python
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

gb.fit(X_train, y_train)
print(f"Stopped at {gb.n_estimators_} trees")
```

### XGBoost (Extreme Gradient Boosting)

**Improvements over standard GB**:

1. **Regularization**: L1 and L2 on leaf weights
2. **Column sampling**: Like Random Forests
3. **Parallel processing**: Tree construction
4. **Cache optimization**: Hardware efficiency
5. **Missing values**: Learns best direction
6. **Tree pruning**: Max depth then prune back
7. **Built-in CV**: Cross-validation during training

**Installation**:
```bash
pip install xgboost
```

**Usage**:
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# DMatrix format (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,  # L1 regularization
    'reg_lambda': 1,  # L2 regularization
    'seed': 42
}

# Train with early stopping
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=50
)

# Predict
y_prob = model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

# Feature importance
importance = model.get_score(importance_type='weight')
print(importance)
```

**Scikit-learn API**:
```python
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

y_pred = xgb_clf.predict(X_test)
```

### LightGBM

**Microsoft's gradient boosting framework**:

**Key Innovations**:
1. **Leaf-wise** tree growth (vs. level-wise)
2. **Histogram-based** splitting
3. **Gradient-based One-Side Sampling** (GOSS)
4. **Exclusive Feature Bundling** (EFB)
5. **Categorical features**: Native support

**Advantages**:
- Faster training than XGBoost
- Lower memory usage
- Better accuracy (often)
- Handles large datasets well

```python
import lightgbm as lgb

# Dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    early_stopping_rounds=10
)

# Predict
y_prob = model.predict(X_test)
```

### CatBoost

**Yandex's gradient boosting** with categorical features focus:

**Features**:
1. **Ordered boosting**: Reduces overfitting
2. **Categorical encoding**: Built-in target encoding
3. **Symmetric trees**: Faster prediction
4. **GPU support**: Efficient training

```python
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    cat_features=['category_col1', 'category_col2'],
    early_stopping_rounds=10,
    verbose=50
)

cat_clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test)
)
```

## Comparison

| Method | Speed | Accuracy | Interpretability | Overfitting | Best For |
|--------|-------|----------|------------------|-------------|----------|
| **Single Tree** | Fast | Low | Excellent | High | Exploration |
| **Random Forest** | Moderate | High | Low | Low | General use |
| **AdaBoost** | Fast | Moderate | Low | Moderate | Binary classification |
| **Gradient Boosting** | Slow | Very High | Low | High | Competitions |
| **XGBoost** | Moderate | Very High | Low | Moderate | Production |
| **LightGBM** | Fast | Very High | Low | Moderate | Large data |
| **CatBoost** | Moderate | Very High | Low | Low | Categorical data |

## Hyperparameter Tuning Strategy

### Random Forest

**Priority Order**:
1. `n_estimators`: 100-500 (more is better)
2. `max_features`: sqrt(p), log2(p), 0.3
3. `max_depth`: 10-30 or None
4. `min_samples_split`: 2-20
5. `min_samples_leaf`: 1-10

### XGBoost

**Step 1**: Fix learning rate (0.1), tune tree parameters
- `max_depth`: 3-10
- `min_child_weight`: 1-10
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0

**Step 2**: Tune regularization
- `reg_alpha`: 0-10
- `reg_lambda`: 0-10

**Step 3**: Lower learning rate, increase n_estimators
- `learning_rate`: 0.01-0.05
- `n_estimators`: Find optimal with early stopping

## Practical Tips

### Random Forest
1. Start with defaults
2. Increase `n_estimators` until OOB error plateaus
3. Tune `max_features` for bias-variance tradeoff
4. Add pruning if overfitting

### Gradient Boosting
1. Start: `learning_rate=0.1`, `n_estimators=100`
2. Tune tree structure (`max_depth`, `min_samples_leaf`)
3. Add regularization (`subsample`, `max_features`)
4. Reduce `learning_rate`, increase `n_estimators`
5. **Always use early stopping**

### XGBoost/LightGBM
1. Start with package defaults
2. Enable early stopping
3. Tune `max_depth` and `min_child_weight`
4. Add subsampling
5. Fine-tune learning rate
6. Add regularization if needed

## Summary

Ensemble methods significantly improve over single models by:

**Bagging (Random Forests)**:
- Reduces variance through averaging
- Trains trees independently in parallel
- Robust and easy to use
- Excellent for general classification

**Boosting (GB, XGBoost, LightGBM)**:
- Reduces bias through sequential learning
- Each model corrects previous errors
- State-of-the-art performance
- Requires careful tuning

**When to Use**:
- **Random Forest**: First choice, robust, minimal tuning
- **XGBoost**: When accuracy is critical, have time to tune
- **LightGBM**: Large datasets, speed important
- **CatBoost**: Many categorical features

Ensemble methods dominate data science competitions and production systems for tabular data.

## Next Section

Continue to [Support Vector Machines](05-svm.md) to learn about maximum margin classifiers.