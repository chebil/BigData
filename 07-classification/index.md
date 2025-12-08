# Chapter 7: Classification Methods

## Overview

Classification is a fundamental supervised learning technique that assigns class labels to observations based on their features. Unlike clustering (unsupervised), classification learns from labeled training data to predict categories for new, unseen instances. This chapter explores major classification algorithms including logistic regression, Naïve Bayes, decision trees, ensemble methods, and support vector machines.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand classification fundamentals** and differences from regression and clustering
2. **Implement logistic regression** for binary and multiclass classification
3. **Apply Naïve Bayes classifier** using conditional probability
4. **Build decision trees** using ID3, C4.5, and CART algorithms
5. **Use ensemble methods** including Random Forest and boosting
6. **Apply Support Vector Machines** with different kernels
7. **Evaluate classifiers** using confusion matrices, ROC curves, and cross-validation
8. **Handle imbalanced datasets** and select appropriate metrics
9. **Implement classification** pipelines in scikit-learn
10. **Deploy classification models** for real-world applications

## Topics Covered

### 1. Classification Fundamentals
- Supervised learning paradigm
- Training and testing sets
- Overfitting and underfitting
- Bias-variance tradeoff
- Feature engineering for classification

### 2. Logistic Regression
- Binary logistic regression
- Sigmoid function and odds ratio
- Maximum likelihood estimation
- Multiclass classification (one-vs-rest, multinomial)
- Regularization (L1, L2)
- ROC curves and threshold selection

### 3. Naïve Bayes Classifier
- Bayes' theorem fundamentals
- Conditional independence assumption
- Gaussian Naïve Bayes
- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes
- Laplace smoothing
- Text classification applications

### 4. Decision Trees
- Tree structure (nodes, branches, leaves)
- Splitting criteria (Entropy, Information Gain, Gini index)
- ID3 algorithm
- C4.5 improvements
- CART (Classification and Regression Trees)
- Pruning techniques
- Handling continuous and categorical variables

### 5. Ensemble Methods
- Bagging (Bootstrap Aggregating)
- Random Forests
- Boosting (AdaBoost, Gradient Boosting)
- XGBoost and LightGBM
- Voting classifiers
- Stacking

### 6. Support Vector Machines
- Maximum margin classifier
- Support vectors
- Kernel trick (linear, polynomial, RBF, sigmoid)
- Soft margin and C parameter
- Multi-class SVM

### 7. Model Evaluation
- Confusion matrix
- Accuracy, precision, recall, F1-score
- ROC curve and AUC
- Precision-Recall curves
- Cross-validation strategies
- Stratified sampling

## Chapter Sections

```{tableofcontents}
```

## Classification vs. Other Learning Methods

### Classification vs. Regression
- **Classification**: Predicts discrete labels (categories)
  - Examples: spam/not spam, disease present/absent, customer will churn/stay
- **Regression**: Predicts continuous values
  - Examples: house price, temperature, stock price

### Classification vs. Clustering
- **Classification (Supervised)**: Uses labeled training data
  - Goal: Learn mapping from features to known labels
  - Evaluation: Compare predictions to actual labels
- **Clustering (Unsupervised)**: No labels provided
  - Goal: Discover inherent groupings in data
  - Evaluation: Internal metrics (silhouette score, etc.)

## Logistic Regression

### Binary Classification

Logistic regression models the probability that an instance belongs to a particular class. Despite its name, it's used for classification, not regression.

**Logistic Function (Sigmoid)**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where \(z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n\)

**Properties**:
- Output range: (0, 1)
- Interpreted as probability: \(P(Y=1|X)\)
- Decision boundary at 0.5 (adjustable)

### Odds and Log-Odds

**Odds**: Ratio of probability of success to failure

\[
\text{Odds} = \frac{P(Y=1)}{P(Y=0)} = \frac{P(Y=1)}{1 - P(Y=1)}
\]

**Log-Odds (Logit)**:

\[
\log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n
\]

### ROC Curve and Threshold Selection

The Receiver Operating Characteristic (ROC) curve plots:
- **True Positive Rate (TPR)** = Sensitivity = Recall on y-axis
- **False Positive Rate (FPR)** = 1 - Specificity on x-axis

\[
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

\[
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
\]

**Area Under Curve (AUC)**:
- Range: [0.5, 1.0]
- 0.5: Random classifier
- 1.0: Perfect classifier
- Interpretation: Probability that classifier ranks random positive instance higher than random negative instance

### Threshold Selection Strategy

**Default threshold**: 0.5

**Custom thresholds** based on business requirements:
- **High precision needed**: Increase threshold (e.g., 0.7)
  - Use case: Email spam detection (avoid false positives)
- **High recall needed**: Decrease threshold (e.g., 0.3)
  - Use case: Disease screening (avoid false negatives)

## Naïve Bayes Classifier

### Bayes' Theorem

\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

Where:
- \(C\): Class label
- \(X\): Feature vector
- \(P(C|X)\): Posterior probability
- \(P(X|C)\): Likelihood
- \(P(C)\): Prior probability
- \(P(X)\): Evidence (marginal probability)

### Naïve Independence Assumption

Assumes features are conditionally independent given the class:

\[
P(X|C) = P(x_1|C) \cdot P(x_2|C) \cdots P(x_n|C) = \prod_{i=1}^{n} P(x_i|C)
\]

### Classification Rule

Choose class \(c\) that maximizes:

\[
\hat{c} = \arg\max_c P(C=c) \prod_{i=1}^{n} P(x_i|C=c)
\]

### Laplace Smoothing

Handles zero probability problem when feature value not seen in training:

\[
P(x_i|C) = \frac{\text{count}(x_i, C) + \alpha}{\text{count}(C) + \alpha \cdot |V|}
\]

Where:
- \(\alpha\): Smoothing parameter (typically 1)
- \(|V|\): Number of possible values for feature

### Variants

**Gaussian Naïve Bayes**: For continuous features
- Assumes normal distribution
- \(P(x_i|C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} e^{-\frac{(x_i-\mu_C)^2}{2\sigma_C^2}}\)

**Multinomial Naïve Bayes**: For discrete counts
- Text classification with word counts
- Document categorization

**Bernoulli Naïve Bayes**: For binary features
- Text classification with word presence/absence
- Binary feature vectors

## Decision Trees

### Algorithm Overview

1. **Start** with all training data at root
2. **Select** best feature to split on (maximize information gain)
3. **Partition** data based on feature values
4. **Repeat** recursively for each partition
5. **Stop** when:
   - Node is pure (all same class)
   - No more features
   - Minimum samples reached
   - Maximum depth reached

### Splitting Criteria

**Entropy** (Information Theory):

\[
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

Where:
- \(S\): Dataset
- \(c\): Number of classes
- \(p_i\): Proportion of class \(i\)

**Information Gain**:

\[
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
\]

Where:
- \(A\): Attribute
- \(S_v\): Subset of \(S\) where attribute \(A\) has value \(v\)

**Gini Impurity** (CART):

\[
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

**Information Gain Ratio** (C4.5): Normalizes by split information to handle bias toward high-cardinality attributes

### Pruning

**Pre-pruning (Early Stopping)**:
- Set maximum depth
- Set minimum samples per leaf
- Set minimum information gain threshold

**Post-pruning**:
- Grow full tree
- Remove branches that don't improve validation performance
- Reduced error pruning
- Cost complexity pruning

### Advantages

✅ Easy to understand and interpret
✅ Visual representation
✅ Handles both numerical and categorical data
✅ No feature scaling required
✅ Captures non-linear relationships
✅ Variable interactions naturally handled
✅ Feature importance scores

### Disadvantages

❌ High variance (unstable)
❌ Greedy algorithm (locally optimal)
❌ Axis-aligned splits only
❌ Overfitting tendency
❌ Biased toward features with many levels
❌ Struggles with imbalanced data

## Ensemble Methods

### Random Forest

**Algorithm**:
1. Create \(n\) bootstrap samples from training data
2. For each sample, build decision tree with:
   - Random subset of features at each split
   - Grow to maximum depth (no pruning)
3. Aggregate predictions:
   - **Classification**: Majority vote
   - **Regression**: Average

**Key Parameters**:
- `n_estimators`: Number of trees (100-500)
- `max_features`: Features per split (\(\sqrt{n}\) for classification)
- `max_depth`: Tree depth (None for full growth)
- `min_samples_split`: Minimum samples to split node

**Advantages**:
- ✅ Reduces overfitting vs. single tree
- ✅ Handles large datasets efficiently
- ✅ Estimates feature importance
- ✅ Works with missing values
- ✅ Robust to outliers

### Gradient Boosting

**Algorithm**:
1. Initialize with simple model (e.g., mean)
2. For \(m = 1\) to \(M\):
   - Compute residuals (errors)
   - Fit new model to residuals
   - Add model to ensemble with learning rate
3. Final prediction = sum of all models

**XGBoost Improvements**:
- Regularization (L1, L2)
- Parallel processing
- Tree pruning
- Built-in cross-validation
- Handling missing values

**Key Parameters**:
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `n_estimators`: Number of boosting rounds
- `max_depth`: Tree depth (3-10)
- `subsample`: Row sampling rate
- `colsample_bytree`: Column sampling rate

## Support Vector Machines (SVM)

### Linear SVM

**Objective**: Find hyperplane that maximizes margin between classes

\[
\text{Maximize: } \frac{2}{||w||}
\]

Subject to: \(y_i(w \cdot x_i + b) \geq 1\) for all \(i\)

**Support Vectors**: Data points closest to decision boundary

### Soft Margin (C Parameter)

Allows misclassification with penalty:
- **Large C**: Hard margin, low bias, high variance
- **Small C**: Soft margin, high bias, low variance

### Kernel Trick

Maps data to higher dimensions without explicit computation:

**Linear Kernel**:
\[K(x, x') = x \cdot x'\]

**Polynomial Kernel**:
\[K(x, x') = (x \cdot x' + c)^d\]

**RBF (Radial Basis Function)**:
\[K(x, x') = e^{-\gamma ||x - x'||^2}\]

**Sigmoid Kernel**:
\[K(x, x') = \tanh(\alpha x \cdot x' + c)\]

### When to Use SVM

**Good for**:
- High-dimensional spaces
- Clear margin of separation
- More features than samples
- Non-linear decision boundaries (with kernels)

**Not ideal for**:
- Very large datasets (training time)
- Noisy data with overlapping classes
- Requires feature scaling
- Difficult to interpret

## Model Evaluation

### Confusion Matrix

```
                Predicted
              Positive  Negative
Actual  Pos     TP        FN
        Neg     FP        TN
```

### Metrics

**Accuracy**:
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Precision**: Of positive predictions, how many correct?
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Recall (Sensitivity)**: Of actual positives, how many found?
\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**F1-Score**: Harmonic mean of precision and recall
\[
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**Specificity**: Of actual negatives, how many correct?
\[
\text{Specificity} = \frac{TN}{TN + FP}
\]

### When to Use Which Metric

**Accuracy**: Balanced classes, equal error costs

**Precision**: False positives costly
- Spam detection (don't block legitimate emails)
- Medical treatment (don't treat healthy patients)

**Recall**: False negatives costly
- Disease screening (don't miss sick patients)
- Fraud detection (don't miss fraudulent transactions)

**F1-Score**: Balance precision and recall, imbalanced classes

**ROC-AUC**: Overall classifier performance, threshold-independent

### Imbalanced Data Strategies

1. **Resampling**:
   - Oversample minority class (SMOTE)
   - Undersample majority class
   - Combination (SMOTE + Tomek links)

2. **Class weights**: Penalize misclassification of minority class

3. **Appropriate metrics**: Use precision, recall, F1, not accuracy

4. **Anomaly detection**: Treat as one-class classification

5. **Ensemble methods**: Often handle imbalance well naturally

## Cross-Validation

### K-Fold Cross-Validation

1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Validate on remaining fold
3. Average K performance scores

**Stratified K-Fold**: Preserves class distribution in each fold

### Leave-One-Out (LOO)

- K = n (number of samples)
- Extreme case of k-fold
- Computationally expensive
- Low bias, high variance

### Time Series Cross-Validation

- Respects temporal order
- Train on past, test on future
- Expanding or rolling window

## Practical Workflow

### 1. Data Preparation
```python
# Load and split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (for SVM, logistic regression)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
```

### 3. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob)}')
```

## Hands-On Practice

### Associated Lab
- **[Lab 7: Classification](../labs/lab-07-classification/README.md)** - Comprehensive classification workflow

### Jupyter Notebooks
1. [Logistic Regression](notebooks/01-logistic-regression.ipynb) - Binary and multiclass classification
2. [Naïve Bayes Classifier](notebooks/02-naive-bayes-classifier.ipynb) - Text and categorical data
3. [Decision Trees](notebooks/03-decision-trees.ipynb) - Building and visualizing trees
4. [Ensemble Methods](notebooks/04-ensemble-methods.ipynb) - Random Forest and boosting
5. [Model Comparison](notebooks/05-model-comparison.ipynb) - Comparing multiple classifiers
6. [Cross-Validation](notebooks/06-cross-validation.ipynb) - Validation strategies

## Case Study: Customer Churn Prediction

### Business Problem
Telecommunications company wants to predict which customers will churn (cancel service) to enable proactive retention.

### Dataset Features
- Demographics: age, gender, location
- Account info: tenure, contract type, payment method
- Usage: call minutes, data usage, customer service contacts
- Billing: monthly charges, total charges

### Approach
1. **EDA**: Understand churn patterns
2. **Feature engineering**: Create interaction terms, aggregations
3. **Model selection**: Compare logistic regression, Random Forest, XGBoost
4. **Threshold tuning**: Balance precision/recall based on retention cost
5. **Deployment**: Score customers monthly, target high-risk for retention offers

### Results
- **Random Forest**: Best overall (AUC = 0.87)
- **XGBoost**: Slightly faster inference (AUC = 0.86)
- **Logistic Regression**: Most interpretable (AUC = 0.82)

### Business Impact
- Identified 75% of churners before leaving
- Reduced churn rate by 12%
- ROI: $2.5M annually from retention campaigns

## Common Pitfalls

❌ Using accuracy for imbalanced datasets
❌ Not scaling features for distance-based algorithms
❌ Data leakage (using test data in training)
❌ Ignoring class imbalance
❌ Overfitting to training data
❌ Not validating assumptions (e.g., independence for Naïve Bayes)
❌ Comparing models without consistent preprocessing
❌ Forgetting to encode categorical variables
❌ Not handling missing values properly
❌ Choosing model based on training performance only

## Additional Resources

### Required Reading
- Textbook Chapter 7: "Advanced Analytical Theory and Methods: Classification"
- EMC Education Services, pp. 191-231

### Recommended Reading
- "An Introduction to Statistical Learning" Chapters 4-5
- "The Elements of Statistical Learning" Chapters 4, 9, 10
- "Hands-On Machine Learning" Chapters 3-7

### Videos
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

### Online Resources
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html)
- [ROC Curve Interactive Demo](https://arogozhnikov.github.io/2015/10/05/roc-curve.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## Summary

Classification assigns categorical labels to observations using supervised learning. Major algorithms include:

- **Logistic Regression**: Probabilistic, interpretable, works for linearly separable data
- **Naïve Bayes**: Fast, works well with high dimensions and limited data
- **Decision Trees**: Interpretable, handles non-linear relationships, prone to overfitting
- **Random Forest**: Robust ensemble method, reduces overfitting
- **Gradient Boosting**: State-of-the-art performance, requires careful tuning
- **SVM**: Powerful for high-dimensional data, kernel trick for non-linearity

Success in classification requires:
- Proper data preprocessing (scaling, encoding)
- Appropriate algorithm selection
- Careful hyperparameter tuning
- Relevant evaluation metrics for business context
- Validation strategy (cross-validation)
- Handling imbalanced data when present

## Next Steps

1. Work through all six classification notebooks
2. Complete [Lab 7: Classification](../labs/lab-07-classification/README.md)
3. Practice with Kaggle classification competitions
4. Apply to your domain-specific problems
5. Move on to [Chapter 8: Time Series Analysis](../08-time-series/index.md)

---

**Quiz 3** (covering Chapters 7-9) will be available in Week 11.