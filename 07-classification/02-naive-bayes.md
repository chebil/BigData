# Naïve Bayes Classifier

## Introduction

Naïve Bayes is a probabilistic classification algorithm based on Bayes' theorem with an assumption of conditional independence between features. Despite its "naïve" assumption that all features are independent, it often performs surprisingly well in practice, especially for text classification, spam filtering, and sentiment analysis.

## Bayes' Theorem

### Fundamental Formula

Bayes' theorem describes the probability of an event based on prior knowledge:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

For classification with class \(C\) and features \(X\):

\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

**Components**:
- \(P(C|X)\): **Posterior probability** - what we want to find
- \(P(X|C)\): **Likelihood** - probability of features given class
- \(P(C)\): **Prior probability** - probability of class
- \(P(X)\): **Evidence** - probability of observing features

### Example: Medical Diagnosis

**Problem**: Patient tests positive for disease. What's probability they have it?

**Given**:
- \(P(\text{Disease}) = 0.01\) (1% prevalence)
- \(P(\text{Positive}|\text{Disease}) = 0.95\) (95% sensitivity)
- \(P(\text{Positive}|\text{No Disease}) = 0.05\) (5% false positive rate)

**Calculate**:

\[
P(\text{Positive}) = P(\text{Positive}|\text{Disease}) \cdot P(\text{Disease}) + P(\text{Positive}|\text{No Disease}) \cdot P(\text{No Disease})
\]
\[
= 0.95 \times 0.01 + 0.05 \times 0.99 = 0.059
\]

\[
P(\text{Disease}|\text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.16
\]

**Result**: Only 16% chance of having disease despite positive test!

This demonstrates the importance of considering base rates (prior probability).

## Naïve Bayes Classification

### The Naïve Assumption

For features \(X = (x_1, x_2, \ldots, x_n)\), naïve Bayes assumes **conditional independence**:

\[
P(X|C) = P(x_1, x_2, \ldots, x_n|C) = P(x_1|C) \cdot P(x_2|C) \cdots P(x_n|C) = \prod_{i=1}^{n} P(x_i|C)
\]

This is "naïve" because features are rarely truly independent in reality.

### Classification Rule

Predict class \(\hat{C}\) that maximizes posterior probability:

\[
\hat{C} = \arg\max_C P(C|X) = \arg\max_C \frac{P(X|C) \cdot P(C)}{P(X)}
\]

Since \(P(X)\) is constant for all classes:

\[
\hat{C} = \arg\max_C P(C) \prod_{i=1}^{n} P(x_i|C)
\]

### Log Probabilities

To avoid numerical underflow (probabilities become very small):

\[
\hat{C} = \arg\max_C \left[\log P(C) + \sum_{i=1}^{n} \log P(x_i|C)\right]
\]

## Training Naïve Bayes

### Step 1: Calculate Prior Probabilities

For each class \(c_k\):

\[
P(C = c_k) = \frac{\text{Number of instances in class } c_k}{\text{Total number of instances}}
\]

### Step 2: Calculate Likelihoods

For each feature \(x_i\) and class \(c_k\):

\[
P(x_i = v|C = c_k) = \frac{\text{Count of } x_i = v \text{ in class } c_k}{\text{Total count in class } c_k}
\]

### Example: Email Classification

**Training Data**:

| Email | Contains "free" | Contains "win" | Contains "money" | Class |
|-------|-----------------|----------------|------------------|-------|
| 1 | Yes | Yes | Yes | Spam |
| 2 | No | No | Yes | Spam |
| 3 | Yes | Yes | No | Spam |
| 4 | No | No | No | Ham |
| 5 | No | No | No | Ham |
| 6 | No | No | Yes | Ham |

**Calculate Priors**:
- \(P(\text{Spam}) = 3/6 = 0.5\)
- \(P(\text{Ham}) = 3/6 = 0.5\)

**Calculate Likelihoods**:

| Feature | P(feature \| Spam) | P(feature \| Ham) |
|---------|-------------------|------------------|
| free=Yes | 2/3 ≈ 0.67 | 0/3 = 0.00 |
| free=No | 1/3 ≈ 0.33 | 3/3 = 1.00 |
| win=Yes | 2/3 ≈ 0.67 | 0/3 = 0.00 |
| win=No | 1/3 ≈ 0.33 | 3/3 = 1.00 |
| money=Yes | 2/3 ≈ 0.67 | 1/3 ≈ 0.33 |
| money=No | 1/3 ≈ 0.33 | 2/3 ≈ 0.67 |

**Classify New Email**: Contains "free" but not "win" or "money"

\[
P(\text{Spam}|X) \propto P(\text{Spam}) \cdot P(\text{free}|\text{Spam}) \cdot P(\neg\text{win}|\text{Spam}) \cdot P(\neg\text{money}|\text{Spam})
\]
\[
= 0.5 \times 0.67 \times 0.33 \times 0.33 \approx 0.037
\]

\[
P(\text{Ham}|X) \propto P(\text{Ham}) \cdot P(\text{free}|\text{Ham}) \cdot P(\neg\text{win}|\text{Ham}) \cdot P(\neg\text{money}|\text{Ham})
\]
\[
= 0.5 \times 0.00 \times 1.00 \times 0.67 = 0
\]

**Prediction**: Spam (higher score)

**Problem**: Zero probability for Ham! This is where Laplace smoothing helps.

## Laplace Smoothing

### The Zero Probability Problem

If a feature value never appears with a class in training data, its probability is 0, making the entire product 0.

### Solution: Add-One (Laplace) Smoothing

Add pseudocount \(\alpha\) (typically 1) to each count:

\[
P(x_i = v|C = c_k) = \frac{\text{count}(x_i = v, C = c_k) + \alpha}{\text{count}(C = c_k) + \alpha \cdot |V|}
\]

Where \(|V|\) is the number of possible values for feature \(x_i\).

### Example with Smoothing (\(\alpha = 1\))

Revisit email example with smoothing:

\[
P(\text{free}|\text{Ham}) = \frac{0 + 1}{3 + 1 \times 2} = \frac{1}{5} = 0.2
\]

Now:

\[
P(\text{Ham}|X) \propto 0.5 \times 0.2 \times 1.0 \times 0.67 \approx 0.067
\]

Still predicts Spam, but Ham has non-zero probability.

## Variants of Naïve Bayes

### 1. Gaussian Naïve Bayes

**For continuous features** assuming normal distribution:

\[
P(x_i|C = c_k) = \frac{1}{\sqrt{2\pi\sigma_{c_k}^2}} \exp\left(-\frac{(x_i - \mu_{c_k})^2}{2\sigma_{c_k}^2}\right)
\]

Where:
- \(\mu_{c_k}\): Mean of feature \(x_i\) for class \(c_k\)
- \(\sigma_{c_k}^2\): Variance of feature \(x_i\) for class \(c_k\)

**Use Case**: Iris classification, medical diagnosis with continuous measurements

**Python Example**:
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 2. Multinomial Naïve Bayes

**For discrete counts** (e.g., word counts in documents):

\[
P(x_i|C = c_k) = \frac{N_{c_k,i} + \alpha}{N_{c_k} + \alpha \cdot |V|}
\]

Where:
- \(N_{c_k,i}\): Count of feature \(i\) in class \(c_k\)
- \(N_{c_k}\): Total count of all features in class \(c_k\)
- \(|V|\): Vocabulary size

**Use Case**: Text classification, document categorization, spam detection

**Python Example**:
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_counts, y_train)
```

### 3. Bernoulli Naïve Bayes

**For binary features** (presence/absence):

\[
P(x_i|C = c_k) = P(i|c_k) \cdot x_i + (1 - P(i|c_k)) \cdot (1 - x_i)
\]

Where \(x_i \in \{0, 1\}\).

**Use Case**: Text classification with binary features (word present or not)

**Python Example**:
```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
X_train_binary = vectorizer.fit_transform(X_train)

model = BernoulliNB(alpha=1.0)
model.fit(X_train_binary, y_train)
```

### 4. Complement Naïve Bayes

**For imbalanced datasets**: Estimates parameters from complement of each class.

\[
P(x_i|C = \neg c_k) = \frac{\sum_{c \neq c_k} N_{c,i} + \alpha}{\sum_{c \neq c_k} N_c + \alpha \cdot |V|}
\]

**Use Case**: Text classification with imbalanced classes

**Python Example**:
```python
from sklearn.naive_bayes import ComplementNB

model = ComplementNB(alpha=1.0)
model.fit(X_train_counts, y_train)
```

## Implementation in Python

### Complete Text Classification Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('emails.csv')
X = df['text']
y = df['label']  # spam or ham

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train model
model = MultinomialNB(alpha=1.0)
model.fit(X_train_counts, y_train)

# Predictions
y_pred = model.predict(X_test_counts)
y_prob = model.predict_proba(X_test_counts)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Most informative features
feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_

# For spam class (assuming index 1)
top_spam_indices = np.argsort(log_probs[1])[-20:]
print("\nTop spam indicators:")
for idx in top_spam_indices[::-1]:
    print(f"{feature_names[idx]}: {np.exp(log_probs[1][idx]):.4f}")
```

### With TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF instead of counts
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Define parameter grid
param_grid = {
    'vectorizer__max_features': [500, 1000, 2000],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__min_df': [1, 2, 5],
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
}

# Grid search
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.3f}")
```

## Advantages

✅ **Fast training and prediction**: Linear time complexity
✅ **Works well with high dimensions**: Text data with thousands of features
✅ **Requires small training set**: Few examples per class can work
✅ **Handles missing values**: Naturally (just don't include in calculation)
✅ **Interpretable**: Can analyze feature probabilities
✅ **Probabilistic predictions**: Outputs class probabilities
✅ **Multiple variants**: For different data types
✅ **Good baseline**: Simple to implement and understand

## Limitations

❌ **Independence assumption**: Rarely true in practice
❌ **Zero-frequency problem**: Requires smoothing
❌ **Probability estimates**: Not well-calibrated
❌ **Correlated features**: Double-counts their effects
❌ **Continuous features**: Gaussian assumption may not hold
❌ **Cannot learn interactions**: Between features
❌ **Outperformed by**: More sophisticated models on complex tasks

## When to Use Naïve Bayes

### Good Choice When:

1. **Text classification**: Especially with bag-of-words
2. **Real-time prediction**: Need very fast inference
3. **Small training set**: Limited labeled data
4. **High-dimensional data**: Many features
5. **Baseline model**: Quick first model
6. **Interpretability**: Need to understand feature importance

### Consider Alternatives When:

1. **Features are correlated**: Use logistic regression or SVM
2. **Need feature interactions**: Use decision trees or neural networks
3. **Complex relationships**: Use ensemble methods
4. **Need calibrated probabilities**: Use logistic regression or calibration
5. **Non-Gaussian continuous features**: Use more flexible models

## Real-World Applications

### 1. Spam Filtering

```python
# Example: Email spam detection
features = [
    'contains_free', 'contains_win', 'contains_money',
    'contains_urgent', 'num_exclamation', 'num_caps_words'
]

model = BernoulliNB()
model.fit(X_train[features], y_train)
```

### 2. Sentiment Analysis

```python
# Example: Movie review sentiment
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(reviews_train)

model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, sentiment_train)
```

### 3. Document Categorization

```python
# Example: News article classification
categories = ['sports', 'politics', 'technology', 'entertainment']

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_vectorized = vectorizer.fit_transform(articles)

model = MultinomialNB()
model.fit(X_vectorized, categories)
```

### 4. Medical Diagnosis

```python
# Example: Disease prediction from symptoms
symptoms = ['fever', 'cough', 'fatigue', 'headache', 'body_ache']

model = BernoulliNB()
model.fit(X_train[symptoms], diagnoses_train)
```

## Comparison with Other Classifiers

| Aspect | Naïve Bayes | Logistic Regression | Decision Tree |
|--------|--------------|---------------------|---------------|
| **Speed** | Very fast | Fast | Moderate |
| **Scalability** | Excellent | Good | Poor |
| **Interpretability** | Good | Excellent | Excellent |
| **Feature interactions** | No | No | Yes |
| **Probability calibration** | Poor | Good | Poor |
| **High dimensions** | Excellent | Good | Poor |
| **Correlated features** | Poor | Moderate | Good |
| **Text classification** | Excellent | Good | Poor |

## Tips and Best Practices

1. **Choose the right variant**:
   - Gaussian for continuous features
   - Multinomial for counts
   - Bernoulli for binary features

2. **Tune smoothing parameter**: Try \(\alpha \in [0.1, 0.5, 1.0, 2.0]\)

3. **Feature selection**: Remove highly correlated features

4. **Text preprocessing**:
   - Remove stop words
   - Lowercase text
   - Consider stemming/lemmatization
   - Try n-grams (1-2 or 1-3)

5. **Handle imbalanced data**: Consider ComplementNB

6. **Probability calibration**: Use CalibratedClassifierCV if needed

7. **Feature engineering**: Despite independence assumption, good features help

8. **Validate assumptions**: Check if independence roughly holds

## Summary

Naïve Bayes is a simple yet powerful probabilistic classifier that:

- Based on Bayes' theorem with independence assumption
- Calculates class probabilities from feature likelihoods
- Extremely fast to train and predict
- Works remarkably well for text classification
- Requires smoothing to handle unseen feature values
- Has multiple variants for different data types
- Provides interpretable feature importance
- Serves as excellent baseline model

Despite its simplifying assumptions, Naïve Bayes often competes with more sophisticated algorithms, especially for text and high-dimensional data.

## Further Reading

- "Machine Learning" by Tom Mitchell - Chapter 6
- "Pattern Recognition and Machine Learning" by Bishop - Section 8.2
- [Scikit-learn Naïve Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)

## Next Section

Continue to [Decision Trees](03-decision-trees.md) to learn about tree-based classification methods.