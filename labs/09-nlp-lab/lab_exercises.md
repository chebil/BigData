# Lab 9: Text Analytics - Sentiment Analysis

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
```

## Part 1: Text Preprocessing (25 points)

### Exercise 1.1: Load Data (5 points)

```python
# Sample movie reviews
reviews = [
    ("This movie was absolutely fantastic! I loved every minute.", "positive"),
    ("Terrible waste of time. Wouldn't recommend.", "negative"),
    ("One of the best films I've ever seen. Masterpiece!", "positive"),
    ("Boring and predictable. Very disappointed.", "negative"),
    ("Amazing performances and beautiful cinematography.", "positive"),
    ("Awful plot and terrible acting. Total disaster.", "negative"),
    ("Excellent story with great character development.", "positive"),
    ("Completely boring. Fell asleep halfway through.", "negative"),
    ("Brilliant! Highly recommended to everyone.", "positive"),
    ("Waste of money. One of the worst movies ever.", "negative")
] * 50  # Repeat for more data

texts, labels = zip(*reviews)
df = pd.DataFrame({'review': texts, 'sentiment': labels})

print(f"Dataset shape: {df.shape}")
print("\nSample reviews:")
print(df.head())
print("\nClass distribution:")
print(df['sentiment'].value_counts())
```

### Exercise 1.2: Text Cleaning (10 points)

**TODO:** Create a text cleaning pipeline

```python
class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean(self, text):
        # TODO: Implement cleaning steps
        # 1. Lowercase
        text = ??
        
        # 2. Remove special characters
        text = ??
        
        # 3. Tokenize
        tokens = ??
        
        # 4. Remove stop words
        tokens = ??
        
        # 5. Lemmatize
        tokens = ??
        
        return ' '.join(tokens)

cleaner = TextCleaner()

# Apply cleaning
df['cleaned_review'] = df['review'].apply(cleaner.clean)

print("\nOriginal vs Cleaned:")
for i in range(3):
    print(f"\nOriginal: {df.loc[i, 'review']}")
    print(f"Cleaned:  {df.loc[i, 'cleaned_review']}")
```

**Questions:**
- Q1: Why remove stop words?
- Q2: When might you want to keep punctuation?
- Q3: Lemmatization vs Stemming - which is better?

### Exercise 1.3: Text Statistics (10 points)

```python
from collections import Counter

# Calculate statistics
all_words = ' '.join(df['cleaned_review']).split()
word_freq = Counter(all_words)

print(f"Total words: {len(all_words)}")
print(f"Unique words: {len(word_freq)}")
print(f"\nTop 20 words:")
for word, count in word_freq.most_common(20):
    print(f"  {word}: {count}")

# Visualize
plt.figure(figsize=(12, 6))
top_words = dict(word_freq.most_common(20))
plt.barh(list(top_words.keys()), list(top_words.values()))
plt.xlabel('Frequency')
plt.title('Top 20 Words')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Questions:**
- Q4: What are the most common words?
- Q5: Do they make sense for sentiment analysis?

---

## Part 2: Feature Extraction (20 points)

### Exercise 2.1: TF-IDF Vectorization (15 points)

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment'], 
    test_size=0.2, random_state=42, stratify=df['sentiment']
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# TODO: Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=100, min_df=2)

# Fit and transform
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Feature names (first 20): {tfidf.get_feature_names_out()[:20]}")

# Visualize TF-IDF scores
feature_names = tfidf.get_feature_names_out()
tfidf_scores = X_train_tfidf.mean(axis=0).A1

top_indices = tfidf_scores.argsort()[-20:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_scores = [tfidf_scores[i] for i in top_indices]

plt.figure(figsize=(12, 6))
plt.barh(top_features, top_scores)
plt.xlabel('Mean TF-IDF Score')
plt.title('Top 20 Features by TF-IDF')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Questions:**
- Q6: What are the most important features?
- Q7: How does TF-IDF differ from Bag of Words?

### Exercise 2.2: N-grams (5 points)

```python
# TODO: Try bigrams
tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
X_train_bigram = tfidf_bigram.fit_transform(X_train)

print(f"Bigram features: {tfidf_bigram.get_feature_names_out()[:30]}")
```

**Questions:**
- Q8: What bigrams are captured?
- Q9: Do bigrams improve sentiment detection?

---

## Part 3: Sentiment Classification (30 points)

### Exercise 3.1: Logistic Regression (10 points)

```python
# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test_tfidf)

# Evaluate
print("Logistic Regression Results:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.show()
```

**Questions:**
- Q10: What is the accuracy?
- Q11: Which class has better precision?
- Q12: Are there more false positives or false negatives?

### Exercise 3.2: Naive Bayes (10 points)

```python
# TODO: Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes Results:")
print(classification_report(y_test, y_pred_nb))

# Compare models
from sklearn.metrics import accuracy_score, f1_score

print("\nModel Comparison:")
print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
print("-" * 40)
print(f"{'Logistic Regression':<20} {accuracy_score(y_test, y_pred_lr):<10.4f} {f1_score(y_test, y_pred_lr, pos_label='positive'):<10.4f}")
print(f"{'Naive Bayes':<20} {accuracy_score(y_test, y_pred_nb):<10.4f} {f1_score(y_test, y_pred_nb, pos_label='positive'):<10.4f}")
```

**Questions:**
- Q13: Which model performs better?
- Q14: Why might Naive Bayes work well for text?

### Exercise 3.3: Test on New Reviews (10 points)

```python
new_reviews = [
    "This was an absolutely amazing movie!",
    "Terrible and boring. Complete waste of time.",
    "Not bad, but could have been better.",
    "Best film of the year! Loved it!"
]

# TODO: Predict sentiment for new reviews
# 1. Clean
cleaned_new = [cleaner.clean(review) for review in new_reviews]

# 2. Vectorize
X_new = tfidf.transform(cleaned_new)

# 3. Predict
predictions_lr = lr_model.predict(X_new)
predictions_nb = nb_model.predict(X_new)
proba_lr = lr_model.predict_proba(X_new)

print("\nPredictions on New Reviews:")
for i, review in enumerate(new_reviews):
    print(f"\nReview: {review}")
    print(f"  LR Prediction: {predictions_lr[i]} (confidence: {proba_lr[i].max():.2%})")
    print(f"  NB Prediction: {predictions_nb[i]}")
```

**Questions:**
- Q15: Are the predictions correct?
- Q16: Which model is more confident?

---

## Part 4: Advanced Analysis (25 points)

### Exercise 4.1: Feature Importance (10 points)

```python
# Get most important features for each class
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = lr_model.coef_[0]

# Top positive indicators
top_positive_idx = coefficients.argsort()[-20:][::-1]
top_positive = feature_names[top_positive_idx]
top_positive_coef = coefficients[top_positive_idx]

# Top negative indicators
top_negative_idx = coefficients.argsort()[:20]
top_negative = feature_names[top_negative_idx]
top_negative_coef = coefficients[top_negative_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(top_positive, top_positive_coef, color='green', alpha=0.7)
axes[0].set_xlabel('Coefficient')
axes[0].set_title('Top Positive Sentiment Indicators')
axes[0].invert_yaxis()

axes[1].barh(top_negative, top_negative_coef, color='red', alpha=0.7)
axes[1].set_xlabel('Coefficient')
axes[1].set_title('Top Negative Sentiment Indicators')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

**Questions:**
- Q17: What words are most positive?
- Q18: What words are most negative?
- Q19: Do the features make intuitive sense?

### Exercise 4.2: Error Analysis (15 points)

```python
# Find misclassified examples
misclassified_mask = y_pred_lr != y_test
misclassified = X_test[misclassified_mask]
true_labels = y_test[misclassified_mask]
predicted_labels = y_pred_lr[misclassified_mask]

print(f"\nMisclassified: {misclassified_mask.sum()} out of {len(y_test)}")
print(f"Error rate: {misclassified_mask.sum() / len(y_test):.2%}")

# Show examples
print("\nMisclassified Examples:")
for i in range(min(5, len(misclassified))):
    print(f"\nReview: {misclassified.iloc[i]}")
    print(f"  True: {true_labels.iloc[i]}")
    print(f"  Predicted: {predicted_labels[i]}")
```

**Questions:**
- Q20: Why were these reviews misclassified?
- Q21: Are there patterns in the errors?
- Q22: How would you improve the model?

---

## Part 5: Production Pipeline (Bonus 10 points)

**TODO:** Create production-ready sentiment analyzer

```python
import joblib

class SentimentAnalyzer:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.vectorizer = None
        self.model = None
    
    def train(self, texts, labels):
        # Clean
        cleaned = [self.cleaner.clean(text) for text in texts]
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(cleaned)
        
        # Train
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)
        
        return self
    
    def predict(self, texts):
        cleaned = [self.cleaner.clean(text) for text in texts]
        X = self.vectorizer.transform(cleaned)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save(self, filename):
        joblib.dump(self, filename)
    
    @staticmethod
    def load(filename):
        return joblib.load(filename)

# Train and save
analyzer = SentimentAnalyzer()
analyzer.train(df['review'], df['sentiment'])
analyzer.save('sentiment_analyzer.pkl')

print("Model saved!")

# Load and test
loaded_analyzer = SentimentAnalyzer.load('sentiment_analyzer.pkl')
test_reviews = ["Great movie!", "Terrible experience"]
preds, probs = loaded_analyzer.predict(test_reviews)

print("\nProduction predictions:")
for review, pred, prob in zip(test_reviews, preds, probs):
    print(f"Review: {review}")
    print(f"  Sentiment: {pred} (confidence: {prob.max():.2%})")
```

**Final Questions:**
- Q23: How would you deploy this in production?
- Q24: How would you monitor performance?
- Q25: What improvements would you make?

Good luck! 🎯
