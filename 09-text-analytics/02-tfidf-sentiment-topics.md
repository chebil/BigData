# TF-IDF, Sentiment Analysis, and Topic Modeling

## Learning Objectives

- Understand TF-IDF vectorization
- Perform sentiment analysis
- Apply topic modeling with LDA
- Build text classification models
- Create practical NLP applications
- Deploy production NLP pipelines

## TF-IDF Vectorization

### Bag of Words (BoW)

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Sample documents
documents = [
    "machine learning is great",
    "deep learning is amazing",
    "machine learning and deep learning",
    "natural language processing"
]

print("BAG OF WORDS (BoW)")
print("="*70)

# Create BoW
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(documents)

# Feature names
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary: {feature_names}")
print(f"Vocabulary size: {len(feature_names)}")

# Convert to DataFrame
bow_df = pd.DataFrame(X_bow.toarray(), columns=feature_names)
print("\nBag of Words Matrix:")
print(bow_df)

print("""
\nBoW CHARACTERISTICS:
  ✓ Simple and intuitive
  ✓ Preserves word frequency
  ✗ No word order information
  ✗ Doesn't capture importance
  ✗ Common words dominate
""")
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Formula**:

\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
\]

Where:
- \(\text{TF}(t, d)\) = Frequency of term \(t\) in document \(d\)
- \(\text{IDF}(t) = \log\frac{N}{\text{df}(t)}\)
- \(N\) = Total number of documents
- \(\text{df}(t)\) = Number of documents containing term \(t\)

```python
print("\nTF-IDF VECTORIZATION")
print("="*70)

# Create TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# Convert to DataFrame
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Matrix:")
print(tfidf_df.round(3))

# Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# BoW heatmap
sns.heatmap(bow_df, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Bag of Words', fontsize=14)
axes[0].set_xlabel('Words')
axes[0].set_ylabel('Documents')

# TF-IDF heatmap
sns.heatmap(tfidf_df, annot=True, fmt='.2f', cmap='Reds', ax=axes[1], cbar_kws={'label': 'TF-IDF'})
axes[1].set_title('TF-IDF', fontsize=14)
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Documents')

plt.tight_layout()
plt.show()

print("""
TF-IDF CHARACTERISTICS:
  ✓ Reduces weight of common words
  ✓ Highlights distinctive words
  ✓ Better for classification
  ✓ Widely used standard
  ✗ Still no word order
  ✗ Sparse representations
""")
```

### Advanced TF-IDF Options

```python
# Advanced TF-IDF
tfidf_advanced = TfidfVectorizer(
    max_features=100,        # Limit vocabulary
    min_df=2,                # Minimum document frequency
    max_df=0.8,              # Maximum document frequency (remove too common)
    ngram_range=(1, 2),      # Unigrams and bigrams
    stop_words='english',    # Remove stop words
    sublinear_tf=True        # Use log(TF) instead of TF
)

large_docs = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for complex tasks",
    "Natural language processing enables text understanding",
    "Computer vision processes visual information",
    "Reinforcement learning learns through trial and error"
] * 10  # Repeat for min_df

X_advanced = tfidf_advanced.fit_transform(large_docs)

print("\nAdvanced TF-IDF Configuration:")
print(f"Vocabulary size: {len(tfidf_advanced.get_feature_names_out())}")
print(f"Matrix shape: {X_advanced.shape}")
print(f"Sparsity: {1 - (X_advanced.nnz / (X_advanced.shape[0] * X_advanced.shape[1])):.2%}")

# Top features
feature_names = tfidf_advanced.get_feature_names_out()
mean_tfidf = X_advanced.mean(axis=0).A1
top_features = sorted(zip(feature_names, mean_tfidf), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 Features by Mean TF-IDF:")
for feature, score in top_features:
    print(f"  {feature}: {score:.4f}")
```

## Sentiment Analysis

### Rule-Based Sentiment (TextBlob)

```python
# Install: pip install textblob
from textblob import TextBlob

reviews = [
    "This product is amazing! I love it so much.",
    "Terrible experience. Waste of money.",
    "It's okay, nothing special.",
    "Absolutely fantastic! Highly recommended!",
    "Disappointing quality. Would not buy again."
]

print("\nRULE-BASED SENTIMENT ANALYSIS (TextBlob)")
print("="*70)

sentiments = []
for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity  # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    sentiments.append({
        'review': review,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment
    })

sentiment_df = pd.DataFrame(sentiments)
print(sentiment_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Polarity distribution
axes[0].barh(range(len(sentiment_df)), sentiment_df['polarity'])
axes[0].set_yticks(range(len(sentiment_df)))
axes[0].set_yticklabels([f"Review {i+1}" for i in range(len(sentiment_df))])
axes[0].axvline(0, color='black', linestyle='--')
axes[0].set_xlabel('Polarity')
axes[0].set_title('Sentiment Polarity')
axes[0].grid(alpha=0.3)

# Sentiment counts
sentiment_counts = sentiment_df['sentiment'].value_counts()
axes[1].bar(sentiment_counts.index, sentiment_counts.values)
axes[1].set_ylabel('Count')
axes[1].set_title('Sentiment Distribution')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### ML-Based Sentiment Analysis

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Sample labeled data
reviews_labeled = [
    ("This is excellent! Love it", "positive"),
    ("Terrible product, very disappointed", "negative"),
    ("Amazing quality, highly recommend", "positive"),
    ("Waste of money, don't buy", "negative"),
    ("Pretty good, satisfied", "positive"),
    ("Poor quality, not worth it", "negative"),
    ("Outstanding! Best purchase ever", "positive"),
    ("Horrible experience, terrible", "negative"),
    ("Great value for money", "positive"),
    ("Completely useless", "negative")
] * 20  # Repeat for more data

texts, labels = zip(*reviews_labeled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print("\nML-BASED SENTIMENT ANALYSIS")
print("="*70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Sentiment Analysis Confusion Matrix')
plt.tight_layout()
plt.show()

# Test on new reviews
new_reviews = [
    "This is absolutely wonderful!",
    "I hate this, complete waste",
    "Pretty decent product"
]

print("\nPredictions on New Reviews:")
for review in new_reviews:
    pred = pipeline.predict([review])[0]
    proba = pipeline.predict_proba([review])[0]
    print(f"Review: {review}")
    print(f"  Prediction: {pred}")
    print(f"  Confidence: {proba.max():.2%}\n")
```

## Topic Modeling with LDA

**Latent Dirichlet Allocation (LDA)**: Discovers topics in document collection

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample documents (simulating different topics)
documents = [
    # Topic 1: Machine Learning
    "machine learning algorithms predict outcomes using data",
    "supervised learning requires labeled training data",
    "neural networks learn complex patterns",
    "deep learning models process hierarchical features",
    
    # Topic 2: Data Science
    "data analysis reveals insights from datasets",
    "statistical methods analyze data distributions",
    "visualization helps understand data patterns",
    "exploratory data analysis discovers relationships",
    
    # Topic 3: Programming
    "python programming language versatile and powerful",
    "software development requires good coding practices",
    "debugging code finds and fixes errors",
    "version control manages code changes"
] * 5  # Repeat for better topic modeling

print("\nTOPIC MODELING WITH LDA")
print("="*70)

# Create document-term matrix
vectorizer = CountVectorizer(max_features=50, stop_words='english')
X = vectorizer.fit_transform(documents)

print(f"Documents: {len(documents)}")
print(f"Vocabulary: {len(vectorizer.get_feature_names_out())}")

# LDA model
n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)

lda.fit(X)

# Display topics
feature_names = vectorizer.get_feature_names_out()

print(f"\nDiscovered Topics (top 10 words each):")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# Document-topic distribution
doc_topics = lda.transform(X)

print(f"\nDocument-Topic Distribution (first 5 documents):")
for i in range(min(5, len(documents))):
    topic_dist = doc_topics[i]
    dominant_topic = topic_dist.argmax()
    print(f"Doc {i+1}: Topic {dominant_topic+1} ({topic_dist[dominant_topic]:.2%})")
    print(f"  Full distribution: {topic_dist}")

# Visualize topic-word matrix
plt.figure(figsize=(12, 8))
top_words_per_topic = 10
topic_words = np.zeros((n_topics, top_words_per_topic))

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-top_words_per_topic:][::-1]
    topic_words[topic_idx] = topic[top_words_idx]

sns.heatmap(topic_words, annot=False, cmap='YlOrRd',
            xticklabels=[feature_names[i] for i in lda.components_[0].argsort()[-top_words_per_topic:][::-1]],
            yticklabels=[f'Topic {i+1}' for i in range(n_topics)])
plt.title('Topic-Word Distribution (LDA)', fontsize=14)
plt.xlabel('Top Words')
plt.ylabel('Topics')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## Text Classification Application

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (subset for speed)
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']

print("\nTEXT CLASSIFICATION: 20 Newsgroups Dataset")
print("="*70)

# Load data
train_data = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'),
                                random_state=42)

test_data = fetch_20newsgroups(subset='test', categories=categories,
                              remove=('headers', 'footers', 'quotes'),
                              random_state=42)

print(f"Training samples: {len(train_data.data)}")
print(f"Test samples: {len(test_data.data)}")
print(f"Categories: {train_data.target_names}")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

y_train = train_data.target
y_test = test_data.target

# Train Naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)

print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nPer-class Performance:")
print(classification_report(y_test, y_pred_nb, target_names=train_data.target_names))

# Example predictions
test_samples = [
    "NASA launched a new spacecraft to Mars",
    "The baseball team won the championship",
    "I need help with 3D graphics rendering",
    "The election results are being disputed"
]

print("\nExample Predictions:")
for sample in test_samples:
    X_sample = vectorizer.transform([sample])
    pred = nb_clf.predict(X_sample)[0]
    pred_proba = nb_clf.predict_proba(X_sample)[0]
    category = train_data.target_names[pred]
    confidence = pred_proba.max()
    
    print(f"\nText: {sample}")
    print(f"  Category: {category}")
    print(f"  Confidence: {confidence:.2%}")
```

## Production NLP Pipeline

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom text preprocessor"""
    def __init__(self, lowercase=True, remove_punctuation=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import re
        
        def clean_text(text):
            if self.lowercase:
                text = text.lower()
            if self.remove_punctuation:
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return text
        
        return [clean_text(text) for text in X]

class NLPPipeline:
    def __init__(self, model_type='sentiment'):
        self.model_type = model_type
        self.pipeline = None
    
    def build_pipeline(self, classifier=None):
        """Build complete NLP pipeline"""
        if classifier is None:
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        
        self.pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', TfidfVectorizer(max_features=500, stop_words='english')),
            ('classifier', classifier)
        ])
        
        return self
    
    def train(self, X_train, y_train):
        """Train the pipeline"""
        if self.pipeline is None:
            self.build_pipeline()
        
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.pipeline.predict_proba(X)
    
    def save(self, filename):
        """Save pipeline"""
        joblib.dump(self.pipeline, filename)
        print(f"Pipeline saved to {filename}")
    
    @staticmethod
    def load(filename):
        """Load pipeline"""
        pipeline = joblib.load(filename)
        nlp = NLPPipeline()
        nlp.pipeline = pipeline
        return nlp

# Example usage
print("\n" + "="*70)
print("PRODUCTION NLP PIPELINE")
print("="*70)

# Create and train pipeline
nlp_pipeline = NLPPipeline()
nlp_pipeline.build_pipeline()
nlp_pipeline.train(X_train[:1000], y_train[:1000])  # Use subset for demo

# Predict
predictions = nlp_pipeline.predict(test_samples)
print("\nPipeline Predictions:")
for sample, pred in zip(test_samples, predictions):
    category = train_data.target_names[pred]
    print(f"Text: {sample}")
    print(f"  Predicted: {category}\n")

# Save
nlp_pipeline.save('nlp_pipeline.pkl')

print("""
PRODUCTION DEPLOYMENT CHECKLIST:

1. PIPELINE COMPONENTS:
   ✓ Preprocessing
   ✓ Vectorization
   ✓ Model

2. SERIALIZATION:
   ✓ Save entire pipeline
   ✓ Version control
   ✓ Dependency management

3. MONITORING:
   ✓ Prediction logging
   ✓ Performance metrics
   ✓ Data drift detection

4. MAINTENANCE:
   ✓ Regular retraining
   ✓ A/B testing
   ✓ Feedback loop
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **TF-IDF** captures word importance across documents
2. **Sentiment analysis** classifies text polarity
3. **TextBlob** provides rule-based sentiment
4. **ML models** offer better accuracy for sentiment
5. **LDA** discovers latent topics in documents
6. **Topic modeling** unsupervised pattern discovery
7. **Text classification** assigns categories to documents
8. **Pipelines** ensure reproducibility
9. **Feature engineering** critical for performance
10. **Production deployment** requires careful planning
:::

## Further Reading

- Jurafsky, D. & Martin, J.H. (2023). "Speech and Language Processing"
- Aggarwal, C.C. & Zhai, C. (2012). "Mining Text Data"
- Scikit-learn: [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- NLTK Book: [Natural Language Processing with Python](https://www.nltk.org/book/)
