# Chapter 9: Text Analytics and Natural Language Processing

## Introduction

Text analytics (also called text mining or natural language processing) extracts meaningful insights from unstructured text data. With 80% of business data being unstructured, text analytics has become essential for modern data science.

**Applications**:
- Sentiment analysis of customer reviews
- Topic modeling for content organization
- Chatbots and virtual assistants
- Document classification
- Information extraction
- Machine translation
- Text summarization

## Learning Objectives

By the end of this chapter, you will:

1. **Understand text preprocessing**: Tokenization, stopword removal, stemming, lemmatization
2. **Master text representation**: Bag-of-words, TF-IDF, word embeddings
3. **Perform sentiment analysis**: Lexicon-based and machine learning approaches
4. **Apply topic modeling**: LDA and NMF for discovering themes
5. **Build text classifiers**: Naïve Bayes, SVM, and deep learning for text
6. **Extract named entities**: NER for identifying people, organizations, locations
7. **Analyze word relationships**: N-grams, collocations, word clouds
8. **Implement in Python**: NLTK, spaCy, scikit-learn, Gensim

## Text Analytics Pipeline

```
[Raw Text] → [Preprocessing] → [Feature Extraction] → [Analysis/Modeling] → [Insights]
```

### Pipeline Stages

1. **Data Collection**: Web scraping, APIs, databases
2. **Preprocessing**: Clean, normalize, tokenize
3. **Feature Engineering**: Convert text to numerical features
4. **Modeling**: Apply ML/DL algorithms
5. **Evaluation**: Assess model performance
6. **Deployment**: Integrate into applications

## Quick Start Example

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Sample movie reviews dataset
reviews = [
    ("This movie was absolutely fantastic! Loved every minute.", "positive"),
    ("Terrible film, waste of time and money.", "negative"),
    ("An amazing masterpiece with great acting.", "positive"),
    ("Boring and predictable plot.", "negative"),
    ("Best movie I've seen this year!", "positive"),
    ("Awful movie, don't bother watching it.", "negative"),
    ("Brilliant cinematography and storytelling.", "positive"),
    ("Disappointing and poorly executed.", "negative")
]

df = pd.DataFrame(reviews, columns=['text', 'sentiment'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.25, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)

# Evaluate
print(classification_report(y_test, y_pred))

# Predict new review
new_review = ["This film was incredible and moving"]
new_tfidf = vectorizer.transform(new_review)
prediction = clf.predict(new_tfidf)
print(f"\nNew review sentiment: {prediction[0]}")
```

## Chapter Structure

This chapter covers:

1. **[Text Preprocessing](01-text-preprocessing.md)**: Cleaning and normalization
2. **[Feature Extraction](02-feature-extraction.md)**: BOW, TF-IDF, embeddings
3. **[Sentiment Analysis](03-sentiment-analysis.md)**: Opinion mining
4. **[Text Classification](04-text-classification.md)**: Document categorization
5. **[Topic Modeling](05-topic-modeling.md)**: LDA, NMF
6. **[Named Entity Recognition](06-ner.md)**: Information extraction
7. **[Advanced NLP](07-advanced-nlp.md)**: Transformers, BERT

## Key Concepts

### Tokens

**Token**: Individual unit of text (word, punctuation, number)

```python
text = "Natural Language Processing is fascinating!"
tokens = text.split()
print(tokens)
# ['Natural', 'Language', 'Processing', 'is', 'fascinating!']
```

### Corpus

**Corpus**: Collection of text documents

```python
corpus = [
    "First document text",
    "Second document text",
    "Third document text"
]
```

### Vocabulary

**Vocabulary**: Set of unique words in corpus

```python
vocab = set()
for doc in corpus:
    vocab.update(doc.split())
print(f"Vocabulary size: {len(vocab)}")
```

## Common Text Analytics Tasks

### 1. Sentiment Analysis

**Goal**: Determine emotional tone (positive/negative/neutral)

**Example**: Product review analysis

```python
from textblob import TextBlob

text = "I absolutely love this product!"
blob = TextBlob(text)
sentiment = blob.sentiment.polarity

if sentiment > 0:
    print("Positive sentiment")
elif sentiment < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")
```

### 2. Text Classification

**Goal**: Assign categories to documents

**Examples**:
- Spam detection
- News article categorization
- Support ticket routing

### 3. Named Entity Recognition (NER)

**Goal**: Identify entities (people, places, organizations)

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Apple Inc.: ORG
# Steve Jobs: PERSON
# Cupertino: GPE
```

### 4. Topic Modeling

**Goal**: Discover hidden themes in document collection

**Methods**: LDA, NMF, LSA

### 5. Text Summarization

**Goal**: Generate concise summary of long text

**Types**:
- Extractive: Select key sentences
- Abstractive: Generate new sentences

## Text Preprocessing Steps

### Standard Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Complete text preprocessing pipeline
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenize
    tokens = text.split()
    
    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    # 8. Rejoin
    return ' '.join(tokens)

# Example
raw = "I'm loving this NEW product!!! Check it out at www.example.com"
clean = preprocess_text(raw)
print(f"Original: {raw}")
print(f"Cleaned: {clean}")
```

## Popular Python Libraries

### NLTK (Natural Language Toolkit)

**Purpose**: Comprehensive NLP library

**Strengths**:
- Educational and research-oriented
- Extensive documentation
- Many corpora and models

```python
import nltk
nltk.download('punkt')  # Tokenizer
nltk.download('averaged_perceptron_tagger')  # POS tagger
```

### spaCy

**Purpose**: Industrial-strength NLP

**Strengths**:
- Fast and efficient
- Production-ready
- Pre-trained models

```python
import spacy
nlp = spacy.load('en_core_web_sm')
```

### Gensim

**Purpose**: Topic modeling and document similarity

**Strengths**:
- Word2Vec implementation
- Efficient LDA
- Doc2Vec

```python
from gensim.models import Word2Vec, LdaModel
```

### Transformers (Hugging Face)

**Purpose**: State-of-the-art transformer models

**Strengths**:
- BERT, GPT, RoBERTa
- Transfer learning
- Many pre-trained models

```python
from transformers import pipeline
sentiment = pipeline('sentiment-analysis')
```

## Common Challenges

### 1. Language Ambiguity

**Problem**: Same word, different meanings

**Example**: "bank" (financial institution vs. river bank)

**Solution**: Context-aware embeddings (BERT)

### 2. Sarcasm and Irony

**Problem**: Literal meaning ≠ intended meaning

**Example**: "Great! Another meeting..." (negative despite "great")

**Solution**: Advanced models with context understanding

### 3. Misspellings and Typos

**Problem**: Variations reduce accuracy

**Example**: "definately" vs "definitely"

**Solution**: Spell checkers, character-level models

### 4. Domain-Specific Language

**Problem**: Technical jargon not in general vocabulary

**Example**: Medical terms, legal language

**Solution**: Domain-specific training data and lexicons

### 5. Multiple Languages

**Problem**: Mixed-language text

**Example**: Code-switching in social media

**Solution**: Language detection, multilingual models

## Evaluation Metrics

### Classification Tasks

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean

### Sentiment Analysis

- **Accuracy**: Match with human labels
- **MAE**: Mean absolute error for rating prediction
- **Correlation**: With human ratings

### Topic Modeling

- **Coherence**: Topic interpretability
- **Perplexity**: Model fit to held-out data
- **Human evaluation**: Subject matter experts

### Information Extraction

- **Exact match**: Correct boundaries and type
- **Partial match**: Overlapping spans
- **Type accuracy**: Correct entity type

## Real-World Applications

### Customer Service

**Use cases**:
- Sentiment monitoring of support tickets
- Automatic ticket routing
- Chatbot responses
- FAQ automation

### Marketing

**Use cases**:
- Brand sentiment tracking
- Campaign effectiveness
- Influencer identification
- Competitor analysis

### Healthcare

**Use cases**:
- Clinical note analysis
- Medical coding
- Drug interaction detection
- Patient sentiment analysis

### Finance

**Use cases**:
- News sentiment for trading
- Fraud detection in communications
- Regulatory compliance
- Risk assessment from reports

### Legal

**Use cases**:
- Contract analysis
- Legal research
- E-discovery
- Case outcome prediction

## Best Practices

### Data Preparation

✅ **Clean thoroughly**: Remove noise early
✅ **Preserve context**: Don't over-preprocess
✅ **Handle imbalanced classes**: Use stratification
✅ **Split properly**: Train/validation/test

### Feature Engineering

✅ **Start simple**: Bag-of-words baseline
✅ **Try TF-IDF**: Usually better than raw counts
✅ **Use n-grams**: Capture phrases (1-3 grams)
✅ **Consider embeddings**: For semantic similarity

### Modeling

✅ **Baseline first**: Simple models often work
✅ **Cross-validate**: Essential for text data
✅ **Tune hyperparameters**: GridSearch or Bayesian
✅ **Ensemble methods**: Combine multiple models

### Evaluation

✅ **Multiple metrics**: Not just accuracy
✅ **Error analysis**: Inspect misclassifications
✅ **Human evaluation**: For subjective tasks
✅ **Monitor over time**: Model drift detection

## Summary

Text analytics transforms unstructured text into actionable insights. Key techniques include:

- **Preprocessing**: Clean and normalize text
- **Feature extraction**: Convert to numerical representation
- **Sentiment analysis**: Understand opinions and emotions
- **Classification**: Categorize documents
- **Topic modeling**: Discover hidden themes
- **NER**: Extract structured information

**Modern NLP** increasingly uses:
- Transfer learning (BERT, GPT)
- Contextual embeddings
- Transformer architectures
- Pre-trained models

## Further Reading

### Books
- "Natural Language Processing with Python" (NLTK Book)
- "Speech and Language Processing" by Jurafsky & Martin
- "Applied Text Analysis with Python" by Bengfort et al.
- "Text Mining in Practice with R" by Kwartler

### Courses
- Stanford CS224N: NLP with Deep Learning
- Coursera: Natural Language Processing Specialization
- fast.ai: NLP Course

### Resources
- spaCy documentation
- Hugging Face Model Hub
- Papers With Code: NLP
- ACL Anthology (research papers)

## Next Steps

Begin with [Text Preprocessing](01-text-preprocessing.md) to learn the foundation of all text analytics tasks.

---

**After completing this chapter**, you'll be equipped to:
- Build sentiment analysis systems
- Create text classifiers
- Extract insights from customer feedback
- Implement document search engines
- Develop content recommendation systems