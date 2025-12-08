# Text Analytics - Preprocessing

## Learning Objectives

- Understand text data characteristics
- Perform text cleaning and normalization
- Apply tokenization and stemming
- Use lemmatization effectively
- Remove stop words
- Build text preprocessing pipelines

## Introduction to Text Data

```python
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("""
TEXT DATA CHALLENGES:

1. UNSTRUCTURED:
   - No fixed schema
   - Variable length
   - Rich but messy

2. HIGH DIMENSIONALITY:
   - Large vocabulary
   - Sparse representations
   - Curse of dimensionality

3. CONTEXT-DEPENDENT:
   - Words have multiple meanings
   - Order matters
   - Sarcasm, idioms, etc.

4. NOISY:
   - Typos and abbreviations
   - Informal language
   - Special characters
""")

# Sample documents
documents = [
    "Machine learning is AMAZING! It's revolutionizing AI. #ML",
    "I love Python programming... it's so versatile!!!",
    "Natural Language Processing (NLP) helps computers understand text.",
    "Data Science combines statistics, programming & domain knowledge.",
    "The quick brown fox jumps over the lazy dog. 🦊",
    "Visit https://example.com for more info! Email: test@example.com"
]

print("\nSample Documents:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")
```

## Text Cleaning

### Lowercase Conversion

```python
def to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

print("\n1. LOWERCASE CONVERSION")
print("="*70)
for i, doc in enumerate(documents[:3], 1):
    cleaned = to_lowercase(doc)
    print(f"Original: {doc}")
    print(f"Cleaned:  {cleaned}\n")
```

### Remove URLs and Emails

```python
def remove_urls_emails(text):
    """Remove URLs and email addresses"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

print("\n2. REMOVE URLs AND EMAILS")
print("="*70)
for doc in documents[-1:]:
    cleaned = remove_urls_emails(doc)
    print(f"Original: {doc}")
    print(f"Cleaned:  {cleaned}\n")
```

### Remove Special Characters and Punctuation

```python
def remove_special_chars(text, keep_punctuation=False):
    """Remove special characters and optionally punctuation"""
    if keep_punctuation:
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    else:
        # Keep only letters and numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

print("\n3. REMOVE SPECIAL CHARACTERS")
print("="*70)
for doc in documents[:3]:
    cleaned = remove_special_chars(doc)
    print(f"Original: {doc}")
    print(f"Cleaned:  {cleaned}\n")
```

### Remove Extra Whitespace

```python
def remove_whitespace(text):
    """Remove extra whitespace"""
    return ' '.join(text.split())

text_with_spaces = "This   has    too     many      spaces"
print("\n4. REMOVE EXTRA WHITESPACE")
print("="*70)
print(f"Original: '{text_with_spaces}'")
print(f"Cleaned:  '{remove_whitespace(text_with_spaces)}'")
```

### Remove Numbers

```python
def remove_numbers(text):
    """Remove numbers from text"""
    return re.sub(r'\d+', '', text)

text_with_numbers = "There are 123 apples and 456 oranges in 2024"
print("\n5. REMOVE NUMBERS")
print("="*70)
print(f"Original: {text_with_numbers}")
print(f"Cleaned:  {remove_numbers(text_with_numbers)}")
```

## Tokenization

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Natural Language Processing is fascinating! It enables machines to understand text."

print("\nTOKENIZATION")
print("="*70)

# Word tokenization
words = word_tokenize(text)
print(f"Original: {text}")
print(f"\nWord tokens: {words}")
print(f"Number of words: {len(words)}")

# Sentence tokenization
long_text = """Machine learning is great. Deep learning is even better! 
What about NLP? It's the best."""

sentences = sent_tokenize(long_text)
print(f"\nOriginal: {long_text}")
print(f"\nSentence tokens:")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

# Custom tokenization
def simple_tokenize(text):
    """Simple whitespace tokenization"""
    return text.lower().split()

print(f"\nSimple tokenization: {simple_tokenize(text)}")
```

## Stop Words Removal

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

print("\nSTOP WORDS REMOVAL")
print("="*70)
print(f"Number of English stop words: {len(stop_words)}")
print(f"Sample stop words: {list(stop_words)[:20]}")

def remove_stopwords(text):
    """Remove stop words from text"""
    words = word_tokenize(text.lower())
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

text = "This is a sample sentence demonstrating the removal of stop words"
print(f"\nOriginal: {text}")
print(f"Without stop words: {remove_stopwords(text)}")

# Custom stop words
custom_stops = stop_words.union({'sample', 'demonstrating'})
filtered_custom = [word for word in word_tokenize(text.lower()) 
                   if word not in custom_stops]
print(f"With custom stops: {' '.join(filtered_custom)}")
```

## Stemming

**Reduces words to root form** (crude, fast)

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer('english')

print("\nSTEMMING")
print("="*70)

words = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly', 'fishing', 'fished', 'fisher']

print(f"{'Word':<15} {'Porter':<15} {'Snowball':<15}")
print("-" * 45)
for word in words:
    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)
    print(f"{word:<15} {porter_stem:<15} {snowball_stem:<15}")

print("""
\nSTEMMING CHARACTERISTICS:
  ✓ Fast
  ✓ Simple rule-based
  ✗ May produce non-words (e.g., 'fishi')
  ✗ Over-stemming: 'university' → 'univers'
  ✗ Under-stemming: 'data', 'datum' → different stems
""")
```

## Lemmatization

**Reduces words to dictionary form** (accurate, slower)

```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()

print("\nLEMMATIZATION")
print("="*70)

words = ['running', 'runs', 'ran', 'runner', 'better', 'best', 'cacti', 'geese', 'am', 'is', 'are']

print(f"{'Word':<15} {'Lemma (noun)':<15} {'Lemma (verb)':<15}")
print("-" * 45)
for word in words:
    lemma_n = lemmatizer.lemmatize(word, pos='n')  # noun
    lemma_v = lemmatizer.lemmatize(word, pos='v')  # verb
    print(f"{word:<15} {lemma_n:<15} {lemma_v:<15}")

print("""
\nLEMMATIZATION CHARACTERISTICS:
  ✓ Produces real words
  ✓ More accurate
  ✓ Context-aware (with POS tags)
  ✗ Slower than stemming
  ✗ Requires POS tagging for best results
""")

# POS-aware lemmatization
def get_wordnet_pos(tag):
    """Convert POS tag to WordNet POS"""
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun

def lemmatize_with_pos(text):
    """Lemmatize text with POS tagging"""
    words = word_tokenize(text.lower())
    pos_tags = pos_tag(words)
    
    lemmas = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmas.append(lemma)
    
    return ' '.join(lemmas)

text = "The striped bats are hanging on their feet for best results"
print(f"\nOriginal: {text}")
print(f"Lemmatized: {lemmatize_with_pos(text)}")
```

## Complete Preprocessing Pipeline

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, lowercase=True, remove_urls=True, remove_numbers=True,
                 remove_punctuation=True, remove_stopwords=True, lemmatize=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean(self, text):
        """Apply all preprocessing steps"""
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def transform(self, documents):
        """Transform multiple documents"""
        return [self.clean(doc) for doc in documents]

# Example usage
preprocessor = TextPreprocessor()

print("\n" + "="*70)
print("COMPLETE PREPROCESSING PIPELINE")
print("="*70)

documents = [
    "Machine Learning is AMAZING! Visit https://ml.com for details.",
    "I love Python programming! It's versatile & powerful.",
    "NLP helps computers understand natural language effectively."
]

for i, doc in enumerate(documents, 1):
    cleaned = preprocessor.clean(doc)
    print(f"\nDocument {i}:")
    print(f"  Original: {doc}")
    print(f"  Cleaned:  {cleaned}")

# Process all documents
cleaned_docs = preprocessor.transform(documents)
print(f"\nTotal documents processed: {len(cleaned_docs)}")
```

## Text Statistics

```python
from collections import Counter
import matplotlib.pyplot as plt

def text_statistics(documents):
    """Calculate text statistics"""
    # Combine all documents
    all_text = ' '.join(documents)
    tokens = word_tokenize(all_text.lower())
    
    stats = {
        'num_documents': len(documents),
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'avg_doc_length': np.mean([len(word_tokenize(doc)) for doc in documents]),
        'vocabulary_size': len(set(tokens)),
        'lexical_diversity': len(set(tokens)) / len(tokens)
    }
    
    return stats, Counter(tokens)

stats, word_freq = text_statistics(cleaned_docs)

print("\nTEXT STATISTICS")
print("="*70)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Most common words
print("\nTop 10 Most Frequent Words:")
for word, count in word_freq.most_common(10):
    print(f"  {word}: {count}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word frequency
top_words = dict(word_freq.most_common(15))
axes[0].barh(list(top_words.keys()), list(top_words.values()))
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 15 Words')
axes[0].invert_yaxis()
axes[0].grid(alpha=0.3)

# Word length distribution
word_lengths = [len(word) for word in word_freq.keys()]
axes[1].hist(word_lengths, bins=15, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Word Length')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Word Length Distribution')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Text preprocessing** is crucial for NLP tasks
2. **Lowercase conversion** standardizes text
3. **Remove noise**: URLs, emails, special characters
4. **Tokenization** splits text into units
5. **Stop words** are common but low-value words
6. **Stemming** is fast but crude
7. **Lemmatization** is accurate but slower
8. **POS tagging** improves lemmatization
9. **Pipelines** ensure consistent preprocessing
10. **Balance** between cleaning and information retention
:::

## Further Reading

- Bird, S. et al. (2009). "Natural Language Processing with Python"
- Manning, C.D. et al. (2008). "Introduction to Information Retrieval"
- NLTK Documentation: [NLTK](https://www.nltk.org/)
