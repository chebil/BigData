# Data Types and Structures

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish between structured, semi-structured, quasi-structured, and unstructured data
- Identify appropriate data storage and processing techniques for different data types
- Understand the challenges of analyzing non-structured data
- Recognize the importance of data structure in analytics workflows

## Introduction

Big Data can come in multiple forms with varying degrees of structure. Understanding these different data types is crucial for selecting appropriate tools and techniques for storage, processing, and analysis. Contrary to traditional data analysis, most Big Data (80-90% of future growth) is unstructured or semi-structured, requiring different approaches than conventional database systems.

## The Four Types of Data Structures

### 1. Structured Data

**Definition**: Data containing a defined data type, format, and structure organized in rows and columns.

**Characteristics**:
- Well-defined schema (database tables)
- Easily searchable and queryable
- Fits into relational database management systems (RDBMS)
- Consistent format and data types

**Examples**:
- Transaction data in databases
- Spreadsheets (CSV, Excel files)
- OLAP data cubes
- Customer relationship management (CRM) systems
- Enterprise resource planning (ERP) data

**Storage and Processing**:
- Relational databases (MySQL, PostgreSQL, Oracle)
- Data warehouses
- SQL queries for analysis

```python
# Example: Structured data in a pandas DataFrame
import pandas as pd

# Customer transaction data (structured)
data = {
    'customer_id': [1001, 1002, 1003, 1004],
    'transaction_date': ['2024-01-15', '2024-01-16', '2024-01-16', '2024-01-17'],
    'amount': [129.99, 45.50, 299.00, 78.25],
    'product_category': ['Electronics', 'Books', 'Clothing', 'Home']
}

df = pd.DataFrame(data)
print(df)
```

### 2. Semi-Structured Data

**Definition**: Textual data with a discernible pattern that enables parsing, typically self-describing with metadata.

**Characteristics**:
- Contains tags or markers to separate elements
- Self-describing structure (XML schema, JSON objects)
- More flexible than structured data
- Can be parsed programmatically

**Examples**:
- XML (eXtensible Markup Language) files
- JSON (JavaScript Object Notation)
- HTML web pages
- Email (MIME format)
- NoSQL database records

**Storage and Processing**:
- NoSQL databases (MongoDB, CouchDB)
- Document stores
- XML/JSON parsers
- Schema-on-read approaches

```python
# Example: Semi-structured data (JSON)
import json

# Product review in JSON format
review = {
    "review_id": "R12345",
    "product": "Laptop",
    "rating": 4.5,
    "reviewer": {
        "name": "John Doe",
        "verified_purchase": True
    },
    "review_text": "Great laptop for the price",
    "helpful_votes": 23,
    "date": "2024-01-20"
}

print(json.dumps(review, indent=2))
```

### 3. Quasi-Structured Data

**Definition**: Textual data with erratic formats that can be formatted with effort, tools, and time.

**Characteristics**:
- Inconsistent data values and formats
- Requires significant preprocessing
- Contains patterns but lacks strict structure
- Often needs custom parsers

**Examples**:
- Web server log files
- Clickstream data with inconsistencies
- Application logs
- Email headers and bodies
- Social media posts with hashtags

**Challenges**:
- Variable field lengths and formats
- Missing or incomplete data
- Inconsistent delimiters
- Requires data cleaning and normalization

**Clickstream Example**:

When a user searches for "EMC data science" and navigates through websites, the clickstream might look like:

```
https://www.google.com/q=EMC+data+science
https://education.emc.com/guest/campaign/data_science.aspx
https://education.emc.com/guest/certification/frameworks/tf_data_science.aspx
```

This clickstream can be parsed to understand:
- User search intent
- Navigation patterns
- Time spent on pages
- Conversion paths

```python
# Example: Parsing clickstream data
import re
from urllib.parse import urlparse, parse_qs

clickstream = [
    "https://www.google.com/search?q=EMC+data+science",
    "https://education.emc.com/guest/campaign/data_science.aspx",
    "https://education.emc.com/guest/certification/frameworks/tf_data_science.aspx"
]

for url in clickstream:
    parsed = urlparse(url)
    print(f"Domain: {parsed.netloc}")
    print(f"Path: {parsed.path}")
    if parsed.query:
        params = parse_qs(parsed.query)
        print(f"Search query: {params.get('q', ['N/A'])[0]}")
    print("-" * 50)
```

### 4. Unstructured Data

**Definition**: Data with no inherent structure, often in the form of free-form text, multimedia, or binary content.

**Characteristics**:
- No predefined data model
- Rich in information but difficult to process
- Requires advanced techniques for analysis
- Often contains valuable insights

**Examples**:
- Text documents (Word, PDF)
- Emails and their attachments
- Social media posts and comments
- Images and photographs
- Audio recordings
- Video files
- Sensor data streams

**Processing Techniques**:
- Natural Language Processing (NLP)
- Computer vision
- Speech recognition
- Text mining and sentiment analysis
- Deep learning models

```python
# Example: Processing unstructured text data
from collections import Counter
import re

# Customer review (unstructured text)
review_text = """
I absolutely love this laptop! The battery life is amazing,
and the screen quality is outstanding. However, the keyboard
could be better. Overall, great value for money.
"""

# Simple text analysis
words = re.findall(r'\w+', review_text.lower())
word_freq = Counter(words)

print("Most common words:", word_freq.most_common(5))

# Sentiment keywords
positive_words = ['love', 'amazing', 'outstanding', 'great']
negative_words = ['however', 'could', 'better']

positive_count = sum(1 for word in words if word in positive_words)
negative_count = sum(1 for word in words if word in negative_words)

print(f"Positive indicators: {positive_count}")
print(f"Negative indicators: {negative_count}")
```

## Data Structure Growth Trends

The distribution of data types is shifting dramatically:

| Era | Dominant Data Type | Primary Storage |
|-----|-------------------|----------------|
| 1990s | Structured (90%) | Relational databases |
| 2000s | Structured (70%), Semi-structured (20%) | Data warehouses, Content management |
| 2010s+ | Unstructured (80%), Semi-structured (15%), Structured (5%) | Hadoop, NoSQL, Cloud storage |

## Mixed Data Scenarios

Real-world systems often contain multiple data types. Consider a software support call center:

**Structured Data**:
- Call timestamps
- Customer ID
- Product type
- Problem category
- Resolution time

**Semi-Structured Data**:
- XML ticket records
- JSON API responses

**Quasi-Structured Data**:
- Email threads
- Chat transcripts

**Unstructured Data**:
- Free-form problem descriptions
- Audio recordings of phone calls
- Attached screenshots or documents

Analyzing all these data types together provides richer insights than analyzing structured data alone.

## Choosing the Right Approach

### For Structured Data:
- Use SQL databases and data warehouses
- Apply traditional BI and reporting tools
- Leverage indexing and relational algebra

### For Semi-Structured Data:
- Use NoSQL databases (MongoDB, Cassandra)
- Employ JSON/XML parsing libraries
- Consider document-oriented storage

### For Quasi-Structured Data:
- Apply regular expressions for parsing
- Use data cleaning and normalization tools
- Consider ETL pipelines

### For Unstructured Data:
- Apply NLP and text mining
- Use deep learning for images/video
- Leverage distributed computing (Hadoop, Spark)
- Consider specialized tools (Elasticsearch for text search)

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Structured data** is well-organized in rows and columns, easily queryable with SQL
2. **Semi-structured data** has some organizational properties (XML, JSON) but lacks rigid schema
3. **Quasi-structured data** has erratic formats requiring significant preprocessing
4. **Unstructured data** lacks inherent structure and requires advanced analytics techniques
5. Most Big Data (80-90%) is unstructured or semi-structured
6. Real-world systems often contain multiple data types that should be analyzed together
7. Different data types require different storage, processing, and analytical approaches
:::

## Practical Exercise

Identify data types in these scenarios:

1. A retail company's customer database with purchase history
2. Twitter posts mentioning a brand name
3. Web server access logs
4. Product reviews on an e-commerce website
5. Sensor readings from IoT devices in JSON format

## Further Reading

- White, T. (2012). "Hadoop: The Definitive Guide"
- Marz, N., & Warren, J. (2015). "Big Data: Principles and Best Practices"
- Provost, F., & Fawcett, T. (2013). "Data Science for Business"
