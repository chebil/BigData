# MapReduce Programming Paradigm

## Learning Objectives

- Understand the MapReduce programming model
- Implement Map and Reduce functions
- Recognize suitable problems for MapReduce
- Optimize MapReduce jobs for performance
- Compare MapReduce with modern alternatives

## Introduction

MapReduce is a programming model for processing large datasets in parallel across distributed clusters. Popularized by Google in 2004, it provides automatic parallelization, fault tolerance, and data distribution.

## The MapReduce Model

### Core Concept

**MapReduce = Map Phase + Shuffle/Sort + Reduce Phase**

```
Input Data
    |
    v
[Map Phase] ──> Intermediate Key-Value Pairs
    |
    v
[Shuffle & Sort] ──> Grouped by Key
    |
    v
[Reduce Phase] ──> Final Output
```

### Key-Value Pairs

**Everything in MapReduce operates on key-value pairs**:

```python
# Input
(key, value) = (filename, line_content)

# Map output
(key, value) = (word, 1)

# Reduce output
(key, value) = (word, count)
```

## Map Phase

### Purpose

Transform input data into intermediate key-value pairs.

### Function Signature

```python
def mapper(key_in, value_in):
    """
    Map function processes one input record at a time
    
    Args:
        key_in: Input key (e.g., line number)
        value_in: Input value (e.g., line content)
    
    Yields:
        (key_out, value_out): Intermediate key-value pairs
    """
    # Process value_in
    # Emit zero or more (key, value) pairs
    yield (key_out, value_out)
```

### Example: Word Count Mapper

```python
def word_count_mapper(document_id, text):
    """
    Input: (document_id, text)
    Output: (word, 1) for each word
    """
    words = text.lower().split()
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:')
        if clean_word:
            yield (clean_word, 1)

# Example usage
text = "Hello World Hello MapReduce"
for key, value in word_count_mapper("doc1", text):
    print(f"({key}, {value})")

# Output:
# (hello, 1)
# (world, 1)
# (hello, 1)
# (mapreduce, 1)
```

## Shuffle and Sort Phase

### Purpose

Group all values associated with the same intermediate key.

### Process

```python
# Map output (unsorted)
[("hello", 1), ("world", 1), ("hello", 1), ("mapreduce", 1)]

# After Shuffle & Sort
[("hello", [1, 1]), ("mapreduce", [1]), ("world", [1])]
```

**Key Points**:
- Automatic in Hadoop/Spark
- Keys are sorted
- Values are grouped into iterables
- Network transfer ("shuffle") happens here

## Reduce Phase

### Purpose

Aggregate values for each key to produce final output.

### Function Signature

```python
def reducer(key, values):
    """
    Reduce function processes all values for a key
    
    Args:
        key: Intermediate key
        values: Iterator of all values for this key
    
    Yields:
        (key, aggregated_value): Final output
    """
    # Aggregate values
    result = aggregate(values)
    yield (key, result)
```

### Example: Word Count Reducer

```python
def word_count_reducer(word, counts):
    """
    Input: (word, [1, 1, 1, ...])
    Output: (word, total_count)
    """
    total = sum(counts)
    yield (word, total)

# Example usage
word = "hello"
counts = [1, 1]
for key, value in word_count_reducer(word, counts):
    print(f"({key}, {value})")

# Output: (hello, 2)
```

## Complete Word Count Example

### Conceptual Implementation

```python
class MapReduce:
    def __init__(self, mapper, reducer):
        self.mapper = mapper
        self.reducer = reducer
    
    def run(self, input_data):
        # MAP PHASE
        intermediate = []
        for key, value in input_data:
            for k, v in self.mapper(key, value):
                intermediate.append((k, v))
        
        # SHUFFLE & SORT
        grouped = {}
        for key, value in intermediate:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)
        
        # REDUCE PHASE
        output = []
        for key in sorted(grouped.keys()):
            for k, v in self.reducer(key, grouped[key]):
                output.append((k, v))
        
        return output

# Define mapper and reducer
def mapper(doc_id, text):
    for word in text.split():
        yield (word.lower(), 1)

def reducer(word, counts):
    yield (word, sum(counts))

# Run MapReduce
input_data = [
    ("doc1", "Hello World"),
    ("doc2", "Hello MapReduce"),
    ("doc3", "MapReduce World")
]

mr = MapReduce(mapper, reducer)
result = mr.run(input_data)

for word, count in result:
    print(f"{word}: {count}")

# Output:
# hello: 2
# mapreduce: 2  
# world: 2
```

### Hadoop Streaming (Python)

```python
#!/usr/bin/env python3
# mapper.py
import sys

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        print(f"{word.lower()}\t1")
```

```python
#!/usr/bin/env python3
# reducer.py
import sys
from collections import defaultdict

counts = defaultdict(int)

for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t')
    counts[word] += int(count)

for word, count in sorted(counts.items()):
    print(f"{word}\t{count}")
```

```bash
# Run with Hadoop Streaming
hadoop jar hadoop-streaming.jar \
  -input /user/data/input \
  -output /user/data/output \
  -mapper mapper.py \
  -reducer reducer.py \
  -file mapper.py \
  -file reducer.py
```

## Common MapReduce Patterns

### 1. Filtering

**Use Case**: Select records matching criteria

```python
def filter_mapper(key, record):
    """Keep only records where age > 25"""
    age = int(record.split(',')[2])
    if age > 25:
        yield (key, record)

def identity_reducer(key, values):
    """Pass through (no aggregation needed)"""
    for value in values:
        yield (key, value)
```

### 2. Summarization

**Use Case**: Calculate aggregates (sum, average, min, max)

```python
def sales_mapper(product_id, sale_record):
    """Extract product sales"""
    product, amount = sale_record.split(',')
    yield (product, float(amount))

def sum_reducer(product, amounts):
    """Total sales per product"""
    yield (product, sum(amounts))

def average_reducer(product, amounts):
    """Average sale amount per product"""
    amounts_list = list(amounts)
    yield (product, sum(amounts_list) / len(amounts_list))
```

### 3. Inverted Index

**Use Case**: Build search index

```python
def inverted_index_mapper(document_id, text):
    """Map: (word, document_id)"""
    for word in text.split():
        yield (word.lower(), document_id)

def inverted_index_reducer(word, doc_ids):
    """Reduce: (word, [list of document IDs])"""
    unique_docs = list(set(doc_ids))
    yield (word, unique_docs)

# Result:
# ("hadoop", ["doc1", "doc3", "doc5"])
# ("spark", ["doc2", "doc3", "doc4"])
```

### 4. Join (Reduce-Side Join)

**Use Case**: Join two datasets

```python
def join_mapper_customers(customer_id, customer_data):
    """Tag with source: customer"""
    yield (customer_id, ("customer", customer_data))

def join_mapper_orders(customer_id, order_data):
    """Tag with source: order"""
    yield (customer_id, ("order", order_data))

def join_reducer(customer_id, tagged_values):
    """Join customer and order data"""
    customer_info = None
    orders = []
    
    for tag, data in tagged_values:
        if tag == "customer":
            customer_info = data
        elif tag == "order":
            orders.append(data)
    
    if customer_info:
        for order in orders:
            yield (customer_id, (customer_info, order))
```

### 5. Top N

**Use Case**: Find top N records

```python
import heapq

def top_n_mapper(key, record):
    """Pass through all records"""
    score = int(record.split(',')[1])
    yield ("all", (score, record))

def top_n_reducer(key, score_record_pairs, n=10):
    """Keep only top N"""
    # Use heap to efficiently track top N
    top_n = heapq.nlargest(n, score_record_pairs, key=lambda x: x[0])
    
    for score, record in top_n:
        yield (key, record)
```

## Combiners

**Purpose**: Mini-reduce that runs on mapper output before shuffle

**Benefit**: Reduces network traffic

```python
# Without Combiner
# Mapper outputs: ("word", 1) × 1,000,000 times
# Network transfer: 1,000,000 records

# With Combiner  
# Mapper outputs: ("word", 1) × 1,000,000 times
# Combiner aggregates locally: ("word", 1000000)
# Network transfer: 1 record!
```

**Implementation**:

```python
def word_count_mapper(doc_id, text):
    for word in text.split():
        yield (word, 1)

def word_count_combiner(word, counts):
    """Combiner is same as reducer for word count"""
    yield (word, sum(counts))

def word_count_reducer(word, counts):
    yield (word, sum(counts))
```

**Note**: Combiner must be:
- **Associative**: (a + b) + c = a + (b + c)
- **Commutative**: a + b = b + a

## Partitioners

**Purpose**: Determine which reducer receives which keys

**Default**: Hash partitioner

```python
def default_partitioner(key, num_reducers):
    return hash(key) % num_reducers
```

**Custom Partitioner**:

```python
def range_partitioner(key, num_reducers):
    """
    Partition alphabetically:
    Reducer 0: A-M
    Reducer 1: N-Z
    """
    if key[0].lower() < 'n':
        return 0
    else:
        return 1
```

## MapReduce Execution Flow

```
1. INPUT SPLIT
   Large file split into chunks (typically HDFS block size)
   
2. MAP
   One mapper per split
   Runs in parallel across cluster
   Outputs intermediate key-value pairs
   
3. COMBINE (optional)
   Local aggregation on each mapper node
   
4. PARTITION
   Determine which reducer gets which keys
   
5. SHUFFLE
   Transfer data across network
   Group by key
   
6. SORT
   Sort keys for each reducer
   
7. REDUCE
   Process all values for each key
   Outputs final results
   
8. OUTPUT
   Write to HDFS (or other storage)
```

## Real-World Examples

### Log Analysis

```python
def log_mapper(filename, log_line):
    """
    Parse Apache log and extract status code
    """
    # Example: 192.168.1.1 - - [01/Jan/2024:12:00:00] "GET /page HTTP/1.1" 200 1234
    parts = log_line.split()
    if len(parts) >= 9:
        status_code = parts[8]
        yield (status_code, 1)

def log_reducer(status_code, counts):
    """
    Count requests by status code
    """
    yield (status_code, sum(counts))

# Result:
# ("200", 150000)  # OK
# ("404", 5000)    # Not Found
# ("500", 100)     # Server Error
```

### Click-Through Rate (CTR)

```python
def ctr_mapper(ad_id, event):
    """
    Track ad impressions and clicks
    """
    event_type, ad_id = event.split(',')
    
    if event_type == 'impression':
        yield (ad_id, ('impression', 1))
    elif event_type == 'click':
        yield (ad_id, ('click', 1))

def ctr_reducer(ad_id, events):
    """
    Calculate CTR = clicks / impressions
    """
    impressions = 0
    clicks = 0
    
    for event_type, count in events:
        if event_type == 'impression':
            impressions += count
        elif event_type == 'click':
            clicks += count
    
    ctr = clicks / impressions if impressions > 0 else 0
    yield (ad_id, {'impressions': impressions, 'clicks': clicks, 'ctr': ctr})
```

## Limitations of MapReduce

### 1. Disk I/O Overhead

- Every map-reduce stage writes to disk
- Iterative algorithms (ML) are slow

### 2. Not Suitable For:

- **Interactive queries**: Too much latency
- **Real-time processing**: Batch-oriented
- **Graph algorithms**: Require many iterations
- **Complex workflows**: Difficult to chain jobs

### 3. Programming Complexity

- Requires thinking in map-reduce paradigm
- Simple operations need boilerplate code

## Modern Alternatives

### Apache Spark

```python
# MapReduce (50+ lines)
# vs.
# Spark (3 lines)
from pyspark import SparkContext

sc = SparkContext()
text = sc.textFile("hdfs://data/input")
counts = text.flatMap(lambda line: line.split()) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("hdfs://data/output")
```

### Apache Flink

- True streaming (not micro-batches)
- Lower latency than Spark Streaming

### Apache Beam

- Unified API for batch and streaming
- Portable across execution engines

## Key Takeaways

:::{admonition} Summary
:class: note

1. **MapReduce processes data in two phases**: Map (transform) and Reduce (aggregate)
2. **Shuffle/Sort** automatically groups values by key
3. **Combiners** reduce network traffic through local aggregation
4. **Partitioners** control data distribution to reducers
5. **MapReduce excels at**: large-scale batch processing, embarrassingly parallel problems
6. **MapReduce struggles with**: iterative algorithms, interactive queries, real-time processing
7. **Modern alternatives** (Spark, Flink) often provide better performance and ease of use
8. **Design patterns** solve common problems: filtering, joins, aggregations, top-N
:::

## Practical Exercises

### Exercise 1: Unique Users Per Day

Given log data with (timestamp, user_id), count unique users per day.

```python
def mapper(key, log_entry):
    # Your code here
    pass

def reducer(key, user_ids):
    # Your code here
    pass
```

### Exercise 2: Secondary Sort

Sort records by temperature within each weather station.

### Exercise 3: Matrix Multiplication

Implement matrix multiplication using MapReduce.

## Further Reading

- Dean, J. & Ghemawat, S. (2004). "MapReduce: Simplified Data Processing on Large Clusters"
- White, T. (2015). "Hadoop: The Definitive Guide", Chapters 2-8
- Lin, J. & Dyer, C. (2010). "Data-Intensive Text Processing with MapReduce"
