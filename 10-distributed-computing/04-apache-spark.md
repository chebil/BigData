# Apache Spark Fundamentals

## Learning Objectives

- Understand Spark architecture and core concepts
- Explain RDDs (Resilient Distributed Datasets)
- Work with Spark DataFrames
- Understand lazy evaluation and transformations
- Apply Spark operations for distributed computing

## Introduction

Apache Spark is a unified analytics engine for large-scale data processing. It provides **in-memory processing** capabilities that make it **10-100x faster** than traditional Hadoop MapReduce for iterative algorithms.

## Why Spark?

### Limitations of Hadoop MapReduce

1. **Disk I/O Overhead**: Every Map and Reduce step writes to disk
2. **No Inter-stage Caching**: Intermediate results always persisted
3. **Complex Programming**: Requires Java and verbose code
4. **Batch-Only**: No support for iterative or interactive workloads

### Spark Advantages

1. **In-Memory Processing**: Cache data in RAM across stages
2. **Unified Framework**: Batch, streaming, SQL, ML, graph processing
3. **Easy APIs**: Python, Scala, Java, R, SQL
4. **Lazy Evaluation**: Optimizes entire execution plan
5. **Fault Tolerance**: Automatic recovery without replication overhead

## Spark vs. Hadoop MapReduce

### Performance Comparison

```python
# Word Count in Hadoop MapReduce (Java)
# Requires:
# - Mapper class (~30 lines)
# - Reducer class (~20 lines)
# - Driver class (~40 lines)
# Total: ~90 lines of Java code

# Word Count in PySpark
text_file = sc.textFile("hdfs://data/textfile.txt")
counts = (text_file
          .flatMap(lambda line: line.split())
          .map(lambda word: (word, 1))
          .reduceByKey(lambda a, b: a + b))
counts.saveAsTextFile("hdfs://output/wordcount")

# Total: 5 lines of Python!
```

### Speed Comparison

| Workload | Hadoop MapReduce | Apache Spark | Speedup |
|----------|-----------------|--------------|----------|
| Logistic Regression (10 iterations) | 346 seconds | 3.4 seconds | **100x** |
| K-means Clustering | 185 seconds | 8.6 seconds | **21x** |
| PageRank | 171 seconds | 23 seconds | **7.4x** |
| Sort (1 TB) | 3,150 seconds | 206 seconds | **15x** |

## Spark Architecture

### Cluster Components

```
Driver Program
  ├── SparkContext (entry point)
  ├── Creates RDDs/DataFrames
  └── Sends tasks to executors

Cluster Manager (YARN, Mesos, or Standalone)
  ├── Allocates resources
  └── Monitors executors

Executors (Worker Nodes)
  ├── Run tasks
  ├── Cache data in memory
  └── Return results to driver
```

### Execution Model

1. **Driver** creates logical execution plan
2. **DAG Scheduler** organizes stages and tasks
3. **Task Scheduler** assigns tasks to executors
4. **Executors** perform computations in parallel
5. **Results** returned to driver

## Core Abstractions

### 1. RDD (Resilient Distributed Dataset)

**Definition**: Immutable, distributed collection of objects that can be processed in parallel.

**Properties**:
- **Resilient**: Automatically recovers from failures
- **Distributed**: Partitioned across cluster nodes
- **Dataset**: Collection of data elements

**Creating RDDs**:

```python
from pyspark import SparkContext

sc = SparkContext("local", "RDD Example")

# Method 1: Parallelize existing collection
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)

# Method 2: Load from external storage
text_rdd = sc.textFile("data/sample.txt")

# Method 3: Transform existing RDD
squared_rdd = rdd.map(lambda x: x ** 2)
```

**RDD Operations**:

#### Transformations (Lazy)

Create new RDD from existing one:

```python
# map: Apply function to each element
rdd = sc.parallelize([1, 2, 3, 4])
squared = rdd.map(lambda x: x ** 2)  # [1, 4, 9, 16]

# filter: Keep elements matching condition
even = rdd.filter(lambda x: x % 2 == 0)  # [2, 4]

# flatMap: Map then flatten
words_rdd = sc.parallelize(["hello world", "spark tutorial"])
words = words_rdd.flatMap(lambda line: line.split())
# Result: ["hello", "world", "spark", "tutorial"]

# distinct: Remove duplicates
rdd = sc.parallelize([1, 2, 2, 3, 3, 3])
unique = rdd.distinct()  # [1, 2, 3]

# union: Combine two RDDs
rdd1 = sc.parallelize([1, 2, 3])
rdd2 = sc.parallelize([3, 4, 5])
combined = rdd1.union(rdd2)  # [1, 2, 3, 3, 4, 5]
```

#### Actions (Trigger Execution)

Return values to driver or write to storage:

```python
# collect: Return all elements to driver
rdd = sc.parallelize([1, 2, 3, 4, 5])
result = rdd.collect()  # [1, 2, 3, 4, 5]

# count: Number of elements
count = rdd.count()  # 5

# first: First element
first = rdd.first()  # 1

# take: First n elements
top3 = rdd.take(3)  # [1, 2, 3]

# reduce: Aggregate elements
sum_val = rdd.reduce(lambda a, b: a + b)  # 15

# saveAsTextFile: Write to disk
rdd.saveAsTextFile("output/results")
```

### 2. DataFrames

**Definition**: Distributed collection of data organized into named columns (like a table).

**Advantages over RDDs**:
- Optimized execution (Catalyst optimizer)
- Schema information
- Better performance
- Easier to use (SQL-like operations)

**Creating DataFrames**:

```python
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("DataFrame Example") \
    .getOrCreate()

# Method 1: From Python list
data = [("Alice", 25), ("Bob", 30), ("Carol", 28)]
df = spark.createDataFrame(data, ["name", "age"])

# Method 2: From pandas DataFrame
import pandas as pd
pandas_df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [25, 30, 28]
})
spark_df = spark.createDataFrame(pandas_df)

# Method 3: Read from file
df = spark.read.csv("data/people.csv", header=True, inferSchema=True)

# Show data
df.show()
# +-----+---+
# | name|age|
# +-----+---+
# |Alice| 25|
# |  Bob| 30|
# |Carol| 28|
# +-----+---+
```

**DataFrame Operations**:

```python
# Select columns
df.select("name").show()

# Filter rows
df.filter(df['age'] > 26).show()
# or
df.filter("age > 26").show()

# Add new column
from pyspark.sql.functions import col
df_with_decade = df.withColumn("decade", (col("age") / 10).cast("int") * 10)

# Group and aggregate
df.groupBy("decade").count().show()

# Sort
df.orderBy("age", ascending=False).show()

# Join
df1 = spark.createDataFrame([("Alice", "Engineering"), ("Bob", "Sales")], 
                             ["name", "dept"])
df_joined = df.join(df1, on="name", how="left")
```

## Lazy Evaluation

**Concept**: Transformations are not executed immediately; Spark builds a **lineage graph** (DAG).

```python
# None of these execute yet!
rdd1 = sc.textFile("large_file.txt")  # Transformation
rdd2 = rdd1.filter(lambda line: "ERROR" in line)  # Transformation
rdd3 = rdd2.map(lambda line: line.split())  # Transformation

# Execution starts here (Action)
result = rdd3.collect()
```

**Benefits**:
1. **Optimization**: Spark can optimize entire pipeline
2. **Pipelining**: Combine multiple transformations
3. **Fault Tolerance**: Recompute lost data from lineage

**Example**: Spark optimizes this:

```python
# User writes:
df.select("col1", "col2", "col3") \
  .filter("col1 > 100") \
  .select("col1")

# Spark optimizes to:
df.filter("col1 > 100") \
  .select("col1")  # Reads only col1!
```

## Caching and Persistence

**Problem**: Recomputing RDDs is expensive.

**Solution**: Cache frequently-used data in memory.

```python
# Cache in memory
rdd = sc.textFile("large_file.txt")
cached_rdd = rdd.cache()  # or rdd.persist()

# First action: Loads and caches
count1 = cached_rdd.count()  # Slow

# Subsequent actions: Use cached data
count2 = cached_rdd.count()  # Fast!
first_10 = cached_rdd.take(10)  # Fast!

# Unpersist when done
cached_rdd.unpersist()
```

**Persistence Levels**:

```python
from pyspark import StorageLevel

# Memory only (default for cache())
rdd.persist(StorageLevel.MEMORY_ONLY)

# Memory + Disk (spill to disk if needed)
rdd.persist(StorageLevel.MEMORY_AND_DISK)

# Serialized in memory (more compact)
rdd.persist(StorageLevel.MEMORY_ONLY_SER)

# Disk only
rdd.persist(StorageLevel.DISK_ONLY)
```

## Practical Example: Log File Analysis

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

# Initialize Spark
spark = SparkSession.builder.appName("LogAnalysis").getOrCreate()

# Load log files
logs = spark.read.text("logs/*.log")

# Extract error lines
errors = logs.filter(col("value").contains("ERROR"))

# Cache because we'll use it multiple times
errors.cache()

# Count total errors
total_errors = errors.count()
print(f"Total errors: {total_errors}")

# Extract error types using regex
from pyspark.sql.functions import regexp_extract

error_types = errors.select(
    regexp_extract(col("value"), r"ERROR: (\w+)", 1).alias("error_type")
)

# Count errors by type
error_summary = error_types \
    .groupBy("error_type") \
    .count() \
    .orderBy(desc("count"))

error_summary.show(10)

# Save results
error_summary.write.csv("output/error_summary", header=True)

# Clean up
errors.unpersist()
spark.stop()
```

## Word Count: Complete Example

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, regexp_replace, desc

# Create Spark session
spark = SparkSession.builder \
    .appName("WordCount") \
    .getOrCreate()

# Read text file
df = spark.read.text("data/shakespeare.txt")

# Process:
# 1. Convert to lowercase
# 2. Remove punctuation
# 3. Split into words
# 4. Explode into separate rows
words_df = df.select(
    explode(
        split(
            regexp_replace(
                lower(col("value")),
                "[^a-z\\s]",
                ""
            ),
            "\\s+"
        )
    ).alias("word")
)

# Remove empty strings
words_df = words_df.filter(col("word") != "")

# Count occurrences
word_counts = words_df \
    .groupBy("word") \
    .count() \
    .orderBy(desc("count"))

# Show top 20 words
word_counts.show(20)

# Save results
word_counts.write \
    .mode("overwrite") \
    .parquet("output/word_counts.parquet")

spark.stop()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Spark is 10-100x faster** than Hadoop MapReduce due to in-memory processing
2. **RDDs** are the fundamental abstraction for distributed collections
3. **DataFrames** provide optimized, schema-aware data structures
4. **Lazy evaluation** enables query optimization before execution
5. **Caching** improves performance for iterative algorithms
6. **Transformations** are lazy; **Actions** trigger execution
7. Spark provides **unified framework** for batch, streaming, SQL, and ML
:::

## Practical Exercises

### Exercise 1: RDD Transformations

Given an RDD of numbers, find the sum of squares of even numbers:

```python
rdd = sc.parallelize(range(1, 101))
# Your code here
result = rdd.filter(lambda x: x % 2 == 0) \
           .map(lambda x: x ** 2) \
           .reduce(lambda a, b: a + b)
print(result)
```

### Exercise 2: DataFrame Analysis

Analyze a CSV file of sales data:

```python
# Load data
sales_df = spark.read.csv("data/sales.csv", header=True, inferSchema=True)

# Tasks:
# 1. Total revenue by product category
# 2. Top 10 customers by purchase amount
# 3. Average order value by month
```

### Exercise 3: Log Analysis

Parse web server logs and extract insights:

```python
# Load Apache access logs
logs = spark.read.text("logs/access.log")

# Extract:
# 1. Top 10 requested URLs
# 2. HTTP status code distribution
# 3. Requests per hour
```

## Further Reading

- [Spark Programming Guide](https://spark.apache.org/docs/latest/programming-guide.html)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- Karau, H. et al. (2015). "Learning Spark"
- Zaharia, M. et al. (2012). "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing"
