# The Hadoop Ecosystem

## Learning Objectives

- Understand the components of the Hadoop ecosystem
- Explain the role of HDFS, MapReduce, and YARN
- Identify ecosystem tools (Pig, Hive, HBase, Mahout)
- Recognize when to use each tool

## Introduction

Apache Hadoop is not a single tool but an **ecosystem** of interconnected projects that work together to store, process, and analyze Big Data. Understanding this ecosystem is crucial for effectively architecting Big Data solutions.

## Core Hadoop Components

### 1. HDFS (Hadoop Distributed File System)

**Purpose**: Distributed storage for massive datasets

**Key Features**:
- Stores files as blocks (default 128 MB) across cluster
- Triple replication for fault tolerance
- Rack-aware placement for reliability
- Optimized for large files and sequential reads

**Architecture**:
```
NameNode (Master)
├── Manages metadata (file locations, permissions)
├── Tracks block locations
└── Single point of coordination

DataNodes (Workers)
├── Store actual data blocks
├── Send heartbeats to NameNode
└── Perform data replication

SecondaryNameNode
└── Assists with checkpoint creation
```

**Example Usage**:
```bash
# Upload file to HDFS
hdfs dfs -put large_dataset.csv /user/data/

# List files
hdfs dfs -ls /user/data/

# Download file
hdfs dfs -get /user/data/large_dataset.csv .

# Check file blocks
hdfs fsck /user/data/large_dataset.csv -files -blocks
```

### 2. MapReduce

**Purpose**: Distributed data processing framework

**Key Concepts**:
- **Map**: Transform input data into key-value pairs
- **Shuffle/Sort**: Group values by key
- **Reduce**: Aggregate values for each key

**Word Count Example**:

```python
# Conceptual Python pseudocode
class Mapper:
    def map(self, document):
        for word in document.split():
            self.emit(word, 1)

class Reducer:
    def reduce(self, word, counts):
        self.emit(word, sum(counts))
```

**Characteristics**:
- Batch processing (not real-time)
- Fault-tolerant through task re-execution
- Data locality: computation moves to data
- Automatic parallelization

### 3. YARN (Yet Another Resource Negotiator)

**Purpose**: Cluster resource management and job scheduling

**Components**:
```
ResourceManager (Master)
├── Schedules applications
├── Allocates resources
└── Monitors cluster health

NodeManagers (Workers)
├── Manage resources on each node
├── Launch containers for applications
└── Report resource usage

ApplicationMaster
├── Negotiates resources for each job
└── Monitors task execution
```

**Benefit**: Enables frameworks beyond MapReduce (Spark, Tez) to run on Hadoop clusters.

## Hadoop Ecosystem Tools

### Apache Pig

**Purpose**: High-level data flow scripting language

**Use Case**: ETL operations without writing Java MapReduce code

**Example** (Pig Latin):
```pig
-- Load customer data
customers = LOAD '/user/data/customers.txt' 
            AS (id:int, name:chararray, email:chararray, age:int);

-- Filter customers older than 25
filtered = FILTER customers BY age > 25;

-- Group by age
grouped = GROUP filtered BY age;

-- Count customers in each age group
counts = FOREACH grouped GENERATE group AS age, COUNT(filtered) AS count;

-- Store results
STORE counts INTO '/user/output/age_distribution';
```

**Advantages**:
- Simple syntax (similar to SQL)
- Automatic optimization
- Extensible with User-Defined Functions (UDFs)
- Good for data transformation pipelines

### Apache Hive

**Purpose**: SQL-like interface for Hadoop data

**Use Case**: Query structured data using familiar SQL syntax

**Example** (HiveQL):
```sql
-- Create external table
CREATE EXTERNAL TABLE sales (
    transaction_id INT,
    customer_id INT,
    product STRING,
    amount DECIMAL(10,2),
    transaction_date DATE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/user/data/sales/';

-- Query data
SELECT 
    product,
    SUM(amount) as total_sales,
    COUNT(*) as num_transactions
FROM sales
WHERE transaction_date >= '2024-01-01'
GROUP BY product
ORDER BY total_sales DESC
LIMIT 10;
```

**Advantages**:
- Familiar SQL syntax
- Metadata management (Hive Metastore)
- Partition pruning for performance
- Integration with BI tools

**Limitations**:
- Batch processing only (no real-time)
- Higher latency than traditional databases
- Not suitable for transactional workloads

### Apache HBase

**Purpose**: NoSQL database for real-time random read/write access

**Use Case**: Billions of rows, millions of columns, real-time queries

**Architecture**:
- Column-family oriented
- Stores data in HDFS
- Provides millisecond latency
- Horizontally scalable

**Example** (HBase Shell):
```bash
# Create table
create 'users', 'profile', 'activity'

# Insert data
put 'users', 'user001', 'profile:name', 'Alice'
put 'users', 'user001', 'profile:email', 'alice@example.com'
put 'users', 'user001', 'activity:last_login', '2024-01-15'

# Get data
get 'users', 'user001'

# Scan table
scan 'users', {STARTROW => 'user000', LIMIT => 10}
```

**Python Example** (happybase library):
```python
import happybase

# Connect to HBase
connection = happybase.Connection('localhost')
table = connection.table('users')

# Write data
table.put(b'user002', {
    b'profile:name': b'Bob',
    b'profile:email': b'bob@example.com',
    b'activity:last_login': b'2024-01-16'
})

# Read data
row = table.row(b'user002')
print(row)

# Scan multiple rows
for key, data in table.scan(row_prefix=b'user'):
    print(key, data)
```

**Use Cases**:
- User profile storage
- Time-series data
- Messaging systems (Facebook Messages)
- Real-time analytics

### Apache Mahout

**Purpose**: Scalable machine learning library

**Algorithms**:
- **Classification**: Logistic regression, Naive Bayes, Random Forests
- **Clustering**: K-means, Fuzzy K-means, Canopy clustering
- **Collaborative Filtering**: User-based, item-based recommendations

**Example** (K-means Clustering):
```bash
# Prepare data in Mahout format
mahout seqdirectory \
  -i /user/data/documents \
  -o /user/output/documents-seq

# Convert to vectors
mahout seq2sparse \
  -i /user/output/documents-seq \
  -o /user/output/documents-vectors

# Run K-means
mahout kmeans \
  -i /user/output/documents-vectors/tfidf-vectors \
  -c /user/output/clusters-initial \
  -o /user/output/clusters \
  -k 10 \
  -x 20
```

**Note**: Mahout has shifted focus from MapReduce to Spark-based implementations.

## Ecosystem Tool Comparison

| Tool | Purpose | Interface | Processing Model | Latency |
|------|---------|-----------|-----------------|----------|
| **Pig** | ETL, data transformation | Pig Latin | Batch | High |
| **Hive** | SQL queries | HiveQL (SQL) | Batch | High |
| **HBase** | NoSQL database | Java API, Shell | Real-time | Low |
| **Mahout** | Machine learning | Java, CLI | Batch | High |
| **Spark** | General processing | Python, Scala, Java | Batch + Streaming | Low-Medium |

## Decision Matrix: Which Tool to Use?

```python
def choose_tool(requirements):
    """
    Decision helper for Hadoop ecosystem tools
    """
    if requirements['latency'] == 'real-time':
        if requirements['data_model'] == 'keyvalue':
            return "HBase"
        else:
            return "Spark Streaming"
    
    elif requirements['query_type'] == 'sql':
        if requirements['data_size'] == 'massive':
            return "Hive (batch) or Spark SQL (faster)"
        else:
            return "Traditional SQL database"
    
    elif requirements['task'] == 'etl':
        if requirements['complexity'] == 'simple':
            return "Pig"
        else:
            return "Spark"
    
    elif requirements['task'] == 'machine_learning':
        if requirements['algorithm'] in ['kmeans', 'recommendation']:
            return "Mahout or Spark MLlib"
        else:
            return "Spark MLlib"
    
    else:
        return "Consider Spark as general-purpose framework"
```

## Architecture Pattern: Lambda Architecture

Combines batch and streaming for comprehensive data processing:

```
Data Sources
    |
    ├─> Batch Layer (Hadoop/Spark Batch)
    |   └─> Batch Views (Hive tables)
    |
    ├─> Speed Layer (Spark Streaming)
    |   └─> Real-time Views (HBase)
    |
    └─> Serving Layer
        └─> Query = Batch Views + Real-time Views
```

## Modern Ecosystem Evolution

**Trend**: Many organizations are moving from traditional Hadoop ecosystem to:

1. **Spark-Centric**: Replace MapReduce, Pig, Hive with Spark/Spark SQL
2. **Cloud-Native**: Use managed services (AWS EMR, Google Dataproc, Azure HDInsight)
3. **Kubernetes**: Run Spark on Kubernetes instead of YARN
4. **Delta Lake/Iceberg**: Modern table formats replacing Hive

## Key Takeaways

:::{admonition} Summary
:class: note

1. **HDFS** provides distributed, fault-tolerant storage
2. **MapReduce** enables parallel batch processing
3. **YARN** manages cluster resources for multiple frameworks
4. **Pig** simplifies ETL with data flow scripts
5. **Hive** enables SQL queries on Hadoop data
6. **HBase** provides real-time NoSQL database capabilities
7. **Mahout** offers scalable machine learning algorithms
8. Modern trend: **Spark is replacing many traditional Hadoop tools**
:::

## Practical Exercise

**Scenario**: Design a data pipeline for an e-commerce platform

**Requirements**:
- Store clickstream data (100 GB/day)
- Daily aggregation of sales metrics
- Real-time user session tracking
- Product recommendation engine

**Your Task**:
1. Which Hadoop ecosystem components would you use?
2. Design the data flow architecture
3. Justify your technology choices

## Further Reading

- Apache Hadoop Documentation: [hadoop.apache.org](https://hadoop.apache.org)
- "Hadoop: The Definitive Guide" by Tom White
- "Programming Pig" by Alan Gates
- "Programming Hive" by Edward Capriolo et al.
- HBase: The Definitive Guide" by Lars George
