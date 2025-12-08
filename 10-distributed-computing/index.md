# Chapter 10: Distributed Computing with Hadoop and Spark

## Introduction

Distributed computing enables processing of massive datasets by distributing workloads across clusters of computers. This chapter explores the fundamental technologies that power Big Data analytics at scale: Apache Hadoop and Apache Spark.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the MapReduce programming paradigm
- Explain Hadoop Distributed File System (HDFS) architecture
- Work with Apache Spark and PySpark
- Process large-scale data using distributed computing frameworks
- Apply Spark SQL and MLlib for analytics
- Choose appropriate tools for different Big Data scenarios

## Chapter Overview

This chapter covers:

1. **Hadoop Ecosystem** - HDFS, MapReduce, YARN, and related tools
2. **HDFS Architecture** - Distributed storage fundamentals
3. **MapReduce Paradigm** - Parallel processing concepts
4. **Apache Spark** - Next-generation distributed computing
5. **PySpark Basics** - Python interface for Spark
6. **Spark SQL** - Structured data processing
7. **Spark MLlib** - Distributed machine learning

## Why Distributed Computing?

Traditional single-machine processing hits fundamental limits:

- **Memory constraints**: Cannot load terabytes into RAM
- **CPU limits**: Single processors cannot handle massive computations
- **I/O bottlenecks**: Disk and network bandwidth limitations
- **Scalability**: Vertical scaling (bigger machines) becomes exponentially expensive

**Solution**: Horizontal scaling - distribute data and computation across many machines.

## Key Technologies

### Apache Hadoop

- **HDFS**: Distributed file system for storing massive datasets
- **MapReduce**: Programming model for parallel data processing
- **YARN**: Resource manager for cluster computing
- **Ecosystem**: Pig, Hive, HBase, Mahout, and more

### Apache Spark

- **In-memory processing**: 10-100x faster than Hadoop MapReduce
- **Unified framework**: Batch, streaming, SQL, ML, graph processing
- **PySpark**: Python API for Spark
- **RDDs and DataFrames**: Core abstractions for distributed data

## Real-World Use Cases

### IBM Watson (Hadoop)

Processed encyclopedias, dictionaries, news feeds, and Wikipedia to compete on Jeopardy! Hadoop enabled:
- Parallel processing of terabytes of text data
- Fast search across distributed knowledge bases
- Real-time query response in under 3 seconds

### LinkedIn (Hadoop)

- Process daily production database logs
- Analyze user activities (views, clicks)
- Feed processed data to production systems
- Develop and test analytical models

### Yahoo! (Hadoop)

One of the largest Hadoop deployments (2012): 42,000 nodes, 350 petabytes
- Search index creation and maintenance
- Web ad placement optimization
- Spam filtering
- Reduced processing time from 26 days to 20 minutes

### Uber (Spark)

- Real-time pricing and surge detection
- Route optimization
- Driver-rider matching
- Processing billions of trips

### Netflix (Spark)

- Recommendation engine
- Streaming analytics
- A/B testing at scale
- Content personalization

## The MapReduce Paradigm

**Core Idea**: Break large tasks into smaller tasks, process in parallel, consolidate results.

```python
# Conceptual MapReduce flow
def mapper(document):
    """Map: Extract key-value pairs"""
    for word in document.split():
        emit(word, 1)

def reducer(word, counts):
    """Reduce: Aggregate values for each key"""
    emit(word, sum(counts))
```

**Advantages**:
- Automatic parallelization
- Fault tolerance
- Data locality optimization
- Scalability to thousands of nodes

## Comparison: Hadoop vs. Spark

| Feature | Hadoop MapReduce | Apache Spark |
|---------|-----------------|-------------|
| **Speed** | Disk-based | In-memory (10-100x faster) |
| **Ease of Use** | Complex Java code | Simple Python/Scala APIs |
| **Processing** | Batch only | Batch, streaming, interactive |
| **Fault Tolerance** | Replication | Lineage-based recovery |
| **Learning Curve** | Steep | Moderate |
| **Use Cases** | Large batch ETL | Iterative ML, real-time, interactive |

## When to Use Each

### Choose Hadoop When:
- Processing massive batch workloads (petabytes)
- Cost is primary concern (commodity hardware)
- Existing Hadoop infrastructure
- Simple map-reduce transformations

### Choose Spark When:
- Iterative algorithms (machine learning)
- Interactive data exploration
- Real-time stream processing
- Complex multi-stage pipelines
- Fast development cycles

## Chapter Structure

The following sections provide hands-on experience with:

1. **HDFS fundamentals** - Understanding distributed storage
2. **MapReduce programming** - Writing distributed algorithms
3. **Spark architecture** - RDDs, DataFrames, and transformations
4. **PySpark development** - Practical distributed computing
5. **Spark SQL** - Structured data analysis
6. **MLlib** - Scalable machine learning

## Prerequisites

Before proceeding, ensure familiarity with:
- Python programming (pandas, numpy)
- Basic SQL
- Linux command line
- Data structures and algorithms

## Setup Requirements

For hands-on exercises, you'll need:
- Docker (recommended for local development)
- PySpark installation
- Jupyter notebooks
- Sample datasets (provided in `data/` directory)

See `labs/lab-00-environment-setup/` for detailed installation instructions.

## Key Takeaways

:::{admonition} Summary
:class: note

1. Distributed computing enables processing of datasets too large for single machines
2. Hadoop provides reliable distributed storage (HDFS) and batch processing (MapReduce)
3. Spark offers faster in-memory processing with richer APIs
4. PySpark makes distributed computing accessible to Python developers
5. Choose tools based on use case: Hadoop for massive batch jobs, Spark for speed and flexibility
:::

## Next Steps

Proceed to the following sections to gain hands-on experience:
- **Section 10.1**: Hadoop Ecosystem overview
- **Section 10.2**: HDFS Architecture deep dive
- **Section 10.3**: MapReduce Programming
- **Section 10.4**: Apache Spark fundamentals
- **Section 10.5**: PySpark Basics
- **Section 10.6**: Spark SQL
- **Section 10.7**: Spark MLlib

## Further Reading

- White, T. (2015). "Hadoop: The Definitive Guide" (4th Edition)
- Karau, H., et al. (2015). "Learning Spark"
- Zaharia, M., et al. (2016). "Apache Spark: A Unified Engine for Big Data Processing"
- Dean, J., & Ghemawat, S. (2004). "MapReduce: Simplified Data Processing on Large Clusters"
