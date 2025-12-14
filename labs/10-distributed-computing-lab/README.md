# Lab 10: Apache Spark - Distributed Computing

## Overview

In this lab, you will work with **Apache Spark** to perform large-scale data processing and machine learning on distributed systems. You'll learn how to work with RDDs, DataFrames, Spark SQL, and MLlib for building scalable data pipelines.

## Learning Objectives

After completing this lab, you will be able to:

- ✅ Set up and configure Apache Spark environment
- ✅ Work with Spark RDDs and transformations/actions
- ✅ Use Spark DataFrames for structured data processing
- ✅ Write Spark SQL queries for big data analysis
- ✅ Implement distributed machine learning using Spark MLlib
- ✅ Optimize Spark jobs for performance
- ✅ Deploy Spark applications in production environments

## Prerequisites

- Completion of Labs 1-9
- Understanding of Python and SQL
- Knowledge of basic machine learning concepts
- Familiarity with distributed computing concepts

## Lab Structure

### Part 1: Spark Fundamentals
- RDD creation and transformations
- Actions and lazy evaluation
- Spark Context configuration
- Cluster vs. Local mode

### Part 2: Spark DataFrames
- Creating DataFrames from various sources
- DataFrame operations and optimizations
- Handling structured and semi-structured data
- Performance tuning

### Part 3: Spark SQL
- SQL queries on DataFrames
- Temporary views and databases
- Window functions and aggregations
- Query optimization

### Part 4: Spark MLlib
- Distributed machine learning pipelines
- Classification with large datasets
- Clustering at scale
- Feature engineering for distributed computing

## Datasets

- **Large Retail Dataset**: Transaction data (~1GB+)
- **Time Series Data**: Stock prices over multiple years
- **Log Files**: Web server logs for analysis
- **Customer Interactions**: E-commerce click stream data

## Time Estimate

**Total Duration**: 3-4 hours

- Setup and RDD basics: 45 minutes
- DataFrames: 45 minutes
- Spark SQL: 45 minutes
- MLlib and optimization: 60 minutes

## Getting Started

### 1. Start Spark in Docker

```bash
docker-compose up -d
docker ps
```

Access Spark UI at http://localhost:8080

### 2. Open Jupyter Lab

Navigate to http://localhost:8888 (password: bigdata)

### 3. Start with the Interactive Notebook

Open: `labs/10-distributed-computing-lab/lab10_spark_interactive.ipynb`

## Learning Path

1. **Read**: Chapter 10 - Distributed Computing
2. **Explore**: Complete walkthrough
3. **Practice**: Interactive notebook
4. **Challenge**: Lab exercises
5. **Optimize**: Performance optimization

## Key Topics

### RDD Basics
```python
from pyspark import SparkContext
sc = SparkContext("local", "RDD Example")
rdd = sc.parallelize([1, 2, 3, 4, 5])
mapped = rdd.map(lambda x: x * 2)
result = mapped.collect()
```

### DataFrames
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Example").getOrCreate()
df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.select("name", "age").filter(df.age > 25).show()
```

### Spark SQL
```python
df.createOrReplaceTempView("users")
result = spark.sql("SELECT age, COUNT(*) FROM users GROUP BY age")
result.show()
```

### MLlib
```python
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=10)
model = lr.fit(training_data)
predictions = model.transform(test_data)
```

## Grading Rubric

| Criterion | Excellent (10) | Good (8) | Fair (6) | Poor (4) |
|-----------|---------------|----------|----------|----------|
| **RDD Operations** | Perfect execution | Correct implementation | Minor issues | Multiple errors |
| **DataFrame Usage** | Efficient operations | Correct operations | Functional but inefficient | Incomplete |
| **SQL Queries** | Complex queries | Standard queries | Basic queries | Incorrect |
| **MLlib Pipeline** | Full distributed ML | Complete implementation | Partial | Incomplete |
| **Performance** | Optimized execution | Good performance | Acceptable | Slow |
| **Code Quality** | Clean, documented | Good structure | Acceptable | Poor |

## Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)

## Questions?

Refer to:
- `complete_walkthrough.md` for solutions
- `spark_optimization_guide.md` for performance tips
- **Chapter 10** in course materials

---

**Happy Spark coding! 🚀**
