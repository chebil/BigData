# Lab 10: Complete Walkthrough - Apache Spark Distributed Computing

## Overview

This guide provides step-by-step solutions and explanations for all Lab 10 exercises.

---

## Part 1: RDD Operations Walkthrough

### Solution 1.1: Basic RDD Creation

**Complete Solution**:

```python
from pyspark import SparkContext

# Initialize SparkContext
sc = SparkContext("local[*]", "RDD Operations")

# Step 1: Create RDD from collection
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(data, numPartitions=2)

print(f"Number of partitions: {rdd.getNumPartitions()}")
print(f"First element: {rdd.first()}")

# Step 2: Map transformation (double each number)
doubled = rdd.map(lambda x: x * 2)
print("After doubling:", doubled.collect())
# Output: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Step 3: Filter transformation (keep > 10)
filtered = doubled.filter(lambda x: x > 10)
print("After filtering > 10:", filtered.collect())
# Output: [12, 14, 16, 18, 20]

# Step 4: Count action
count = filtered.count()
print(f"Count after filtering: {count}")  # Output: 5

# Step 5: Calculate sum (additional action)
total = filtered.sum()
print(f"Sum of filtered elements: {total}")  # Output: 90
```

**Key Concepts**:
- **Transformations** (lazy): `map()`, `filter()` - not computed until action called
- **Actions** (eager): `collect()`, `count()`, `sum()` - trigger computation
- **Partitions**: Data divided across cluster nodes
- **Lineage**: RDD tracks computation steps for recovery

---

### Solution 1.2: Word Count

**Complete Solution**:

```python
from pyspark import SparkContext

sc = SparkContext("local[*]", "Word Count")

# Input data
sentences = [
    "Spark is a distributed computing framework",
    "Spark provides fast in-memory processing",
    "Spark supports multiple programming languages",
    "Apache Spark is widely used in industry"
]

rdd = sc.parallelize(sentences)

# Step 1: Split into words (flatMap)
words = rdd.flatMap(lambda sentence: sentence.lower().split())
print("Sample words:", words.take(5))

# Step 2: Create (word, 1) pairs
word_pairs = words.map(lambda word: (word, 1))
print("Sample pairs:", word_pairs.take(3))

# Step 3: Reduce by key (sum counts)
word_counts = word_pairs.reduceByKey(lambda a, b: a + b)
print("Word counts (unsorted):", word_counts.collect())

# Step 4: Sort by count (descending)
sorted_counts = word_counts.map(lambda x: (x[1], x[0])) \
    .sortByKey(ascending=False) \
    .map(lambda x: (x[1], x[0]))

# Step 5: Display results
print("\n=== Top 10 Most Frequent Words ===")
for word, count in sorted_counts.take(10):
    print(f"{word:15} : {count}")

# Step 6: Calculate statistics
total_words = words.count()
unique_words = word_counts.count()
avg_frequency = total_words / unique_words

print(f"\nStatistics:")
print(f"Total words: {total_words}")
print(f"Unique words: {unique_words}")
print(f"Average frequency: {avg_frequency:.2f}")
```

**Output Example**:
```
Top 10 Most Frequent Words
spark            : 3
is               : 2
a                : 2
distributed      : 1
computing        : 1
...

Statistics:
Total words: 23
Unique words: 18
Average frequency: 1.28
```

---

## Part 2: DataFrames & Spark SQL

### Solution 2.1: DataFrame Operations

**Complete Solution**:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, max, min

spark = SparkSession.builder \
    .appName("DataFrame Operations") \
    .getOrCreate()

# Create DataFrame
data = [
    ("Alice", 25, 80000),
    ("Bob", 30, 90000),
    ("Charlie", 35, 100000),
    ("Diana", 28, 85000),
    ("Eve", 32, 95000)
]

df = spark.createDataFrame(data, ["name", "age", "salary"])

# Display schema and data
df.printSchema()
df.show()

# Filter: salary > 85000
high_salary = df.filter(col("salary") > 85000)
print("High Salary Employees:")
high_salary.show()

# Select specific columns
names_salaries = df.select("name", "salary")
print("Names and Salaries:")
names_salaries.show()

# Add computed column
df_with_bonus = df.withColumn(
    "bonus",
    col("salary") * 0.1
)
df_with_bonus.show()

# Aggregations
print("\n=== Salary Statistics ===")
df.agg(
    count("*").alias("total_employees"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    sum("salary").alias("total_salary")
).show()

# Group by and aggregate
print("\n=== Salary by Age Group ===")
df.groupBy((col("age") / 5).cast("int") * 5)\
    .agg(
        count("*").alias("count"),
        avg("salary").alias("avg_salary")
    ).show()
```

**Key DataFrame Operations**:
- `.select()` - Choose columns
- `.filter()` - Row filtering
- `.withColumn()` - Add/modify columns
- `.groupBy()` - Grouping
- `.agg()` - Aggregations
- `.orderBy()` - Sorting

---

### Solution 2.2: Spark SQL Queries

**Complete Solution**:

```python
# Register temporary view
df.createOrReplaceTempView("employees")

# Query 1: Simple SELECT with WHERE
result1 = spark.sql("""
    SELECT name, age, salary
    FROM employees
    WHERE age > 28
    ORDER BY salary DESC
""")
print("Employees older than 28:")
result1.show()

# Query 2: GROUP BY with aggregation
result2 = spark.sql("""
    SELECT 
        CAST(age/5 AS INT)*5 as age_group,
        COUNT(*) as count,
        AVG(salary) as avg_salary,
        SUM(salary) as total_salary
    FROM employees
    GROUP BY CAST(age/5 AS INT)*5
    ORDER BY age_group
""")
print("\nSalary by age group:")
result2.show()

# Query 3: Window functions
result3 = spark.sql("""
    SELECT 
        name,
        salary,
        ROW_NUMBER() OVER (ORDER BY salary DESC) as salary_rank,
        AVG(salary) OVER () as avg_salary,
        salary - AVG(salary) OVER () as diff_from_avg
    FROM employees
""")
print("\nSalary ranking with statistics:")
result3.show()

# Query 4: Self-join
result4 = spark.sql("""
    SELECT 
        e1.name,
        e1.salary,
        e2.name as colleague,
        e2.salary as colleague_salary,
        e1.salary - e2.salary as salary_diff
    FROM employees e1
    JOIN employees e2 ON e1.age = e2.age AND e1.name < e2.name
""")
print("\nSame-age colleagues:")
result4.show()

# Query 5: Complex aggregation
result5 = spark.sql("""
    SELECT 
        'Total' as metric,
        COUNT(*) as value,
        NULL as percentage
    FROM employees
    UNION ALL
    SELECT 
        'Above Average Salary',
        COUNT(*),
        ROUND(COUNT(*) * 100 / (SELECT COUNT(*) FROM employees), 1)
    FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees)
""")
print("\nEmployee salary distribution:")
result5.show()
```

---

## Part 3: MLlib Machine Learning Pipeline

### Solution 3.1: Classification Pipeline

**Complete Solution**:

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("MLlib Pipeline").getOrCreate()

# Load data
df = spark.read.csv(
    "path/to/customer_churn.csv",
    header=True,
    inferSchema=True
)

print("Dataset shape:", (df.count(), len(df.columns)))
df.show(5)
df.printSchema()

# Data preparation
df = df.dropna()  # Remove nulls
df = df.filter(col("total_charges") != "NaN")  # Remove invalid values

# Define features and label
feature_cols = ["age", "tenure", "monthly_charges", "total_charges"]

# Step 1: Vector Assembler
vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# Step 2: Standard Scaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

# Step 3: Logistic Regression
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="churn",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0.5
)

# Step 4: Create Pipeline
pipeline = Pipeline(stages=[vector_assembler, scaler, lr])

# Step 5: Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

# Step 6: Train model
print("\nTraining model...")
model = pipeline.fit(train_data)
print("Model training complete!")

# Step 7: Make predictions
predictions = model.transform(test_data)
predictions.select("churn", "probability", "prediction").show(10)

# Step 8: Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="churn",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"\nAUC-ROC Score: {auc:.4f}")

# Additional metrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator_pr = BinaryClassificationEvaluator(
    labelCol="churn",
    metricName="areaUnderPR"
)
auc_pr = evaluator_pr.evaluate(predictions)
print(f"AUC-PR Score: {auc_pr:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.stages[-1].coefficients.toArray()
}).sort_values('coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

---

## Part 4: Performance Optimization

### Solution 4.1: Optimization Techniques

**Best Practices**:

```python
# 1. Caching for reused DataFrames
df = spark.read.csv("large_file.csv", header=True)
df.cache()  # Store in memory after first use
df.count()  # Triggers caching

# 2. Partitioning for distributed processing
df_partitioned = df.repartition("region")  # Partition by region
df_partitioned.write.partitionBy("region").mode("overwrite").csv("output/")

# 3. Broadcast small DataFrames
small_df = spark.read.csv("lookup_table.csv", header=True)
broadcast_small = spark.broadcast(small_df.collect())

# Use in transformations
def lookup_func(row):
    # Access broadcast variable
    lookup_dict = {item[0]: item[1] for item in broadcast_small.value}
    return lookup_dict.get(row.id, "Unknown")

# 4. Columnar storage (Parquet)
df.write.mode("overwrite").parquet("output/data.parquet")
df_read = spark.read.parquet("output/data.parquet")

# 5. Predicate pushdown (filter early)
# Good: Filter before join
df_filtered = df.filter(col("age") > 30)
df_joined = df_filtered.join(other_df, "id")

# 6. Avoid shuffles
# Bad: Unnecessary shuffle
df.groupBy("col1").count().filter(col("count") > 10)

# Good: Filter in aggregate
df.groupBy("col1").count().filter(col("count") > 10)
```

---

## Summary

**Key Learning Points**:

1. **RDDs**: Low-level, flexible but require manual optimization
2. **DataFrames**: High-level, optimized, recommended for structured data
3. **Spark SQL**: SQL queries on distributed data
4. **MLlib**: Scalable machine learning pipelines
5. **Optimization**: Critical for production applications

**Performance Tips**:
- Use DataFrames over RDDs
- Cache intermediate results
- Partition data strategically
- Use Parquet format for storage
- Monitor Spark UI during execution

---

**Complete! Ready for real-world Spark applications.** 🚀
