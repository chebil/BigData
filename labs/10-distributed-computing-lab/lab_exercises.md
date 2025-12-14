# Lab 10: Exercises - Apache Spark Distributed Computing

## Exercise Set 1: RDD Operations (Beginner)

### Exercise 1.1: Basic RDD Creation

Create an RDD from a Python collection and perform basic transformations:

```python
from pyspark import SparkContext

# Create a SparkContext
sc = SparkContext("local", "Exercise 1.1")

# Create RDD from collection
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(data, numPartitions=2)

# TODO: Implement the following
# 1. Map: Double each number
doubled = rdd.map(lambda x: x * 2)

# 2. Filter: Keep only numbers > 10
filtered = doubled.filter(lambda x: x > 10)

# 3. Collect results and print
result = filtered.collect()
print("Result:", result)

# 4. Count elements
count = filtered.count()
print(f"Count: {count}")
```

**Tasks**:
- [ ] Create and execute the RDD transformations
- [ ] Print the results
- [ ] Verify the count is correct
- [ ] Explain lazy evaluation

---

### Exercise 1.2: Word Count - Classic RDD Example

**Objective**: Implement the classic MapReduce word count problem using RDDs.

```python
# Data: List of sentences
sentences = [
    "Spark is a distributed computing framework",
    "Spark provides fast in-memory processing",
    "Spark supports multiple programming languages"
]

rdd = sc.parallelize(sentences)

# TODO: Implement word count
# 1. Split each sentence into words
words = rdd.flatMap(lambda sentence: sentence.lower().split())

# 2. Map each word to (word, 1)
word_pairs = words.map(lambda word: (word, 1))

# 3. Reduce by key to count
word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

# 4. Sort by count (descending)
sorted_counts = word_counts.sortByKey()

# 5. Collect and display
result = sorted_counts.collect()
for word, count in result:
    print(f"{word}: {count}")
```

**Tasks**:
- [ ] Implement the word count pipeline
- [ ] Display top 10 most frequent words
- [ ] Calculate total unique words
- [ ] Explain each transformation step

---

## Exercise Set 2: DataFrames & Spark SQL (Intermediate)

### Exercise 2.1: DataFrame Creation and Manipulation

**Objective**: Work with DataFrames and perform SQL-like operations.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count

spark = SparkSession.builder.appName("Exercise 2.1").getOrCreate()

# Create DataFrame from data
data = [
    ("Alice", 25, 80000),
    ("Bob", 30, 90000),
    ("Charlie", 35, 100000),
    ("Diana", 28, 85000),
    ("Eve", 32, 95000)
]

schema = ["name", "age", "salary"]
df = spark.createDataFrame(data, schema=schema)

# TODO: Implement the following
# 1. Display DataFrame
df.show()

# 2. Filter employees with salary > 85000
high_salary = df.filter(col("salary") > 85000)
print("High salary employees:")
high_salary.show()

# 3. Select specific columns
names_salaries = df.select("name", "salary")
print("Names and salaries:")
names_salaries.show()

# 4. Aggregate statistics
print("Salary statistics:")
df.agg(
    avg("salary").alias("avg_salary"),
    sum("salary").alias("total_salary"),
    count("*").alias("total_employees")
).show()
```

**Tasks**:
- [ ] Create and manipulate DataFrame
- [ ] Apply filtering operations
- [ ] Calculate aggregate statistics
- [ ] Write results to CSV

---

### Exercise 2.2: Spark SQL Queries

**Objective**: Use Spark SQL for data analysis.

```python
# Register temporary view
df.createOrReplaceTempView("employees")

# TODO: Implement SQL queries
# 1. Simple SELECT
result = spark.sql("SELECT * FROM employees WHERE age > 28")
result.show()

# 2. GROUP BY with aggregation
result = spark.sql("""
    SELECT 
        age,
        COUNT(*) as count,
        AVG(salary) as avg_salary
    FROM employees
    GROUP BY age
    ORDER BY age
""")
result.show()

# 3. JOIN operation (with self)
result = spark.sql("""
    SELECT 
        e1.name,
        e1.salary,
        e2.name as colleague_name
    FROM employees e1
    JOIN employees e2 ON e1.age = e2.age AND e1.name < e2.name
""")
result.show()
```

**Tasks**:
- [ ] Write 5 different SQL queries
- [ ] Use WHERE, GROUP BY, ORDER BY clauses
- [ ] Perform JOIN operations
- [ ] Explain query execution plans

---

## Exercise Set 3: MLlib - Machine Learning (Advanced)

### Exercise 3.1: Classification Pipeline with Spark MLlib

**Objective**: Build a distributed machine learning pipeline.

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Load data
df = spark.read.csv("data/customer_churn.csv", header=True, inferSchema=True)

# TODO: Implement ML pipeline
# 1. Prepare features
feature_cols = ["age", "tenure", "monthly_charges", "total_charges"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 2. Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# 3. Create classifier
lr = LogisticRegression(featuresCol="scaled_features", labelCol="churn", maxIter=10)

# 4. Create pipeline
pipeline = Pipeline(stages=[vector_assembler, scaler, lr])

# 5. Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 6. Train model
model = pipeline.fit(train_data)

# 7. Make predictions
predictions = model.transform(test_data)

# 8. Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.3f}")
```

**Tasks**:
- [ ] Build complete ML pipeline
- [ ] Train and evaluate model
- [ ] Calculate multiple metrics
- [ ] Explain distributed training advantages

---

## Exercise Set 4: Performance Optimization (Advanced)

### Exercise 4.1: Spark Job Optimization

**Objective**: Identify and implement performance optimizations.

```python
# TODO: Analyze and optimize the following code

# Version 1: Inefficient (multiple shuffles)
df1 = spark.read.csv("data/large_dataset.csv", header=True)
df1 = df1.filter(col("value") > 100)
df1.cache()  # Cache intermediate result
df1.write.mode("overwrite").csv("output/v1")

# Version 2: Optimized (single pass)
df2 = spark.read.csv("data/large_dataset.csv", header=True)
df2 = df2.filter(col("value") > 100) \
    .select("id", "value") \
    .cache()
df2.write.mode("overwrite").csv("output/v2")

# Exercise tasks:
# 1. Compare execution times
# 2. Check DAG (Directed Acyclic Graph) visualization
# 3. Implement partitioning strategy
df2 = df2.repartition(4)  # Adjust based on cluster size

# 4. Use broadcast variables for small datasets
small_data = {...}
broadcast_var = spark.broadcast(small_data)
```

**Tasks**:
- [ ] Profile job execution
- [ ] Identify bottlenecks
- [ ] Apply caching strategically
- [ ] Optimize partitioning

---

## Exercise Set 5: Real-World Applications (Capstone)

### Exercise 5.1: E-Commerce Analytics Pipeline

**Objective**: Build a complete Spark pipeline for e-commerce analytics.

```python
# Load transaction data
transactions = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)
customers = spark.read.csv("data/customers.csv", header=True, inferSchema=True)
products = spark.read.csv("data/products.csv", header=True, inferSchema=True)

# TODO: Implement comprehensive analysis

# 1. Join datasets
analysis = transactions.join(customers, "customer_id") \
    .join(products, "product_id")

# 2. Calculate metrics
from pyspark.sql.window import Window

window_spec = Window.partitionBy("customer_id").orderBy(col("date"))
analysis = analysis.withColumn(
    "cumulative_spend",
    sum("amount").over(window_spec)
)

# 3. Customer segmentation
customer_stats = analysis.groupBy("customer_id") \
    .agg(
        count("transaction_id").alias("num_purchases"),
        sum("amount").alias("total_spend"),
        avg("amount").alias("avg_spend")
    )

# 4. Product recommendations (using association rules)
from pyspark.ml.fpm import FPGrowth

fpgrowth = FPGrowth(itemsCol="items", minSupport=0.1)
model = fpgrowth.fit(transactions)
recommendations = model.associationRules

# 5. Save results
customer_stats.write.mode("overwrite").parquet("output/customer_segments")
recommendations.write.mode("overwrite").parquet("output/recommendations")
```

**Tasks**:
- [ ] Load and join multiple datasets
- [ ] Calculate customer lifetime value
- [ ] Generate product recommendations
- [ ] Create visualizations of results
- [ ] Document insights and findings

---

## Submission Checklist

Before submitting your work:

- [ ] All exercises completed
- [ ] Code runs without errors
- [ ] Results are documented
- [ ] Visualizations included
- [ ] Performance optimizations applied
- [ ] Jupyter notebook is clean and organized
- [ ] Comments explain key code sections
- [ ] Summary of findings written

---

## Grading Criteria

- **Exercise Completion**: 40%
- **Code Quality**: 20%
- **Results & Analysis**: 20%
- **Optimization Efforts**: 10%
- **Documentation**: 10%

---

**Total Points**: 100

Good luck! 🚀
