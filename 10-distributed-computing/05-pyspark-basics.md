# PySpark Basics

## Learning Objectives

- Set up PySpark environment
- Create and manipulate Spark DataFrames
- Perform data transformations and aggregations
- Read and write data in multiple formats
- Optimize PySpark code for performance

## Introduction

PySpark is the Python API for Apache Spark, making distributed computing accessible to Python developers. This section provides hands-on experience with PySpark for real-world data processing tasks.

## Installation and Setup

### Local Installation

```bash
# Install PySpark
pip install pyspark

# Optional: Install additional libraries
pip install pyspark[sql]
pip install pandas numpy matplotlib
```

### Verify Installation

```python
import pyspark
print(pyspark.__version__)

# Create simple Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Test") \
    .getOrCreate()

print(spark.version)
spark.stop()
```

### Jupyter Notebook Setup

```python
# Configure Spark for Jupyter
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("JupyterNotebook") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

## Creating SparkSession

**SparkSession** is the entry point for all Spark functionality.

```python
from pyspark.sql import SparkSession

# Basic session
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Session with configuration
spark = SparkSession.builder \
    .appName("ConfiguredApp") \
    .master("local[4]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .enableHiveSupport() \
    .getOrCreate()

# Access SparkContext
sc = spark.sparkContext
print(f"Spark version: {spark.version}")
print(f"Master: {sc.master}")
```

## Creating DataFrames

### From Python Collections

```python
# From list of tuples
data = [
    ("Alice", 25, "Engineering"),
    ("Bob", 30, "Sales"),
    ("Carol", 28, "Engineering"),
    ("David", 35, "Marketing")
]

df = spark.createDataFrame(data, ["name", "age", "department"])
df.show()
df.printSchema()
```

### From Pandas DataFrame

```python
import pandas as pd
import numpy as np

# Create pandas DataFrame
pandas_df = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'purchase_amount': np.random.uniform(10, 500, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(pandas_df)
spark_df.show(5)
```

### Reading from Files

```python
# CSV
df_csv = spark.read.csv(
    "data/customers.csv",
    header=True,
    inferSchema=True,
    sep=","
)

# JSON
df_json = spark.read.json("data/logs.json")

# Parquet (recommended for Spark)
df_parquet = spark.read.parquet("data/sales.parquet")

# With options
df_csv_advanced = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("dateFormat", "yyyy-MM-dd") \
    .option("mode", "DROPMALFORMED") \
    .csv("data/transactions.csv")

# Multiple files
df_multi = spark.read.csv(["data/file1.csv", "data/file2.csv"], header=True)

# Wildcard patterns
df_pattern = spark.read.parquet("data/year=2024/month=*/day=*/*.parquet")
```

## DataFrame Operations

### Schema and Structure

```python
# Show schema
df.printSchema()

# Get column names
print(df.columns)

# Get data types
print(df.dtypes)

# Count rows and columns
print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")

# Show first n rows
df.show(10)

# Show without truncation
df.show(10, truncate=False)

# Get first n rows as Python list
rows = df.head(5)
for row in rows:
    print(row)
```

### Selecting and Filtering

```python
from pyspark.sql.functions import col, column

# Select columns
df.select("name", "age").show()

# Multiple ways to select
df.select(df.name, df["age"]).show()
df.select(col("name"), column("age")).show()

# Select all columns
df.select("*").show()

# Filter rows
df.filter(df.age > 25).show()
df.filter("age > 25").show()  # SQL expression
df.filter((col("age") > 25) & (col("department") == "Engineering")).show()

# Where is alias for filter
df.where(df.age > 25).show()

# Select distinct
df.select("department").distinct().show()

# Drop duplicates
df.dropDuplicates(["department"]).show()
```

### Adding and Removing Columns

```python
from pyspark.sql.functions import lit, expr, when

# Add column with literal value
df_with_country = df.withColumn("country", lit("USA"))

# Add computed column
df_with_bonus = df.withColumn("bonus", col("age") * 100)

# Add conditional column
df_with_category = df.withColumn(
    "age_category",
    when(col("age") < 30, "Young")
    .when(col("age") < 40, "Middle")
    .otherwise("Senior")
)

# Rename column
df_renamed = df.withColumnRenamed("name", "employee_name")

# Drop column
df_dropped = df.drop("department")

# Drop multiple columns
df_dropped_multi = df.drop("department", "age")
```

### Aggregations

```python
from pyspark.sql.functions import sum, avg, count, min, max, stddev

# Simple aggregations
df.agg({"age": "avg"}).show()

# Multiple aggregations
df.agg(
    avg("age").alias("avg_age"),
    min("age").alias("min_age"),
    max("age").alias("max_age"),
    count("*").alias("total_count")
).show()

# Group by
df.groupBy("department").count().show()

df.groupBy("department").agg(
    avg("age").alias("avg_age"),
    count("*").alias("count")
).show()

# Multiple grouping columns
df.groupBy("department", "age_category").count().show()
```

### Sorting

```python
from pyspark.sql.functions import desc, asc

# Sort ascending
df.orderBy("age").show()

# Sort descending
df.orderBy(desc("age")).show()
df.orderBy(col("age").desc()).show()

# Multiple columns
df.orderBy("department", desc("age")).show()

# Sort is alias for orderBy
df.sort(desc("age")).show()
```

## Common Functions

```python
from pyspark.sql.functions import *

# String functions
df.select(
    upper(col("name")).alias("upper_name"),
    lower(col("name")).alias("lower_name"),
    length(col("name")).alias("name_length"),
    concat(col("name"), lit(" - "), col("department")).alias("combined")
).show()

# Date/Time functions
from datetime import datetime

date_df = spark.createDataFrame([
    (1, "2024-01-15"),
    (2, "2024-02-20"),
    (3, "2024-03-10")
], ["id", "date_str"])

date_df.select(
    col("date_str"),
    to_date(col("date_str")).alias("date"),
    year(to_date(col("date_str"))).alias("year"),
    month(to_date(col("date_str"))).alias("month"),
    dayofmonth(to_date(col("date_str"))).alias("day"),
    dayofweek(to_date(col("date_str"))).alias("dow")
).show()

# Null handling
df.select(
    col("age"),
    when(col("age").isNull(), 0).otherwise(col("age")).alias("age_filled"),
    coalesce(col("age"), lit(25)).alias("age_coalesced")
).show()

# Fill nulls
df.fillna(0).show()  # Fill all nulls with 0
df.fillna({"age": 0, "name": "Unknown"}).show()  # Fill specific columns

# Drop nulls
df.dropna().show()  # Drop rows with any null
df.dropna(subset=["age"]).show()  # Drop rows with null in specific columns
```

## Joins

```python
# Create two DataFrames
employees = spark.createDataFrame([
    (1, "Alice", "Engineering"),
    (2, "Bob", "Sales"),
    (3, "Carol", "Engineering")
], ["emp_id", "name", "dept_name"])

departments = spark.createDataFrame([
    ("Engineering", "Building A"),
    ("Sales", "Building B"),
    ("Marketing", "Building C")
], ["dept_name", "location"])

# Inner join (default)
inner_join = employees.join(departments, on="dept_name", how="inner")
inner_join.show()

# Left outer join
left_join = employees.join(departments, on="dept_name", how="left")
left_join.show()

# Right outer join
right_join = employees.join(departments, on="dept_name", how="right")
right_join.show()

# Full outer join
full_join = employees.join(departments, on="dept_name", how="outer")
full_join.show()

# Join on different column names
df1 = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
df2 = spark.createDataFrame([(1, "X"), (2, "Y")], ["idx", "data"])
joined = df1.join(df2, df1.id == df2.idx)
joined.show()
```

## Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead

# Sample data: sales by employee and month
sales_data = [
    ("Alice", "2024-01", 1000),
    ("Alice", "2024-02", 1500),
    ("Alice", "2024-03", 1200),
    ("Bob", "2024-01", 800),
    ("Bob", "2024-02", 900),
    ("Bob", "2024-03", 1100)
]

sales_df = spark.createDataFrame(sales_data, ["employee", "month", "sales"])

# Define window
window_spec = Window.partitionBy("employee").orderBy("month")

# Apply window functions
result = sales_df.select(
    col("employee"),
    col("month"),
    col("sales"),
    row_number().over(window_spec).alias("row_num"),
    rank().over(window_spec).alias("rank"),
    lag(col("sales"), 1).over(window_spec).alias("prev_month_sales"),
    lead(col("sales"), 1).over(window_spec).alias("next_month_sales"),
    sum(col("sales")).over(window_spec).alias("running_total")
)

result.show()
```

## Writing Data

```python
# Write to CSV
df.write.csv("output/employees.csv", header=True, mode="overwrite")

# Write to Parquet (recommended)
df.write.parquet("output/employees.parquet", mode="overwrite")

# Write to JSON
df.write.json("output/employees.json", mode="overwrite")

# Partitioned write
df.write.partitionBy("department").parquet("output/employees_partitioned")

# Write modes
df.write.mode("overwrite").parquet("output/data")  # Overwrite existing
df.write.mode("append").parquet("output/data")     # Append to existing
df.write.mode("ignore").parquet("output/data")     # Ignore if exists
df.write.mode("error").parquet("output/data")      # Error if exists (default)
```

## Performance Optimization

### Caching

```python
# Cache for reuse
df_cached = df.cache()
# or
df.persist()

# Use cached DataFrame
count1 = df_cached.count()
count2 = df_cached.filter("age > 30").count()

# Release cache
df_cached.unpersist()
```

### Repartitioning

```python
# Check current partitions
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Repartition (shuffle data)
df_repart = df.repartition(10)

# Repartition by column (for joins)
df_repart_col = df.repartition(10, "department")

# Coalesce (reduce partitions without shuffle)
df_coalesce = df.coalesce(2)
```

### Broadcast Joins

```python
from pyspark.sql.functions import broadcast

# For joining small table with large table
small_df = spark.read.csv("data/small_lookup.csv", header=True)
large_df = spark.read.parquet("data/large_fact_table.parquet")

# Broadcast small table
result = large_df.join(broadcast(small_df), on="key")
```

## Complete Example: Sales Analysis

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Initialize
spark = SparkSession.builder.appName("SalesAnalysis").getOrCreate()

# Load data
sales = spark.read.parquet("data/sales.parquet")

# Data exploration
print(f"Total records: {sales.count()}")
sales.printSchema()
sales.show(5)

# Data cleaning
sales_clean = sales \
    .dropna(subset=["order_id", "customer_id", "amount"]) \
    .filter(col("amount") > 0) \
    .withColumn("order_date", to_date(col("order_date")))

# Feature engineering
sales_enhanced = sales_clean \
    .withColumn("year", year(col("order_date"))) \
    .withColumn("month", month(col("order_date"))) \
    .withColumn("amount_category",
                when(col("amount") < 50, "Low")
                .when(col("amount") < 200, "Medium")
                .otherwise("High"))

# Analysis 1: Revenue by product category
revenue_by_category = sales_enhanced \
    .groupBy("product_category") \
    .agg(
        sum("amount").alias("total_revenue"),
        count("*").alias("order_count"),
        avg("amount").alias("avg_order_value")
    ) \
    .orderBy(desc("total_revenue"))

revenue_by_category.show()

# Analysis 2: Top customers
top_customers = sales_enhanced \
    .groupBy("customer_id") \
    .agg(
        sum("amount").alias("total_spent"),
        count("*").alias("order_count")
    ) \
    .orderBy(desc("total_spent")) \
    .limit(10)

top_customers.show()

# Analysis 3: Monthly trends
monthly_trends = sales_enhanced \
    .groupBy("year", "month") \
    .agg(sum("amount").alias("monthly_revenue")) \
    .orderBy("year", "month")

monthly_trends.show()

# Save results
revenue_by_category.write.parquet("output/revenue_by_category", mode="overwrite")
top_customers.write.parquet("output/top_customers", mode="overwrite")
monthly_trends.write.parquet("output/monthly_trends", mode="overwrite")

spark.stop()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **SparkSession** is the entry point for PySpark applications
2. DataFrames support **SQL-like operations** with optimization
3. Use **caching** for DataFrames accessed multiple times
4. **Repartitioning** optimizes parallel processing
5. **Broadcast joins** speed up joins with small tables
6. **Window functions** enable advanced analytics
7. **Parquet format** is optimal for Spark workloads
:::

## Further Reading

- [PySpark API Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
- Chambers, B. & Zaharia, M. (2018). "Spark: The Definitive Guide"
