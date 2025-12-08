# Spark SQL and Structured Data Processing

## Learning Objectives

- Query structured data using Spark SQL
- Understand Catalyst optimizer
- Work with different data sources (Parquet, JSON, JDBC)
- Perform complex analytics with SQL
- Optimize Spark SQL queries

## Introduction

Spark SQL is Spark's module for working with structured data. It provides a programming interface to query data using SQL or DataFrame API, with automatic optimizations through the Catalyst query optimizer.

## Why Spark SQL?

### Advantages

1. **Familiar SQL Interface**: Use SQL for Big Data analytics
2. **Performance**: Catalyst optimizer + Tungsten execution engine
3. **Unified API**: Same code works on different data sources
4. **Integration**: Seamless with Spark ecosystem (MLlib, Streaming)
5. **Schema Enforcement**: Type safety and validation

### Spark SQL vs. Hive

| Feature | Hive | Spark SQL |
|---------|------|----------|
| **Execution** | MapReduce | In-memory |
| **Speed** | Slow | 10-100x faster |
| **Latency** | Minutes | Seconds |
| **Use Case** | Batch ETL | Interactive + Batch |
| **ACID** | Limited | Better support |

## SparkSession: Entry Point

```python
from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("SparkSQLExample") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .enableHiveSupport() \
    .getOrCreate()

# Access SQL context
sqlContext = spark.sql

# Access Spark context
sc = spark.sparkContext
```

## Reading Data

### CSV Files

```python
# Basic read
df = spark.read.csv("data/sales.csv", header=True, inferSchema=True)

# Advanced options
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ",") \
    .option("nullValue", "NA") \
    .option("dateFormat", "yyyy-MM-dd") \
    .option("mode", "DROPMALFORMED") \
    .load("data/sales.csv")

# With explicit schema
from pyspark.sql.types import *

schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_id", IntegerType(), True),
    StructField("product", StringType(), True),
    StructField("amount", DecimalType(10, 2), True),
    StructField("order_date", DateType(), True)
])

df = spark.read.csv("data/sales.csv", header=True, schema=schema)
```

### JSON Files

```python
# Read JSON
df = spark.read.json("data/users.json")

# Multi-line JSON
df = spark.read \
    .option("multiLine", "true") \
    .json("data/complex.json")

# Nested JSON
df.printSchema()
# root
#  |-- user: struct
#  |    |-- id: long
#  |    |-- name: string
#  |-- orders: array
#  |    |-- element: struct
#  |    |    |-- order_id: long
#  |    |    |-- amount: double

# Flatten nested structures
df.select("user.id", "user.name", "orders").show()
```

### Parquet Files (Recommended)

```python
# Read Parquet (columnar format)
df = spark.read.parquet("data/sales.parquet")

# Parquet preserves schema
df.printSchema()  # Schema automatically inferred

# Partitioned Parquet
df = spark.read.parquet("data/sales_partitioned/year=2024/month=*/")

# Predicate pushdown (reads only needed partitions)
df_filtered = spark.read.parquet("data/sales_partitioned") \
    .filter("year = 2024 AND month = 1")
# Only reads year=2024/month=1 partition!
```

### JDBC Databases

```python
# Read from PostgreSQL
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "sales") \
    .option("user", "username") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# With query
query = "(SELECT * FROM sales WHERE amount > 1000) AS sales_filtered"
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", query) \
    .option("user", "username") \
    .option("password", "password") \
    .load()

# Parallel read with partitioning
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "sales") \
    .option("user", "username") \
    .option("password", "password") \
    .option("partitionColumn", "order_id") \
    .option("lowerBound", "1") \
    .option("upperBound", "1000000") \
    .option("numPartitions", "10") \
    .load()
```

## SQL Queries

### Creating Temporary Views

```python
# Load data
df = spark.read.parquet("data/sales.parquet")

# Create temporary view
df.createOrReplaceTempView("sales")

# Now you can query with SQL
result = spark.sql("""
    SELECT product, SUM(amount) as total_sales
    FROM sales
    WHERE order_date >= '2024-01-01'
    GROUP BY product
    ORDER BY total_sales DESC
""")

result.show()
```

### Global Temporary Views

```python
# Global view (accessible across sessions)
df.createGlobalTempView("global_sales")

# Access with global_temp prefix
spark.sql("SELECT * FROM global_temp.global_sales").show()
```

## Common SQL Operations

### Filtering and Selection

```python
# SQL way
result = spark.sql("""
    SELECT customer_id, product, amount
    FROM sales
    WHERE amount > 100 AND order_date BETWEEN '2024-01-01' AND '2024-12-31'
""")

# DataFrame API way (equivalent)
from pyspark.sql.functions import col

result = df.select("customer_id", "product", "amount") \
    .filter((col("amount") > 100) & 
            (col("order_date").between('2024-01-01', '2024-12-31')))
```

### Aggregations

```python
# GROUP BY with aggregations
result = spark.sql("""
    SELECT 
        product,
        COUNT(*) as order_count,
        SUM(amount) as total_sales,
        AVG(amount) as avg_sale,
        MIN(amount) as min_sale,
        MAX(amount) as max_sale,
        STDDEV(amount) as stddev_sale
    FROM sales
    GROUP BY product
    HAVING total_sales > 10000
    ORDER BY total_sales DESC
""")

result.show()
```

### Joins

```python
# Create views
customers_df.createOrReplaceTempView("customers")
orders_df.createOrReplaceTempView("orders")

# Inner join
result = spark.sql("""
    SELECT 
        c.customer_name,
        c.email,
        o.order_id,
        o.amount,
        o.order_date
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.amount > 500
""")

# Left outer join
result = spark.sql("""
    SELECT 
        c.customer_name,
        COUNT(o.order_id) as order_count,
        COALESCE(SUM(o.amount), 0) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_name
""")
```

### Subqueries

```python
# Subquery in WHERE clause
result = spark.sql("""
    SELECT *
    FROM orders
    WHERE customer_id IN (
        SELECT customer_id 
        FROM customers 
        WHERE country = 'USA'
    )
""")

# Subquery in FROM clause
result = spark.sql("""
    SELECT 
        category,
        AVG(total_sales) as avg_sales_per_product
    FROM (
        SELECT 
            product,
            category,
            SUM(amount) as total_sales
        FROM sales
        GROUP BY product, category
    )
    GROUP BY category
""")
```

### Window Functions

```python
# Ranking
result = spark.sql("""
    SELECT 
        customer_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_number,
        RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) as amount_rank,
        DENSE_RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) as dense_rank
    FROM orders
""")

# Running totals
result = spark.sql("""
    SELECT 
        order_date,
        amount,
        SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total,
        AVG(amount) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day
    FROM sales
""")

# Lag and Lead
result = spark.sql("""
    SELECT 
        order_date,
        amount,
        LAG(amount, 1) OVER (ORDER BY order_date) as previous_day_sales,
        LEAD(amount, 1) OVER (ORDER BY order_date) as next_day_sales,
        amount - LAG(amount, 1) OVER (ORDER BY order_date) as daily_change
    FROM daily_sales
""")
```

### Common Table Expressions (CTEs)

```python
result = spark.sql("""
    WITH monthly_sales AS (
        SELECT 
            YEAR(order_date) as year,
            MONTH(order_date) as month,
            SUM(amount) as total_sales
        FROM sales
        GROUP BY YEAR(order_date), MONTH(order_date)
    ),
    sales_growth AS (
        SELECT 
            year,
            month,
            total_sales,
            LAG(total_sales) OVER (ORDER BY year, month) as prev_month_sales,
            (total_sales - LAG(total_sales) OVER (ORDER BY year, month)) / 
                LAG(total_sales) OVER (ORDER BY year, month) * 100 as growth_pct
        FROM monthly_sales
    )
    SELECT *
    FROM sales_growth
    WHERE growth_pct > 10
""")
```

## Advanced SQL Features

### PIVOT

```python
# Transform rows to columns
result = spark.sql("""
    SELECT * 
    FROM (
        SELECT region, month, sales
        FROM regional_sales
    )
    PIVOT (
        SUM(sales)
        FOR month IN ('Jan', 'Feb', 'Mar', 'Apr')
    )
""")
```

### CUBE and ROLLUP

```python
# CUBE: All combinations of grouping columns
result = spark.sql("""
    SELECT 
        region,
        product_category,
        SUM(sales) as total_sales
    FROM sales
    GROUP BY region, product_category WITH CUBE
""")

# ROLLUP: Hierarchical aggregations
result = spark.sql("""
    SELECT 
        year,
        quarter,
        month,
        SUM(sales) as total_sales
    FROM sales
    GROUP BY year, quarter, month WITH ROLLUP
""")
```

## User-Defined Functions (UDFs)

### Python UDF

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

# Define Python function
def categorize_amount(amount):
    if amount < 50:
        return "Low"
    elif amount < 200:
        return "Medium"
    else:
        return "High"

# Register as UDF
categorize_udf = udf(categorize_amount, StringType())

# Use in DataFrame API
df.withColumn("category", categorize_udf(col("amount"))).show()

# Register for SQL
spark.udf.register("categorize_amount", categorize_amount, StringType())

# Use in SQL
result = spark.sql("""
    SELECT 
        product,
        amount,
        categorize_amount(amount) as category
    FROM sales
""")
```

### Pandas UDF (Vectorized)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Scalar Pandas UDF (more efficient)
@pandas_udf(StringType())
def categorize_amount_pandas(amounts: pd.Series) -> pd.Series:
    return amounts.apply(lambda x: 
        "Low" if x < 50 else "Medium" if x < 200 else "High"
    )

# Use in DataFrame
df.withColumn("category", categorize_amount_pandas(col("amount"))).show()

# Grouped aggregate Pandas UDF
@pandas_udf("double")
def mean_with_outliers_removed(values: pd.Series) -> float:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    filtered = values[(values >= q1 - 1.5*iqr) & (values <= q3 + 1.5*iqr)]
    return filtered.mean()

df.groupBy("product").agg(
    mean_with_outliers_removed(col("amount")).alias("trimmed_mean")
).show()
```

## Performance Optimization

### Caching and Persistence

```python
# Cache frequently-used DataFrame
df.cache()
# or
df.persist()

# Check if cached
print(df.is_cached)

# Use cached data
result1 = df.filter(col("amount") > 100).count()
result2 = df.filter(col("product") == "Laptop").count()

# Unpersist when done
df.unpersist()
```

### Broadcast Joins

```python
from pyspark.sql.functions import broadcast

# Small dimension table
dim_products = spark.read.parquet("data/products.parquet")

# Large fact table
fact_sales = spark.read.parquet("data/sales.parquet")

# Broadcast small table
result = fact_sales.join(
    broadcast(dim_products),
    on="product_id"
)

# SQL way
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)  # 10 MB
result = spark.sql("""
    SELECT /*+ BROADCAST(p) */
        s.order_id,
        p.product_name,
        s.amount
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
""")
```

### Partitioning

```python
# Repartition before expensive operations
df_repartitioned = df.repartition(200, "customer_id")

# Coalesce to reduce partitions
df_coalesced = df.coalesce(10)

# Partition by column when writing
df.write.partitionBy("year", "month").parquet("data/sales_partitioned")

# Read with partition pruning
df = spark.read.parquet("data/sales_partitioned") \
    .filter("year = 2024 AND month = 1")
# Reads only year=2024/month=1 directory!
```

### Adaptive Query Execution (AQE)

```python
# Enable AQE (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# AQE will:
# 1. Dynamically coalesce partitions
# 2. Optimize join strategies at runtime
# 3. Handle skewed data automatically
```

## Complete Example: E-Commerce Analytics

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Initialize
spark = SparkSession.builder \
    .appName("EcommerceAnalytics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Load data
orders = spark.read.parquet("data/orders.parquet")
customers = spark.read.parquet("data/customers.parquet")
products = spark.read.parquet("data/products.parquet")

# Create views
orders.createOrReplaceTempView("orders")
customers.createOrReplaceTempView("customers")
products.createOrReplaceTempView("products")

# Analysis 1: Customer Lifetime Value
clv = spark.sql("""
    WITH customer_stats AS (
        SELECT 
            c.customer_id,
            c.customer_name,
            c.registration_date,
            COUNT(o.order_id) as total_orders,
            SUM(o.amount) as total_spent,
            AVG(o.amount) as avg_order_value,
            MIN(o.order_date) as first_order_date,
            MAX(o.order_date) as last_order_date,
            DATEDIFF(MAX(o.order_date), MIN(o.order_date)) as customer_lifespan_days
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.customer_name, c.registration_date
    )
    SELECT 
        *,
        CASE 
            WHEN total_spent > 10000 THEN 'VIP'
            WHEN total_spent > 5000 THEN 'Gold'
            WHEN total_spent > 1000 THEN 'Silver'
            ELSE 'Bronze'
        END as customer_tier
    FROM customer_stats
    ORDER BY total_spent DESC
""")

clv.write.parquet("output/customer_lifetime_value", mode="overwrite")

# Analysis 2: Product Performance with Trends
product_performance = spark.sql("""
    WITH monthly_product_sales AS (
        SELECT 
            p.product_name,
            p.category,
            YEAR(o.order_date) as year,
            MONTH(o.order_date) as month,
            SUM(o.amount) as monthly_revenue,
            COUNT(o.order_id) as monthly_orders
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.product_name, p.category, YEAR(o.order_date), MONTH(o.order_date)
    )
    SELECT 
        product_name,
        category,
        year,
        month,
        monthly_revenue,
        monthly_orders,
        LAG(monthly_revenue) OVER (
            PARTITION BY product_name 
            ORDER BY year, month
        ) as prev_month_revenue,
        (monthly_revenue - LAG(monthly_revenue) OVER (
            PARTITION BY product_name 
            ORDER BY year, month
        )) / LAG(monthly_revenue) OVER (
            PARTITION BY product_name 
            ORDER BY year, month
        ) * 100 as revenue_growth_pct
    FROM monthly_product_sales
    ORDER BY product_name, year, month
""")

product_performance.write.partitionBy("year", "month") \
    .parquet("output/product_performance", mode="overwrite")

# Analysis 3: RFM Segmentation
rfm = spark.sql("""
    WITH rfm_calc AS (
        SELECT 
            customer_id,
            DATEDIFF(CURRENT_DATE(), MAX(order_date)) as recency,
            COUNT(order_id) as frequency,
            SUM(amount) as monetary
        FROM orders
        GROUP BY customer_id
    ),
    rfm_quantiles AS (
        SELECT 
            customer_id,
            recency,
            frequency,
            monetary,
            NTILE(5) OVER (ORDER BY recency DESC) as r_score,
            NTILE(5) OVER (ORDER BY frequency) as f_score,
            NTILE(5) OVER (ORDER BY monetary) as m_score
        FROM rfm_calc
    )
    SELECT 
        customer_id,
        recency,
        frequency,
        monetary,
        r_score,
        f_score,
        m_score,
        CONCAT(r_score, f_score, m_score) as rfm_score,
        CASE 
            WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'Promising'
            WHEN r_score <= 2 AND f_score >= 4 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as customer_segment
    FROM rfm_quantiles
""")

rfm.write.parquet("output/rfm_segmentation", mode="overwrite")

print("Analytics completed successfully!")
spark.stop()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Spark SQL provides SQL interface** to Big Data with Spark performance
2. **Catalyst optimizer** automatically optimizes query plans
3. **DataFrame and SQL APIs** are interchangeable and equally performant
4. **Support for multiple data sources**: Parquet, JSON, CSV, JDBC, Hive
5. **Window functions** enable advanced analytics without self-joins
6. **UDFs extend functionality** but Pandas UDFs are more efficient
7. **Broadcast joins** optimize joins with small tables
8. **Partitioning and AQE** critical for performance at scale
:::

## Further Reading

- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Catalyst Optimizer](https://databricks.com/glossary/catalyst-optimizer)
- Chambers, B. & Zaharia, M. (2018). "Spark: The Definitive Guide", Chapters 4-12
