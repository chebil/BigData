# Spark MLlib: Distributed Machine Learning

## Learning Objectives

- Understand MLlib architecture and components
- Build machine learning pipelines with Spark
- Perform classification, regression, and clustering at scale
- Feature engineering and transformation
- Model evaluation and hyperparameter tuning
- Deploy ML models in production

## Introduction

MLlib is Spark's scalable machine learning library that provides distributed implementations of common algorithms. It's designed to run efficiently on large datasets that don't fit in memory on a single machine.

## Why MLlib?

### Advantages

1. **Scalability**: Train on terabytes of data
2. **Speed**: In-memory computing, 100x faster than Hadoop Mahout
3. **Integration**: Seamless with Spark SQL, DataFrames, Streaming
4. **Ease of Use**: High-level APIs in Python, Scala, Java, R
5. **Algorithms**: Comprehensive library covering most use cases

### MLlib vs. Scikit-Learn

| Feature | Scikit-Learn | Spark MLlib |
|---------|-------------|------------|
| **Data Size** | Fits in RAM | Distributed (TB+) |
| **Speed** | Fast on small data | Optimized for big data |
| **Algorithms** | Comprehensive | Core algorithms |
| **Deployment** | Single machine | Cluster |
| **Learning Curve** | Easy | Moderate |

## ML Pipeline Architecture

### Components

```
DataFrame
    |
    v
[Transformer] -> Transform data (feature engineering)
    |
    v
[Estimator] -> Train model (fit)
    |
    v
[Model] -> Make predictions (transform)
    |
    v
Predictions DataFrame
```

### Key Concepts

1. **Transformer**: Converts one DataFrame to another (e.g., feature scaling)
2. **Estimator**: Learns from data (e.g., LogisticRegression)
3. **Pipeline**: Chains transformers and estimators
4. **Model**: Trained estimator (transformer)
5. **Evaluator**: Computes metrics

## Basic Setup

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.regression import *
from pyspark.ml.clustering import *
from pyspark.ml.evaluation import *

# Create Spark session
spark = SparkSession.builder \
    .appName("MLlibExample") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load data
df = spark.read.csv("data/customer_data.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)
```

## Feature Engineering

### Vector Assembler

```python
from pyspark.ml.feature import VectorAssembler

# Combine features into single vector column
feature_cols = ['age', 'income', 'purchase_frequency', 'avg_purchase_amount']

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_assembled = assembler.transform(df)
df_assembled.select("features", "churned").show(5, truncate=False)
```

### String Indexer

```python
from pyspark.ml.feature import StringIndexer

# Convert categorical string to numeric index
indexer = StringIndexer(
    inputCol="product_category",
    outputCol="category_index"
)

indexer_model = indexer.fit(df)
df_indexed = indexer_model.transform(df)

df_indexed.select("product_category", "category_index").distinct().show()
```

### One-Hot Encoding

```python
from pyspark.ml.feature import OneHotEncoder

# Convert categorical index to binary vectors
encoder = OneHotEncoder(
    inputCols=["category_index"],
    outputCols=["category_vec"]
)

df_encoded = encoder.fit(df_indexed).transform(df_indexed)
df_encoded.select("product_category", "category_index", "category_vec").show(5, truncate=False)
```

### Standard Scaler

```python
from pyspark.ml.feature import StandardScaler

# Standardize features (mean=0, std=1)
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
```

### Min-Max Scaler

```python
from pyspark.ml.feature import MinMaxScaler

# Scale features to [0, 1]
min_max_scaler = MinMaxScaler(
    inputCol="features",
    outputCol="scaled_features"
)

df_scaled = min_max_scaler.fit(df_assembled).transform(df_assembled)
```

### Bucketizer

```python
from pyspark.ml.feature import Bucketizer

# Discretize continuous variable into bins
splits = [-float("inf"), 25, 35, 50, float("inf")]

bucketizer = Bucketizer(
    splits=splits,
    inputCol="age",
    outputCol="age_bucket"
)

df_bucketed = bucketizer.transform(df)
df_bucketed.select("age", "age_bucket").show(10)
```

## Classification

### Logistic Regression

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Prepare data
df = spark.read.csv("data/customer_churn.csv", header=True, inferSchema=True)

# Feature engineering
feature_cols = ['age', 'income', 'tenure', 'monthly_charges', 'total_charges']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Split data
train, test = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="churned",
    maxIter=10,
    regParam=0.01
)

model = lr.fit(train)

# Make predictions
predictions = model.transform(test)
predictions.select("features", "churned", "prediction", "probability").show(10, truncate=False)

# Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="churned",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Model coefficients
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")
```

### Random Forest Classifier

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=5,
    seed=42
)

rf_model = rf.fit(train)

# Predictions
predictions = rf_model.transform(test)

# Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
feature_importances = rf_model.featureImportances
for i, importance in enumerate(feature_importances):
    print(f"Feature {feature_cols[i]}: {importance:.4f}")
```

### Gradient-Boosted Trees

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    maxDepth=5
)

gbt_model = gbt.fit(train)
predictions = gbt_model.transform(test)

# Evaluate
accuracy = evaluator.evaluate(predictions)
print(f"GBT Accuracy: {accuracy:.4f}")
```

## Regression

### Linear Regression

```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Load data
df = spark.read.csv("data/housing_prices.csv", header=True, inferSchema=True)

# Feature engineering
feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'distance_to_city']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Split data
train, test = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LinearRegression(
    featuresCol="features",
    labelCol="price",
    maxIter=10,
    regParam=0.3,
    elasticNetParam=0.8
)

lr_model = lr.fit(train)

# Model summary
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"RMSE: {lr_model.summary.rootMeanSquaredError:.2f}")
print(f"R²: {lr_model.summary.r2:.4f}")

# Predictions
predictions = lr_model.transform(test)
predictions.select("features", "price", "prediction").show(10)

# Evaluate
evaluator = RegressionEvaluator(
    labelCol="price",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"Test RMSE: {rmse:.2f}")
```

### Decision Tree Regression

```python
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="price",
    maxDepth=5
)

dt_model = dt.fit(train)
predictions = dt_model.transform(test)

rmse = evaluator.evaluate(predictions)
print(f"Decision Tree RMSE: {rmse:.2f}")
```

### Random Forest Regression

```python
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="price",
    numTrees=100,
    maxDepth=5
)

rf_model = rf.fit(train)
predictions = rf_model.transform(test)

rmse = evaluator.evaluate(predictions)
print(f"Random Forest RMSE: {rmse:.2f}")
```

## Clustering

### K-Means

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Load data
df = spark.read.csv("data/customer_segments.csv", header=True, inferSchema=True)

# Feature engineering
feature_cols = ['annual_income', 'spending_score']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Train K-Means
kmeans = KMeans(
    featuresCol="features",
    k=5,
    seed=42,
    maxIter=20
)

model = kmeans.fit(df_assembled)

# Cluster centers
centers = model.clusterCenters()
print("Cluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Predictions
predictions = model.transform(df_assembled)
predictions.select("features", "prediction").show(10, truncate=False)

# Evaluate (Silhouette score)
evaluator = ClusteringEvaluator(
    featuresCol="features",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.4f}")

# Within Set Sum of Squared Errors
wssse = model.summary.trainingCost
print(f"Within Set Sum of Squared Errors: {wssse:.2f}")
```

### Elbow Method for Optimal K

```python
import matplotlib.pyplot as plt

# Try different values of k
k_values = range(2, 11)
wssse_values = []
silhouette_values = []

for k in k_values:
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(df_assembled)
    
    wssse = model.summary.trainingCost
    wssse_values.append(wssse)
    
    predictions = model.transform(df_assembled)
    silhouette = evaluator.evaluate(predictions)
    silhouette_values.append(silhouette)
    
    print(f"k={k}: WSSSE={wssse:.2f}, Silhouette={silhouette:.4f}")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_values, wssse_values, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WSSSE')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(k_values, silhouette_values, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimal_k_analysis.png')
```

## ML Pipelines

### Complete Pipeline Example

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# Load data
df = spark.read.csv("data/titanic.csv", header=True, inferSchema=True)

# Define pipeline stages

# Stage 1: Index categorical variables
sex_indexer = StringIndexer(inputCol="Sex", outputCol="sex_index")
embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="embarked_index")

# Stage 2: Assemble features
feature_cols = ['Pclass', 'sex_index', 'Age', 'SibSp', 'Parch', 'Fare', 'embarked_index']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Stage 3: Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Stage 4: Train classifier
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="Survived",
    numTrees=100
)

# Create pipeline
pipeline = Pipeline(stages=[
    sex_indexer,
    embarked_indexer,
    assembler,
    scaler,
    rf
])

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train pipeline
model = pipeline.fit(train)

# Make predictions
predictions = model.transform(test)
predictions.select("Survived", "prediction", "probability").show(10)

# Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Save pipeline
model.write().overwrite().save("models/titanic_pipeline")

# Load pipeline
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load("models/titanic_pipeline")
```

## Hyperparameter Tuning

### Cross-Validation

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Define model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [10, 50, 100]) \
    .build()

# Cross-validator
cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="label"),
    numFolds=5,
    seed=42
)

# Train
cv_model = cv.fit(train)

# Best model
best_model = cv_model.bestModel
print(f"Best regParam: {best_model.getRegParam()}")
print(f"Best elasticNetParam: {best_model.getElasticNetParam()}")
print(f"Best maxIter: {best_model.getMaxIter()}")

# Evaluate
predictions = cv_model.transform(test)
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")
```

### Train-Validation Split

```python
from pyspark.ml.tuning import TrainValidationSplit

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Train-validation split (faster than cross-validation)
tvs = TrainValidationSplit(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassClassificationEvaluator(labelCol="label"),
    trainRatio=0.8,
    seed=42
)

tvs_model = tvs.fit(train)

# Best parameters
best_model = tvs_model.bestModel
print(f"Best numTrees: {best_model.getNumTrees}")
print(f"Best maxDepth: {best_model.getMaxDepth()}")
```

## Model Persistence

```python
# Save model
model.write().overwrite().save("models/my_model")

# Save pipeline
pipeline_model.write().overwrite().save("models/my_pipeline")

# Load model
from pyspark.ml.classification import LogisticRegressionModel
loaded_model = LogisticRegressionModel.load("models/my_model")

# Load pipeline
from pyspark.ml import PipelineModel
loaded_pipeline = PipelineModel.load("models/my_pipeline")

# Use loaded model
predictions = loaded_model.transform(new_data)
```

## Complete Example: Customer Churn Prediction

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load data
df = spark.read.csv("data/telecom_churn.csv", header=True, inferSchema=True)

print(f"Total records: {df.count()}")
df.printSchema()

# Handle missing values
df = df.na.fill({
    'TotalCharges': 0,
    'tenure': 0
})

# Feature engineering pipeline

# Categorical features
categorical_cols = ['Gender', 'Contract', 'PaymentMethod', 'InternetService']
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") 
            for col in categorical_cols]

encoders = [OneHotEncoder(inputCols=[f"{col}_index"], 
                          outputCols=[f"{col}_vec"]) 
            for col in categorical_cols]

# Numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Assemble all features
feature_cols = [f"{col}_vec" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Model
gbt = GBTClassifier(
    featuresCol="scaled_features",
    labelCol="Churn",
    maxIter=20
)

# Create pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train.count()} records")
print(f"Test set: {test.count()} records")

# Hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 7, 10]) \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .build()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="Churn"),
    numFolds=3,
    seed=42
)

# Train
print("Training model...")
cv_model = cv.fit(train)

# Predictions
print("Making predictions...")
predictions = cv_model.transform(test)

# Evaluation
bin_evaluator = BinaryClassificationEvaluator(labelCol="Churn")
mc_evaluator = MulticlassClassificationEvaluator(labelCol="Churn")

auc = bin_evaluator.evaluate(predictions, {bin_evaluator.metricName: "areaUnderROC"})
accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})

print("\n=== Model Performance ===")
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
predictions_pd = predictions.select("Churn", "prediction").toPandas()
from sklearn.metrics import confusion_matrix, classification_report

print("\n=== Confusion Matrix ===")
print(confusion_matrix(predictions_pd['Churn'], predictions_pd['prediction']))

print("\n=== Classification Report ===")
print(classification_report(predictions_pd['Churn'], predictions_pd['prediction']))

# Save model
print("\nSaving model...")
cv_model.write().overwrite().save("models/churn_prediction_model")
print("Model saved successfully!")

spark.stop()
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **MLlib provides distributed ML** for datasets that don't fit in memory
2. **Pipelines chain transformers and estimators** for reproducible workflows
3. **Feature engineering critical**: VectorAssembler, StringIndexer, OneHotEncoder, Scalers
4. **Supports main ML tasks**: Classification, Regression, Clustering
5. **Hyperparameter tuning** via CrossValidator or TrainValidationSplit
6. **Models are serializable** for production deployment
7. **Integration with Spark SQL** enables end-to-end data pipelines
8. **Trade-off**: Fewer algorithms than scikit-learn, but scales to massive data
:::

## Practical Exercises

### Exercise 1: Customer Segmentation

Use K-Means to segment customers based on purchasing behavior.

### Exercise 2: House Price Prediction

Build a regression pipeline with feature engineering and hyperparameter tuning.

### Exercise 3: Spam Detection

Implement text classification using TF-IDF and Logistic Regression.

## Further Reading

- [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLlib API Documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html)
- Chambers, B. & Zaharia, M. (2018). "Spark: The Definitive Guide", Chapters 24-28
- Ryza, S. et al. (2017). "Advanced Analytics with Spark"
