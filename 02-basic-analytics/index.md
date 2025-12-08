# Chapter 2: Basic Data Analytics Methods

## Introduction

This chapter establishes the statistical and analytical foundations essential for Big Data analytics. While Big Data introduces scale challenges, the fundamental principles of statistics, probability, and sampling remain crucial for drawing valid insights from massive datasets.

## Learning Objectives

By the end of this chapter, you will be able to:

- Apply descriptive statistics to summarize large datasets
- Understand probability distributions and their applications
- Perform statistical hypothesis testing
- Design and implement sampling strategies for Big Data
- Recognize and avoid common statistical pitfalls
- Use Python libraries (NumPy, Pandas, SciPy) for statistical analysis

## Chapter Overview

This chapter covers:

1. **Descriptive Statistics** - Measures of central tendency, dispersion, and distribution shape
2. **Probability Theory** - Foundations for statistical inference
3. **Probability Distributions** - Normal, binomial, Poisson, and others
4. **Sampling Methods** - Techniques for working with large datasets
5. **Statistical Inference** - Hypothesis testing and confidence intervals
6. **Correlation and Association** - Measuring relationships between variables

## Why Statistics for Big Data?

### The Volume Paradox

**More data ≠ More insight** (automatically)

Challenges with Big Data:
- **Spurious correlations**: With enough data, random patterns appear significant
- **Multiple testing problem**: Testing thousands of hypotheses increases false positives
- **Selection bias**: Non-random data collection creates misleading patterns
- **Overfitting**: Models memorize noise instead of learning patterns

### Statistical Foundations Remain Critical

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example: Simpson's Paradox in Big Data
# Aggregated data can show opposite trends

# Department A: 20% acceptance rate (100/500 applicants)
dept_a = pd.DataFrame({
    'gender': ['M']*400 + ['F']*100,
    'admitted': [80]*1 + [20]*1  # 80 M admitted, 20 F admitted
})

# Department B: 30% acceptance rate (50/200 applicants)  
dept_b = pd.DataFrame({
    'gender': ['M']*50 + ['F']*150,
    'admitted': [10]*1 + [40]*1  # 10 M admitted, 40 F admitted
})

# Overall: Males 22.2% (90/450), Females 24% (60/250)
# But in EACH department, acceptance rate is equal or favors females!
# This is why proper statistical analysis matters even with big data
```

## Key Concepts

### Population vs. Sample

- **Population**: Complete set of all items
- **Sample**: Subset selected from population
- **Sampling**: Process of selecting sample
- **Inference**: Drawing conclusions about population from sample

### Types of Data

**Quantitative (Numerical)**:
- **Continuous**: Can take any value in range (height, temperature, price)
- **Discrete**: Countable values (number of purchases, page views)

**Qualitative (Categorical)**:
- **Nominal**: No natural order (color, country, product category)
- **Ordinal**: Natural ordering (satisfaction rating, education level)

### Measurement Scales

1. **Nominal**: Categories without order (gender, product type)
2. **Ordinal**: Categories with order (small/medium/large)
3. **Interval**: Numeric with equal intervals, no true zero (temperature in Celsius)
4. **Ratio**: Numeric with equal intervals and true zero (age, income, weight)

## Real-World Applications

### E-Commerce Analytics

```python
# Descriptive statistics for customer behavior
import pandas as pd
import numpy as np

# Load customer purchase data
df = pd.read_csv('data/customer_purchases.csv')

# Summary statistics
print(df['purchase_amount'].describe())
# count, mean, std, min, 25%, 50%, 75%, max

# Identify outliers using IQR method
Q1 = df['purchase_amount'].quantile(0.25)
Q3 = df['purchase_amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['purchase_amount'] < Q1 - 1.5*IQR) | 
              (df['purchase_amount'] > Q3 + 1.5*IQR)]

print(f"Outliers: {len(outliers)} out of {len(df)} purchases")
```

### A/B Testing

```python
from scipy import stats

# Test if new website design increases conversion
control_group = [0, 1, 0, 0, 1, 1, 0, ...]  # Old design
treatment_group = [1, 1, 0, 1, 1, 1, 1, ...]  # New design

control_rate = np.mean(control_group)
treatment_rate = np.mean(treatment_group)

# Chi-square test
contingency_table = pd.crosstab(
    ['control']*len(control_group) + ['treatment']*len(treatment_group),
    control_group + treatment_group
)

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

if p_value < 0.05:
    print(f"Significant difference! p-value: {p_value:.4f}")
    print(f"Treatment conversion: {treatment_rate:.2%}")
    print(f"Control conversion: {control_rate:.2%}")
else:
    print("No significant difference detected")
```

### Quality Control

```python
# Monitor manufacturing process
import numpy as np
import matplotlib.pyplot as plt

# Widget weights should be 100g ± 2g
weights = np.random.normal(100, 0.8, 1000)  # Sample measurements

# Control chart
mean = np.mean(weights)
std = np.std(weights)

ucl = mean + 3*std  # Upper control limit
lcl = mean - 3*std  # Lower control limit

plt.figure(figsize=(12, 6))
plt.plot(weights, 'b.', alpha=0.5)
plt.axhline(mean, color='g', linestyle='-', label='Mean')
plt.axhline(ucl, color='r', linestyle='--', label='UCL')
plt.axhline(lcl, color='r', linestyle='--', label='LCL')
plt.axhline(102, color='orange', linestyle=':', label='Spec Upper')
plt.axhline(98, color='orange', linestyle=':', label='Spec Lower')
plt.legend()
plt.title('Statistical Process Control Chart')
plt.xlabel('Sample Number')
plt.ylabel('Weight (g)')
plt.show()

# Out of control?
out_of_control = np.sum((weights > ucl) | (weights < lcl))
print(f"Out of control points: {out_of_control}")
```

## Python Libraries for Statistical Analysis

### Essential Libraries

```python
import numpy as np              # Numerical computing
import pandas as pd             # Data manipulation
import scipy.stats as stats     # Statistical functions
import matplotlib.pyplot as plt # Visualization
import seaborn as sns          # Statistical visualization
from sklearn import preprocessing # Data preprocessing

# For Big Data
import dask.dataframe as dd    # Parallel pandas
from pyspark.sql import functions as F  # Spark SQL
```

### Quick Start Example

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Generate sample data
np.random.seed(42)
data = {
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.lognormal(10.5, 0.5, 1000),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 1000, p=[0.2, 0.5, 0.3])
}
df = pd.DataFrame(data)

# Descriptive statistics
print(df.describe())
print(df['satisfaction'].value_counts())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(df['age'], bins=30, edgecolor='black')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

# Box plot
df.boxplot(column='income', ax=axes[1])
axes[1].set_title('Income Distribution')

# Bar chart
df['satisfaction'].value_counts().plot(kind='bar', ax=axes[2])
axes[2].set_title('Satisfaction Levels')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()
```

## Big Data Considerations

### Computational Challenges

```python
# Traditional approach (doesn't scale)
df = pd.read_csv('massive_file.csv')  # Loads entire file into memory
mean = df['column'].mean()  # Works only if data fits in RAM

# Big Data approach 1: Chunking
chunk_size = 100000
running_sum = 0
running_count = 0

for chunk in pd.read_csv('massive_file.csv', chunksize=chunk_size):
    running_sum += chunk['column'].sum()
    running_count += len(chunk)

mean = running_sum / running_count

# Big Data approach 2: Dask (parallel pandas)
import dask.dataframe as dd

ddf = dd.read_csv('massive_file.csv')
mean = ddf['column'].mean().compute()

# Big Data approach 3: Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('massive_file.csv', header=True, inferSchema=True)
mean = df.select(avg('column')).collect()[0][0]
```

### Sampling Strategies

**When to sample**:
- Initial exploratory analysis
- Algorithm development and testing
- Visualization (can't plot billions of points)
- Interactive analysis

**When to use full data**:
- Final model training
- Production analytics
- Rare event detection
- Regulatory compliance

## Chapter Structure

The following sections provide comprehensive coverage of:

1. **Descriptive Statistics** - Summarizing data distributions
2. **Probability Theory** - Foundations for inference
3. **Probability Distributions** - Common distributions and applications
4. **Sampling Methods** - Techniques for Big Data
5. **Statistical Inference** - Hypothesis testing and confidence intervals
6. **Correlation and Association** - Measuring relationships

## Prerequisites

Before proceeding, ensure familiarity with:
- Python programming (Chapter 1)
- Basic mathematics (algebra)
- Pandas and NumPy basics

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Statistics is essential** even with Big Data
2. **More data amplifies** both signal and noise
3. **Sampling remains valuable** for efficiency and exploration
4. **Visualization complements** numerical summaries
5. **Proper methodology prevents** spurious conclusions
6. **Python provides rich tools** for statistical analysis
:::

## Next Steps

Proceed to the following sections:
- **Section 2.1**: Descriptive Statistics
- **Section 2.2**: Probability Theory Basics
- **Section 2.3**: Probability Distributions
- **Section 2.4**: Sampling Methods
- **Section 2.5**: Statistical Inference
- **Section 2.6**: Correlation and Association

## Further Reading

- Wickham, H. & Grolemund, G. (2017). "R for Data Science" (concepts apply to Python)
- VanderPlas, J. (2016). "Python Data Science Handbook"
- Downey, A. (2014). "Think Stats: Exploratory Data Analysis in Python"
- Ross, S. (2014). "A First Course in Probability"
