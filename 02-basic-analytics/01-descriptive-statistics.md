# Descriptive Statistics

## Learning Objectives

- Calculate and interpret measures of central tendency
- Compute measures of dispersion and variability
- Understand distribution shapes and moments
- Identify and handle outliers
- Visualize distributions effectively
- Apply descriptive statistics to Big Data

## Introduction

Descriptive statistics summarize and describe the main features of a dataset. They provide simple summaries about the sample and measures, forming the foundation of quantitative data analysis.

## Measures of Central Tendency

### Mean (Arithmetic Average)

**Definition**: Sum of all values divided by count

**Formula**: 
\[
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
\]

**Python Implementation**:

```python
import numpy as np
import pandas as pd

# Sample data: customer ages
ages = [25, 30, 35, 28, 42, 38, 29, 45, 31, 27]

# Method 1: NumPy
mean_age = np.mean(ages)
print(f"Mean age: {mean_age:.2f}")

# Method 2: Pandas
df = pd.DataFrame({'age': ages})
mean_age = df['age'].mean()
print(f"Mean age: {mean_age:.2f}")

# Method 3: Manual calculation
mean_age = sum(ages) / len(ages)
print(f"Mean age: {mean_age:.2f}")
```

**Properties**:
- ✅ Uses all data points
- ✅ Mathematically tractable
- ❌ Sensitive to outliers
- ❌ May not be representative with skewed data

### Median (Middle Value)

**Definition**: Middle value when data is sorted

**Formula**:
\[
\text{Median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even}
\end{cases}
\]

**Python Implementation**:

```python
# Data with outlier
salaries = [45000, 48000, 50000, 52000, 55000, 58000, 60000, 250000]

mean_salary = np.mean(salaries)
median_salary = np.median(salaries)

print(f"Mean salary: ${mean_salary:,.0f}")  # $77,250 (misleading!)
print(f"Median salary: ${median_salary:,.0f}")  # $53,500 (typical)

# The median is more representative here
```

**Properties**:
- ✅ Robust to outliers
- ✅ Representative of "typical" value
- ❌ Doesn't use all information
- ❌ Less mathematically tractable

### Mode (Most Frequent Value)

**Definition**: Value that appears most often

**Python Implementation**:

```python
from scipy import stats

# Categorical data: favorite colors
colors = ['blue', 'red', 'blue', 'green', 'blue', 'red', 'blue', 'yellow']

# Method 1: SciPy
mode_result = stats.mode(colors, keepdims=True)
mode_value = mode_result.mode[0]
mode_count = mode_result.count[0]
print(f"Mode: {mode_value} (appears {mode_count} times)")

# Method 2: Pandas (better for multiple modes)
df = pd.DataFrame({'color': colors})
mode_value = df['color'].mode()[0]
print(f"Mode: {mode_value}")

# Method 3: Value counts
print(df['color'].value_counts())
```

**Properties**:
- ✅ Only measure for categorical data
- ✅ Can have multiple modes (bimodal, multimodal)
- ❌ May not exist or be unique
- ❌ Ignores magnitude of values

### When to Use Each

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Symmetric distribution: Mean ≈ Median ≈ Mode
symmetric = np.random.normal(100, 15, 1000)

# Right-skewed: Mode < Median < Mean
right_skewed = np.random.exponential(20, 1000)

# Left-skewed: Mean < Median < Mode  
left_skewed = 100 - np.random.exponential(20, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title in zip(axes, 
                           [symmetric, right_skewed, left_skewed],
                           ['Symmetric', 'Right-Skewed', 'Left-Skewed']):
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(data), color='r', linestyle='--', label=f'Mean: {np.mean(data):.1f}')
    ax.axvline(np.median(data), color='g', linestyle='-', label=f'Median: {np.median(data):.1f}')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Decision rule:
# - Symmetric distribution → Use mean
# - Skewed distribution → Use median
# - Categorical data → Use mode
```

## Measures of Dispersion

### Range

**Definition**: Difference between maximum and minimum

```python
data = [10, 15, 20, 25, 30, 35, 40]

data_range = np.max(data) - np.min(data)
print(f"Range: {data_range}")  # 30

# Or using pandas
df = pd.DataFrame({'values': data})
data_range = df['values'].max() - df['values'].min()
```

**Issues**: Extremely sensitive to outliers

### Interquartile Range (IQR)

**Definition**: Range of middle 50% of data

\[
\text{IQR} = Q_3 - Q_1
\]

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # Note outlier

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

# Pandas method
df = pd.DataFrame({'values': data})
Q1 = df['values'].quantile(0.25)
Q3 = df['values'].quantile(0.75)
IQR = Q3 - Q1

# Outlier detection using IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['values'] < lower_bound) | (df['values'] > upper_bound)]
print(f"Outliers: {outliers['values'].tolist()}")
```

### Variance

**Population Variance**:
\[
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
\]

**Sample Variance** (Bessel's correction):
\[
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
\]

```python
data = [2, 4, 6, 8, 10]

# Population variance
pop_var = np.var(data, ddof=0)
print(f"Population variance: {pop_var:.2f}")

# Sample variance (default in pandas)
sample_var = np.var(data, ddof=1)
print(f"Sample variance: {sample_var:.2f}")

# Pandas (uses ddof=1 by default)
df = pd.DataFrame({'values': data})
variance = df['values'].var()
print(f"Variance: {variance:.2f}")
```

### Standard Deviation

**Definition**: Square root of variance

\[
\sigma = \sqrt{\sigma^2}
\]

```python
data = [2, 4, 6, 8, 10]

std_dev = np.std(data, ddof=1)
print(f"Standard deviation: {std_dev:.2f}")

# Interpretation: On average, values deviate from mean by {std_dev}

# Coefficient of Variation (CV): Relative variability
mean = np.mean(data)
cv = (std_dev / mean) * 100
print(f"Coefficient of Variation: {cv:.1f}%")
```

**Empirical Rule (Normal Distribution)**:
- 68% of data within 1 standard deviation of mean
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate normal distribution
mu, sigma = 100, 15
data = np.random.normal(mu, sigma, 10000)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')

# Overlay normal curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r-', linewidth=2)

# Mark standard deviations
for i in range(1, 4):
    plt.axvline(mu + i*sigma, color='g', linestyle='--', alpha=0.5)
    plt.axvline(mu - i*sigma, color='g', linestyle='--', alpha=0.5)

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
plt.show()

# Verify empirical rule
within_1std = np.sum((data >= mu-sigma) & (data <= mu+sigma)) / len(data)
within_2std = np.sum((data >= mu-2*sigma) & (data <= mu+2*sigma)) / len(data)
within_3std = np.sum((data >= mu-3*sigma) & (data <= mu+3*sigma)) / len(data)

print(f"Within 1σ: {within_1std:.1%} (expected 68%)")
print(f"Within 2σ: {within_2std:.1%} (expected 95%)")
print(f"Within 3σ: {within_3std:.1%} (expected 99.7%)")
```

## Distribution Shape

### Skewness

**Definition**: Measure of asymmetry

\[
\text{Skewness} = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3
\]

```python
from scipy.stats import skew, kurtosis

# Right-skewed (positive skew)
right_skewed = np.random.exponential(2, 1000)
print(f"Right-skewed skewness: {skew(right_skewed):.2f}")  # > 0

# Left-skewed (negative skew)
left_skewed = -np.random.exponential(2, 1000)
print(f"Left-skewed skewness: {skew(left_skewed):.2f}")  # < 0

# Symmetric
symmetric = np.random.normal(0, 1, 1000)
print(f"Symmetric skewness: {skew(symmetric):.2f}")  # ≈ 0

# Interpretation:
# Skewness > 0: Right-skewed (tail on right)
# Skewness < 0: Left-skewed (tail on left)
# Skewness ≈ 0: Symmetric
```

### Kurtosis

**Definition**: Measure of "tailedness"

\[
\text{Kurtosis} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}
\]

```python
# High kurtosis (heavy tails, more outliers)
heavy_tails = np.random.standard_t(3, 1000)
print(f"Heavy tails kurtosis: {kurtosis(heavy_tails):.2f}")  # > 0

# Low kurtosis (light tails, fewer outliers)
light_tails = np.random.uniform(-3, 3, 1000)
print(f"Light tails kurtosis: {kurtosis(light_tails):.2f}")  # < 0

# Normal distribution
normal = np.random.normal(0, 1, 1000)
print(f"Normal kurtosis: {kurtosis(normal):.2f}")  # ≈ 0 (excess kurtosis)

# Interpretation:
# Kurtosis > 0: Leptokurtic (heavy tails, peaked)
# Kurtosis < 0: Platykurtic (light tails, flat)
# Kurtosis ≈ 0: Mesokurtic (normal-like)
```

## Complete Descriptive Statistics

### Pandas describe() Method

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.lognormal(10.5, 0.5, 1000),
    'satisfaction_score': np.random.randint(1, 11, 1000)
})

# Comprehensive summary
print(df.describe())
# Output includes: count, mean, std, min, 25%, 50%, 75%, max

# Include all percentiles
print(df.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# Include categorical data
df['category'] = np.random.choice(['A', 'B', 'C'], 1000)
print(df.describe(include='all'))
```

### Custom Summary Function

```python
import scipy.stats as stats

def comprehensive_summary(data):
    """
    Comprehensive descriptive statistics
    """
    summary = {
        'count': len(data),
        'missing': np.sum(pd.isna(data)),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data, keepdims=True).mode[0] if len(data) > 0 else None,
        'std': np.std(data, ddof=1),
        'var': np.var(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'cv': (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else None
    }
    return pd.Series(summary)

# Apply to each column
df_summary = df.select_dtypes(include=[np.number]).apply(comprehensive_summary)
print(df_summary)
```

## Visualization

### Histogram

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic histogram
axes[0, 0].hist(df['age'], bins=30, edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Histogram with density curve
axes[0, 1].hist(df['age'], bins=30, density=True, alpha=0.7, edgecolor='black')
df['age'].plot(kind='density', ax=axes[0, 1], color='red', linewidth=2)
axes[0, 1].set_title('Age Distribution with Density')

# Box plot
df.boxplot(column='income', ax=axes[1, 0])
axes[1, 0].set_title('Income Distribution')
axes[1, 0].set_ylabel('Income ($)')

# Violin plot (combines box plot and density)
sns.violinplot(data=df, y='satisfaction_score', ax=axes[1, 1])
axes[1, 1].set_title('Satisfaction Score Distribution')

plt.tight_layout()
plt.show()
```

### Box Plot Anatomy

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(100, 15, 1000)

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(data, vert=True, patch_artist=True)

# Annotate components
Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

ax.text(1.3, Q1, f'Q1 = {Q1:.1f}', fontsize=10)
ax.text(1.3, Q2, f'Median = {Q2:.1f}', fontsize=10, color='orange')
ax.text(1.3, Q3, f'Q3 = {Q3:.1f}', fontsize=10)
ax.text(1.3, Q1 - 1.5*IQR, f'Lower Whisker', fontsize=10)
ax.text(1.3, Q3 + 1.5*IQR, f'Upper Whisker', fontsize=10)

ax.set_title('Box Plot Anatomy')
ax.set_ylabel('Value')
plt.show()
```

## Big Data Considerations

### Efficient Computation

```python
import pandas as pd
import dask.dataframe as dd

# For massive datasets, use Dask
# Dask DataFrame mimics pandas but computes lazily

# Read large CSV with Dask
ddf = dd.read_csv('large_file.csv')

# Compute statistics (triggers computation)
mean = ddf['column'].mean().compute()
std = ddf['column'].std().compute()

# Or get multiple stats at once
stats = ddf['column'].describe().compute()
print(stats)

# Spark alternative
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev, min, max

spark = SparkSession.builder.getOrCreate()
df_spark = spark.read.csv('large_file.csv', header=True, inferSchema=True)

summary = df_spark.select(
    mean('column').alias('mean'),
    stddev('column').alias('std'),
    min('column').alias('min'),
    max('column').alias('max')
).collect()[0]

print(f"Mean: {summary['mean']:.2f}")
print(f"Std: {summary['std']:.2f}")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Mean** is best for symmetric distributions; **median** for skewed
2. **Standard deviation** quantifies spread around the mean
3. **IQR** is robust to outliers for dispersion
4. **Skewness** and **kurtosis** describe distribution shape
5. **Visualizations** complement numerical summaries
6. **Box plots** efficiently show distribution characteristics
7. Use **Dask or Spark** for massive datasets
:::

## Practical Exercises

See `exercises/chapter-02-exercises.md` for hands-on practice.

## Further Reading

- Downey, A. (2014). "Think Stats" (Free online)
- Wickham, H. & Grolemund, G. (2017). "R for Data Science", Chapter 7
