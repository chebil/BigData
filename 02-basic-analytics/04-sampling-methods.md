# Sampling Methods

## Learning Objectives

- Understand different sampling techniques
- Apply appropriate sampling methods to Big Data
- Calculate sample sizes for desired precision
- Recognize and avoid sampling bias
- Implement stratified, systematic, and cluster sampling
- Use reservoir sampling for streaming data

## Introduction

In Big Data analytics, we often cannot process the entire population due to computational constraints, time, or cost. **Sampling** allows us to draw inferences about a population from a representative subset.

## Population vs Sample

**Population**: Complete set of all items

**Sample**: Subset of the population

**Goal**: Make inferences about population from sample

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Population: All customers (millions)
np.random.seed(42)
population_size = 1000000
population = pd.DataFrame({
    'customer_id': range(population_size),
    'age': np.random.normal(45, 15, population_size),
    'income': np.random.lognormal(10.5, 0.5, population_size),
    'purchases': np.random.poisson(5, population_size)
})

print(f"Population size: {len(population):,}")
print(f"\nPopulation parameters:")
print(f"Mean age: {population['age'].mean():.2f}")
print(f"Mean income: ${population['income'].mean():,.2f}")
print(f"Mean purchases: {population['purchases'].mean():.2f}")

# Sample: 1000 customers
sample_size = 1000
sample = population.sample(n=sample_size, random_state=42)

print(f"\nSample size: {len(sample):,}")
print(f"\nSample statistics (estimates):")
print(f"Mean age: {sample['age'].mean():.2f}")
print(f"Mean income: ${sample['income'].mean():,.2f}")
print(f"Mean purchases: {sample['purchases'].mean():.2f}")

print(f"\nSample represents {sample_size/population_size:.2%} of population")
```

## Sampling Methods

### 1. Simple Random Sampling (SRS)

**Every member has equal probability** of selection

```python
import numpy as np
import pandas as pd

# Simple random sample
def simple_random_sample(data, n, random_state=None):
    return data.sample(n=n, random_state=random_state)

# Example
sample_srs = simple_random_sample(population, n=1000, random_state=42)

print("Simple Random Sampling:")
print(f"Sample size: {len(sample_srs)}")
print(f"Mean age: {sample_srs['age'].mean():.2f}")
```

**Advantages**:
- ✅ Unbiased
- ✅ Simple to implement
- ✅ Representative if sample size is large

**Disadvantages**:
- ❌ May not represent small subgroups
- ❌ Requires complete population list
- ❌ Can be inefficient

### 2. Stratified Sampling

**Divide population into strata**, then sample from each

```python
import pandas as pd
import numpy as np

def stratified_sample(data, strata_col, n, random_state=None):
    """
    Stratified sampling: proportional allocation
    """
    return data.groupby(strata_col, group_keys=False).apply(
        lambda x: x.sample(frac=n/len(data), random_state=random_state)
    )

# Create age groups (strata)
population['age_group'] = pd.cut(population['age'], 
                                  bins=[0, 30, 50, 100],
                                  labels=['Young', 'Middle', 'Senior'])

print("Population age distribution:")
print(population['age_group'].value_counts(normalize=True))

# Stratified sample
sample_stratified = stratified_sample(population, 'age_group', n=1000, random_state=42)

print("\nStratified sample age distribution:")
print(sample_stratified['age_group'].value_counts(normalize=True))

print("\nComparison:")
print(f"Population mean age: {population['age'].mean():.2f}")
print(f"SRS mean age: {sample_srs['age'].mean():.2f}")
print(f"Stratified mean age: {sample_stratified['age'].mean():.2f}")
```

**Advantages**:
- ✅ Ensures representation of subgroups
- ✅ More precise than SRS
- ✅ Can compare strata

**Disadvantages**:
- ❌ Requires knowledge of population distribution
- ❌ More complex implementation

### 3. Systematic Sampling

**Select every k-th element**

```python
def systematic_sample(data, n):
    """
    Systematic sampling: select every k-th element
    """
    k = len(data) // n  # Sampling interval
    start = np.random.randint(0, k)
    indices = np.arange(start, len(data), k)[:n]
    return data.iloc[indices]

# Systematic sample
sample_systematic = systematic_sample(population, n=1000)

print("Systematic Sampling:")
print(f"Sampling interval k = {len(population) // 1000}")
print(f"Sample size: {len(sample_systematic)}")
print(f"Mean income: ${sample_systematic['income'].mean():,.2f}")
```

**Advantages**:
- ✅ Simple and quick
- ✅ Spreads sample across population

**Disadvantages**:
- ❌ Can introduce bias if data has periodicity
- ❌ Not truly random

### 4. Cluster Sampling

**Divide population into clusters**, randomly select clusters

```python
import numpy as np

# Create geographic clusters
population['region'] = np.random.choice(['North', 'South', 'East', 'West'], 
                                        len(population))

def cluster_sample(data, cluster_col, n_clusters, random_state=None):
    """
    Cluster sampling: randomly select clusters
    """
    np.random.seed(random_state)
    all_clusters = data[cluster_col].unique()
    selected_clusters = np.random.choice(all_clusters, n_clusters, replace=False)
    return data[data[cluster_col].isin(selected_clusters)]

# Cluster sample: select 2 out of 4 regions
sample_cluster = cluster_sample(population, 'region', n_clusters=2, random_state=42)

print("Cluster Sampling:")
print(f"Total clusters: {population['region'].nunique()}")
print(f"Selected clusters: {sample_cluster['region'].unique()}")
print(f"Sample size: {len(sample_cluster):,}")
```

**Advantages**:
- ✅ Cost-effective for geographically dispersed populations
- ✅ Practical for large populations

**Disadvantages**:
- ❌ Higher variance than SRS
- ❌ Clusters must be representative

### 5. Reservoir Sampling (For Streaming Data)

**Sample from unknown or infinite stream**

```python
import numpy as np

def reservoir_sampling(stream, k, random_state=None):
    """
    Reservoir sampling: maintain sample of size k from stream
    Each element has equal probability k/n of being in sample
    """
    np.random.seed(random_state)
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            # Fill reservoir
            reservoir.append(item)
        else:
            # Randomly replace elements
            j = np.random.randint(0, i + 1)
            if j < k:
                reservoir[j] = item
    
    return reservoir

# Simulate data stream
stream = (i for i in range(1000000))  # 1 million items

# Sample 1000 items
sample_reservoir = reservoir_sampling(stream, k=1000, random_state=42)

print("Reservoir Sampling (Streaming):")
print(f"Sample size: {len(sample_reservoir)}")
print(f"Sample mean: {np.mean(sample_reservoir):.2f}")
print(f"Expected mean: {999999/2:.2f}")  # True mean of 0 to 999999
```

**Use case**: Twitter streams, log files, real-time data

## Sample Size Determination

### For Estimating a Mean

\[
n = \left(\frac{Z \cdot \sigma}{E}\right)^2
\]

where:
- \(n\) = required sample size
- \(Z\) = Z-score (e.g., 1.96 for 95% confidence)
- \(\sigma\) = population standard deviation
- \(E\) = margin of error

```python
import numpy as np
from scipy import stats

def sample_size_mean(sigma, margin_of_error, confidence=0.95):
    """
    Calculate required sample size for estimating a mean
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z * sigma / margin_of_error) ** 2
    return int(np.ceil(n))

# Example: Average customer age
sigma_age = 15  # Estimated population std dev
margin = 1      # Want estimate within ±1 year

n_required = sample_size_mean(sigma_age, margin, confidence=0.95)

print("Sample Size Calculation for Mean:")
print(f"Population std dev: {sigma_age}")
print(f"Desired margin of error: ±{margin}")
print(f"Confidence level: 95%")
print(f"\nRequired sample size: {n_required}")

# Different scenarios
scenarios = [
    (15, 1, 0.95),
    (15, 0.5, 0.95),
    (15, 1, 0.99),
]

print("\nSample size for different scenarios:")
for sigma, margin, conf in scenarios:
    n = sample_size_mean(sigma, margin, conf)
    print(f"σ={sigma}, E=±{margin}, confidence={conf:.0%}: n={n}")
```

### For Estimating a Proportion

\[
n = \frac{Z^2 \cdot p(1-p)}{E^2}
\]

```python
def sample_size_proportion(p, margin_of_error, confidence=0.95):
    """
    Calculate required sample size for estimating a proportion
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z ** 2 * p * (1 - p)) / (margin_of_error ** 2)
    return int(np.ceil(n))

# Example: Conversion rate estimation
p_estimated = 0.5  # Conservative estimate (maximizes sample size)
margin = 0.03      # Want estimate within ±3%

n_required = sample_size_proportion(p_estimated, margin, confidence=0.95)

print("\nSample Size Calculation for Proportion:")
print(f"Estimated proportion: {p_estimated}")
print(f"Desired margin of error: ±{margin:.1%}")
print(f"Confidence level: 95%")
print(f"\nRequired sample size: {n_required}")

# Common margins
for margin in [0.01, 0.03, 0.05]:
    n = sample_size_proportion(0.5, margin, 0.95)
    print(f"Margin ±{margin:.0%}: n={n}")
```

## Sampling Bias

### Types of Bias

1. **Selection Bias**: Systematic exclusion of certain groups
2. **Nonresponse Bias**: Sampled individuals don't respond
3. **Voluntary Response Bias**: Self-selected participants
4. **Survivorship Bias**: Only analyzing "survivors"

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Demonstrate selection bias
np.random.seed(42)

# True population: Age is uniformly distributed
population = pd.DataFrame({
    'age': np.random.uniform(18, 80, 10000),
    'online': np.random.rand(10000) < 0.8  # 80% are online
})

# Younger people more likely to be online
population.loc[population['age'] < 40, 'online'] = np.random.rand((population['age'] < 40).sum()) < 0.95
population.loc[population['age'] > 60, 'online'] = np.random.rand((population['age'] > 60).sum()) < 0.50

# Biased sample: only survey online users
biased_sample = population[population['online'] == True].sample(1000)

# Unbiased sample: random sample from all
unbiased_sample = population.sample(1000)

# Compare
plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.hist(population['age'], bins=30, alpha=0.7, edgecolor='black')
plt.title('True Population')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.axvline(population['age'].mean(), color='r', linestyle='--', 
            label=f'Mean={population["age"].mean():.1f}')
plt.legend()

plt.subplot(132)
plt.hist(biased_sample['age'], bins=30, alpha=0.7, edgecolor='black', color='orange')
plt.title('Biased Sample (Online Only)')
plt.xlabel('Age')
plt.axvline(biased_sample['age'].mean(), color='r', linestyle='--',
            label=f'Mean={biased_sample["age"].mean():.1f}')
plt.legend()

plt.subplot(133)
plt.hist(unbiased_sample['age'], bins=30, alpha=0.7, edgecolor='black', color='green')
plt.title('Unbiased Random Sample')
plt.xlabel('Age')
plt.axvline(unbiased_sample['age'].mean(), color='r', linestyle='--',
            label=f'Mean={unbiased_sample["age"].mean():.1f}')
plt.legend()

plt.tight_layout()
plt.show()

print("Selection Bias Example:")
print(f"True population mean age: {population['age'].mean():.2f}")
print(f"Biased sample mean age: {biased_sample['age'].mean():.2f}")
print(f"Unbiased sample mean age: {unbiased_sample['age'].mean():.2f}")
print(f"\nBias: {biased_sample['age'].mean() - population['age'].mean():.2f} years")
```

## Big Data Sampling Strategies

### Sampling with Pandas

```python
import pandas as pd
import numpy as np

# Large dataset
np.random.seed(42)
big_data = pd.DataFrame({
    'user_id': range(10000000),
    'revenue': np.random.lognormal(5, 2, 10000000)
})

print(f"Original dataset: {len(big_data):,} rows")
print(f"Memory usage: {big_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Sample 1%
sample = big_data.sample(frac=0.01, random_state=42)

print(f"\nSample: {len(sample):,} rows")
print(f"Memory usage: {sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Statistics comparison
print(f"\nPopulation mean revenue: ${big_data['revenue'].mean():.2f}")
print(f"Sample mean revenue: ${sample['revenue'].mean():.2f}")
```

### Sampling with Dask (Parallel)

```python
import dask.dataframe as dd
import numpy as np

# Create large Dask dataframe
# ddf = dd.read_parquet('large_dataset.parquet')

# Sample 10%
# sample = ddf.sample(frac=0.1).compute()

print("Dask sampling for very large datasets")
print("Allows sampling from datasets larger than memory")
```

### Sampling with Spark

```python
# PySpark sampling
from pyspark.sql import SparkSession

# Initialize Spark
# spark = SparkSession.builder.appName("Sampling").getOrCreate()

# Load data
# df = spark.read.parquet("hdfs://large_dataset")

# Simple random sample (10%)
# sample = df.sample(fraction=0.1, seed=42)

# Stratified sample
# sample_stratified = df.sampleBy("category", fractions={'A': 0.1, 'B': 0.2}, seed=42)

print("Spark sampling for distributed Big Data")
print("Efficient sampling across cluster")
```

## Bootstrapping

**Resampling with replacement** to estimate sampling distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Original sample
sample = np.random.normal(100, 15, 50)
sample_mean = sample.mean()

# Bootstrap resampling
n_bootstrap = 10000
bootstrap_means = []

for _ in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = np.random.choice(sample, size=len(sample), replace=True)
    bootstrap_means.append(bootstrap_sample.mean())

bootstrap_means = np.array(bootstrap_means)

# Bootstrap confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(sample_mean, color='r', linestyle='--', linewidth=2, 
            label=f'Sample Mean = {sample_mean:.2f}')
plt.axvline(ci_lower, color='g', linestyle='--', linewidth=2)
plt.axvline(ci_upper, color='g', linestyle='--', linewidth=2, 
            label=f'95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]')
plt.xlabel('Bootstrap Sample Mean')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Sample Mean')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("Bootstrap Analysis:")
print(f"Original sample mean: {sample_mean:.2f}")
print(f"Bootstrap mean: {bootstrap_means.mean():.2f}")
print(f"Bootstrap std error: {bootstrap_means.std():.2f}")
print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Sampling saves time and resources** while maintaining accuracy
2. **Simple random sampling** is unbiased but may miss subgroups
3. **Stratified sampling** ensures representation of important groups
4. **Sample size depends on** desired precision and confidence level
5. **Avoid sampling bias** through proper design
6. **Reservoir sampling** handles streaming data
7. **Bootstrapping** estimates uncertainty without assumptions
8. **For Big Data**, use Dask or Spark for efficient sampling
:::

## Further Reading

- Cochran, W. (1977). "Sampling Techniques" (3rd Edition)
- Thompson, S. (2012). "Sampling" (3rd Edition)
- Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap"
