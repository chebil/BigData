# Probability Distributions

## Learning Objectives

- Understand discrete and continuous probability distributions
- Work with common distributions (Normal, Binomial, Poisson, etc.)
- Calculate probabilities, expected values, and variances
- Apply distributions to real-world problems
- Visualize and analyze distribution properties
- Use SciPy for distribution calculations

## Introduction

Probability distributions describe how probabilities are distributed over values. They are fundamental to statistics, hypothesis testing, and machine learning algorithms.

## Discrete vs Continuous Distributions

### Discrete Distributions

**Random variable** takes countable values

**Probability Mass Function (PMF)**: \(P(X = x)\)

**Examples**: Number of customers, coin flips, defects

### Continuous Distributions

**Random variable** takes uncountable values (any real number)

**Probability Density Function (PDF)**: \(f(x)\)

Note: \(P(X = x) = 0\) for continuous variables!

Instead: \(P(a \leq X \leq b) = \int_a^b f(x)dx\)

**Examples**: Height, temperature, time

## Discrete Distributions

### 1. Bernoulli Distribution

**Single trial** with two outcomes (success/failure)

**PMF**:
\[
P(X = x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases}
\]

**Mean**: \(E[X] = p\)

**Variance**: \(Var(X) = p(1-p)\)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bernoulli distribution
p = 0.3  # Probability of success

bernoulli_dist = stats.bernoulli(p)

# PMF
x = [0, 1]
pmf = bernoulli_dist.pmf(x)

plt.figure(figsize=(8, 5))
plt.bar(x, pmf, width=0.3, alpha=0.7)
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution (p={p})')
plt.xticks([0, 1], ['Failure', 'Success'])
plt.grid(alpha=0.3, axis='y')
plt.show()

print(f"Mean: {bernoulli_dist.mean()}")
print(f"Variance: {bernoulli_dist.var()}")

# Simulate
samples = bernoulli_dist.rvs(size=1000, random_state=42)
print(f"\nSimulated success rate: {samples.mean():.3f} (expected: {p})")
```

### 2. Binomial Distribution

**n independent Bernoulli trials**

**PMF**:
\[
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
\]

**Mean**: \(E[X] = np\)

**Variance**: \(Var(X) = np(1-p)\)

**Use cases**: Number of successes in fixed trials

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Binomial distribution
n = 20  # Number of trials
p = 0.3  # Probability of success

binom_dist = stats.binom(n, p)

# PMF
x = np.arange(0, n+1)
pmf = binom_dist.pmf(x)

plt.figure(figsize=(10, 6))
plt.bar(x, pmf, alpha=0.7)
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.axvline(binom_dist.mean(), color='r', linestyle='--', 
            label=f'Mean = {binom_dist.mean()}')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.show()

print(f"Mean: {binom_dist.mean()}")
print(f"Variance: {binom_dist.var()}")
print(f"Std Dev: {binom_dist.std():.2f}")

# Example: 20 coin flips
print(f"\nP(exactly 10 heads) = {binom_dist.pmf(10):.4f}")
print(f"P(at least 15 heads) = {1 - binom_dist.cdf(14):.4f}")
print(f"P(5 to 10 heads) = {binom_dist.cdf(10) - binom_dist.cdf(4):.4f}")
```

**Real-world example: Quality Control**

```python
# A factory produces items with 5% defect rate
# Inspect 100 items, what's the probability of finding ≤ 3 defects?

n_items = 100
defect_rate = 0.05

qc_dist = stats.binom(n_items, defect_rate)

P_at_most_3 = qc_dist.cdf(3)
print(f"Quality Control Analysis:")
print(f"P(at most 3 defects in 100 items) = {P_at_most_3:.2%}")
print(f"\nExpected defects: {qc_dist.mean():.1f}")
print(f"Standard deviation: {qc_dist.std():.2f}")

# Probability of unusual result (>10 defects)
P_more_than_10 = 1 - qc_dist.cdf(10)
print(f"\nP(more than 10 defects) = {P_more_than_10:.2%}")
if P_more_than_10 < 0.05:
    print("Finding >10 defects would be unusual - investigate process!")
```

### 3. Poisson Distribution

**Models number of events** in fixed interval

**PMF**:
\[
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\]

where \(\lambda\) = average rate of events

**Mean**: \(E[X] = \lambda\)

**Variance**: \(Var(X) = \lambda\)

**Use cases**: Arrivals, defects, rare events

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Compare different λ values
lambdas = [1, 4, 10]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 6))

for lam, color in zip(lambdas, colors):
    poisson_dist = stats.poisson(lam)
    x = np.arange(0, 25)
    pmf = poisson_dist.pmf(x)
    
    plt.plot(x, pmf, 'o-', color=color, label=f'λ={lam}', alpha=0.7)

plt.xlabel('Number of Events (k)')
plt.ylabel('Probability')
plt.title('Poisson Distribution for Different λ')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("Poisson Distribution Properties:")
for lam in lambdas:
    poisson_dist = stats.poisson(lam)
    print(f"\u03bb={lam}: Mean={poisson_dist.mean()}, Var={poisson_dist.var()}")
```

**Real-world examples**:

```python
from scipy import stats

# Example 1: Website traffic
avg_visitors_per_hour = 50
traffic_dist = stats.poisson(avg_visitors_per_hour)

print("Website Traffic Analysis:")
print(f"P(exactly 50 visitors) = {traffic_dist.pmf(50):.4f}")
print(f"P(at least 60 visitors) = {1 - traffic_dist.cdf(59):.4f}")
print(f"P(40 to 60 visitors) = {traffic_dist.cdf(60) - traffic_dist.cdf(39):.4f}")

# Example 2: Call center
avg_calls_per_minute = 3
call_dist = stats.poisson(avg_calls_per_minute)

print(f"\nCall Center Analysis:")
print(f"Average calls/minute: {avg_calls_per_minute}")
print(f"P(no calls in a minute) = {call_dist.pmf(0):.4f}")
print(f"P(more than 5 calls) = {1 - call_dist.cdf(5):.4f}")

# Example 3: Defects in manufacturing
avg_defects_per_batch = 2.5
defect_dist = stats.poisson(avg_defects_per_batch)

print(f"\nManufacturing Defects:")
print(f"Average defects/batch: {avg_defects_per_batch}")
print(f"P(zero defects) = {defect_dist.pmf(0):.4f}")
print(f"P(≤ 2 defects) = {defect_dist.cdf(2):.4f}")
```

## Continuous Distributions

### 1. Uniform Distribution

**All values equally likely** in interval [a, b]

**PDF**:
\[
f(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
\]

**Mean**: \(E[X] = \frac{a+b}{2}\)

**Variance**: \(Var(X) = \frac{(b-a)^2}{12}\)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Uniform distribution on [0, 10]
a, b = 0, 10
uniform_dist = stats.uniform(loc=a, scale=b-a)

# PDF
x = np.linspace(-2, 12, 1000)
pdf = uniform_dist.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, linewidth=2)
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title(f'Uniform Distribution U({a}, {b})')
plt.axvline(uniform_dist.mean(), color='r', linestyle='--', label='Mean')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Mean: {uniform_dist.mean()}")
print(f"Variance: {uniform_dist.var()}")

# Probabilities (areas under curve)
print(f"\nP(3 ≤ X ≤ 7) = {uniform_dist.cdf(7) - uniform_dist.cdf(3):.2f}")
print(f"P(X ≤ 5) = {uniform_dist.cdf(5):.2f}")
```

### 2. Normal (Gaussian) Distribution

**Most important distribution** in statistics!

**PDF**:
\[
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\]

**Mean**: \(\mu\)

**Variance**: \(\sigma^2\)

**Properties**:
- Bell-shaped, symmetric
- 68-95-99.7 rule
- Central Limit Theorem

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Normal distribution
mu = 100  # Mean
sigma = 15  # Standard deviation

norm_dist = stats.norm(loc=mu, scale=sigma)

# PDF
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm_dist.pdf(x)

plt.figure(figsize=(12, 7))
plt.plot(x, pdf, linewidth=2, label='PDF')
plt.fill_between(x, pdf, alpha=0.3)

# Mark mean and standard deviations
plt.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean (μ={mu})')
for i in range(1, 4):
    plt.axvline(mu + i*sigma, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(mu - i*sigma, color='gray', linestyle=':', alpha=0.5)

# 68-95-99.7 rule
plt.fill_between(x[(x >= mu-sigma) & (x <= mu+sigma)], 
                 pdf[(x >= mu-sigma) & (x <= mu+sigma)], 
                 alpha=0.5, label='68% (±1σ)')
plt.fill_between(x[(x >= mu-2*sigma) & (x <= mu+2*sigma)], 
                 pdf[(x >= mu-2*sigma) & (x <= mu+2*sigma)], 
                 alpha=0.3, label='95% (±2σ)')

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title(f'Normal Distribution N(μ={mu}, σ={sigma})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("68-95-99.7 Rule:")
print(f"P({mu-sigma} ≤ X ≤ {mu+sigma}) = {norm_dist.cdf(mu+sigma) - norm_dist.cdf(mu-sigma):.2%}")
print(f"P({mu-2*sigma} ≤ X ≤ {mu+2*sigma}) = {norm_dist.cdf(mu+2*sigma) - norm_dist.cdf(mu-2*sigma):.2%}")
print(f"P({mu-3*sigma} ≤ X ≤ {mu+3*sigma}) = {norm_dist.cdf(mu+3*sigma) - norm_dist.cdf(mu-3*sigma):.2%}")
```

**Standard Normal Distribution** (Z-distribution):

```python
# Standard normal: μ=0, σ=1
std_norm = stats.norm(0, 1)

# Z-scores
print("\nStandard Normal (Z) Distribution:")
print(f"P(Z ≤ 1.96) = {std_norm.cdf(1.96):.4f}")
print(f"P(Z ≥ 1.96) = {1 - std_norm.cdf(1.96):.4f}")
print(f"P(-1.96 ≤ Z ≤ 1.96) = {std_norm.cdf(1.96) - std_norm.cdf(-1.96):.4f}")

# Convert to Z-score
value = 115
z_score = (value - mu) / sigma
print(f"\nZ-score for x={value}: {z_score:.2f}")
print(f"This is {abs(z_score):.2f} standard deviations above the mean")
```

**Real-world example: IQ Scores**

```python
# IQ scores: μ=100, σ=15
iq_dist = stats.norm(100, 15)

print("IQ Score Analysis:")
print(f"P(IQ > 130) = {1 - iq_dist.cdf(130):.2%}")
print(f"P(IQ < 70) = {iq_dist.cdf(70):.2%}")
print(f"P(85 ≤ IQ ≤ 115) = {iq_dist.cdf(115) - iq_dist.cdf(85):.2%}")

# What IQ score is at 95th percentile?
iq_95th = iq_dist.ppf(0.95)
print(f"\n95th percentile IQ: {iq_95th:.1f}")

# What percentage above 140?
print(f"Percentage with IQ > 140: {(1 - iq_dist.cdf(140)):.2%}")
```

### 3. Exponential Distribution

**Models time between events** in Poisson process

**PDF**:
\[
f(x) = \lambda e^{-\lambda x} \quad \text{for } x \geq 0
\]

**Mean**: \(E[X] = \frac{1}{\lambda}\)

**Variance**: \(Var(X) = \frac{1}{\lambda^2}\)

**Memoryless property**: \(P(X > s+t | X > s) = P(X > t)\)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Exponential distribution
lam = 0.5  # Rate parameter (events per unit time)

exp_dist = stats.expon(scale=1/lam)

# PDF
x = np.linspace(0, 10, 1000)
pdf = exp_dist.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, linewidth=2)
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.title(f'Exponential Distribution (λ={lam})')
plt.axvline(exp_dist.mean(), color='r', linestyle='--', label=f'Mean = {exp_dist.mean():.2f}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Mean (average time between events): {exp_dist.mean():.2f}")
print(f"Variance: {exp_dist.var():.2f}")
```

**Real-world examples**:

```python
# Example 1: Customer service
avg_call_duration = 5  # minutes
call_time_dist = stats.expon(scale=avg_call_duration)

print("Customer Service Call Duration:")
print(f"Average: {avg_call_duration} minutes")
print(f"P(call ≤ 3 minutes) = {call_time_dist.cdf(3):.2%}")
print(f"P(call > 10 minutes) = {1 - call_time_dist.cdf(10):.2%}")

# Example 2: Product lifetime
mean_lifetime = 1000  # hours
lifetime_dist = stats.expon(scale=mean_lifetime)

print(f"\nProduct Lifetime Analysis:")
print(f"Mean lifetime: {mean_lifetime} hours")
print(f"P(fails within 500 hours) = {lifetime_dist.cdf(500):.2%}")
print(f"P(lasts > 2000 hours) = {1 - lifetime_dist.cdf(2000):.2%}")
```

## Comparing Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from different distributions
np.random.seed(42)

distributions = {
    'Normal': stats.norm(50, 10).rvs(1000),
    'Uniform': stats.uniform(30, 40).rvs(1000),
    'Exponential': stats.expon(scale=10).rvs(1000),
    'Poisson': stats.poisson(50).rvs(1000)
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, data) in enumerate(distributions.items()):
    axes[idx].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{name} Distribution', fontsize=14)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')
    axes[idx].axvline(data.mean(), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean={data.mean():.2f}')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\nDistribution Comparison:")
for name, data in distributions.items():
    print(f"\n{name}:")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Std Dev: {data.std():.2f}")
    print(f"  Skewness: {stats.skew(data):.2f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.2f}")
```

## Central Limit Theorem

**Key insight**: Sum/mean of many independent random variables tends toward normal distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Start with uniform distribution (NOT normal!)
population = stats.uniform(0, 10)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sample_sizes = [1, 2, 5, 10, 30, 100]

for idx, n in enumerate(sample_sizes):
    # Sample means
    sample_means = [population.rvs(n).mean() for _ in range(10000)]
    
    ax = axes[idx // 3, idx % 3]
    ax.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay normal distribution
    mu = population.mean()
    sigma = population.std() / np.sqrt(n)
    x = np.linspace(min(sample_means), max(sample_means), 100)
    ax.plot(x, stats.norm(mu, sigma).pdf(x), 'r-', linewidth=2, label='Theoretical')
    
    ax.set_title(f'Sample Size n={n}')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Central Limit Theorem: Distribution of Sample Means', fontsize=16)
plt.tight_layout()
plt.show()

print("Central Limit Theorem Demonstration:")
print("Even though population is uniform, sample means approach normal!")
print(f"Population mean: {population.mean()}")
print(f"Population std: {population.std():.3f}")
for n in [10, 30, 100]:
    print(f"\nSample size n={n}:")
    print(f"  Expected std of sample mean: {population.std() / np.sqrt(n):.3f}")
```

## Q-Q Plots: Testing Normality

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate different datasets
datasets = {
    'Normal': stats.norm(0, 1).rvs(200),
    'Uniform': stats.uniform(-2, 4).rvs(200),
    'Exponential': stats.expon(scale=1).rvs(200),
    'T-distribution': stats.t(df=3).rvs(200)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, data) in enumerate(datasets.items()):
    stats.probplot(data, dist="norm", plot=axes[idx])
    axes[idx].set_title(f'Q-Q Plot: {name} Data')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Q-Q Plot Interpretation:")
print("- Points on line → data is normally distributed")
print("- Curve → data is not normal")
print("\nShapiro-Wilk Test for Normality:")
for name, data in datasets.items():
    statistic, p_value = stats.shapiro(data)
    print(f"{name:15s}: p-value = {p_value:.4f}", end="")
    if p_value > 0.05:
        print(" → Normal")
    else:
        print(" → Not Normal")
```

## Practical Applications

### A/B Testing with Normal Approximation

```python
import numpy as np
from scipy import stats

# Conversion rates
control = {'n': 10000, 'conversions': 1200}
treatment = {'n': 10000, 'conversions': 1350}

# Proportions
p1 = control['conversions'] / control['n']
p2 = treatment['conversions'] / treatment['n']

# Standard errors
se1 = np.sqrt(p1 * (1 - p1) / control['n'])
se2 = np.sqrt(p2 * (1 - p2) / treatment['n'])

# Z-test for difference
se_diff = np.sqrt(se1**2 + se2**2)
z_score = (p2 - p1) / se_diff
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print("A/B Test Results:")
print(f"Control: {p1:.2%}")
print(f"Treatment: {p2:.2%}")
print(f"Lift: {(p2/p1 - 1):.1%}")
print(f"\nZ-score: {z_score:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\nResult is statistically significant!")
else:
    print("\nResult is not statistically significant")
```

### Confidence Intervals

```python
import numpy as np
from scipy import stats

# Sample data
np.random.seed(42)
sample = stats.norm(100, 15).rvs(50)

# Calculate 95% confidence interval
mean = sample.mean()
std_err = stats.sem(sample)  # Standard error of mean
confidence = 0.95

# Using t-distribution (small sample)
ci = stats.t.interval(confidence, df=len(sample)-1, loc=mean, scale=std_err)

print(f"Sample mean: {mean:.2f}")
print(f"Standard error: {std_err:.2f}")
print(f"95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"\nInterpretation: We are 95% confident that the true population")
print(f"mean lies between {ci[0]:.2f} and {ci[1]:.2f}")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Discrete distributions** (PMF): Binomial, Poisson, Bernoulli
2. **Continuous distributions** (PDF): Normal, Uniform, Exponential
3. **Normal distribution** is most important (Central Limit Theorem)
4. **68-95-99.7 rule** for normal distribution
5. **Poisson** for rare events, **Exponential** for wait times
6. **Central Limit Theorem**: sample means → normal
7. **Q-Q plots** test normality assumption
8. **SciPy stats** provides all distribution functions
:::

## Further Reading

- Wasserman, L. (2004). "All of Statistics", Chapter 3
- Ross, S. (2014). "A First Course in Probability", Chapters 4-5
- SciPy Stats Documentation: [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
