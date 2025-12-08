# Statistical Inference

## Learning Objectives

- Understand hypothesis testing framework
- Conduct t-tests, z-tests, and chi-square tests
- Calculate and interpret p-values and confidence intervals
- Distinguish between Type I and Type II errors
- Apply statistical tests to real-world problems
- Understand statistical vs practical significance

## Introduction

Statistical inference allows us to draw conclusions about populations based on sample data. It's essential for A/B testing, experimental design, and data-driven decision making.

## Key Concepts

### Point Estimates vs Interval Estimates

**Point Estimate**: Single value estimate (e.g., sample mean)

**Interval Estimate**: Range of plausible values (confidence interval)

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Sample data
sample = np.random.normal(100, 15, 50)

# Point estimate
point_estimate = sample.mean()

# 95% Confidence interval
conf_interval = stats.t.interval(
    confidence=0.95,
    df=len(sample)-1,
    loc=sample.mean(),
    scale=stats.sem(sample)
)

print(f"Point estimate (sample mean): {point_estimate:.2f}")
print(f"95% Confidence Interval: [{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")
print(f"\nInterpretation: We are 95% confident that the true population")
print(f"mean lies between {conf_interval[0]:.2f} and {conf_interval[1]:.2f}")
```

### Standard Error

**Standard Error of the Mean (SEM)**:
\[
\text{SEM} = \frac{s}{\sqrt{n}}
\]

```python
import numpy as np
from scipy import stats

sample = np.array([95, 100, 105, 98, 102, 97, 103])

sample_std = sample.std(ddof=1)  # Sample standard deviation
standard_error = stats.sem(sample)

print(f"Sample std dev: {sample_std:.2f}")
print(f"Standard error: {standard_error:.2f}")
print(f"\nSEM decreases as sample size increases: {sample_std / np.sqrt(len(sample)):.2f}")
```

## Confidence Intervals

### Confidence Interval for Mean (Known σ)

\[
\bar{x} \pm Z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
\]

```python
import numpy as np
from scipy import stats

# Known population std dev
sigma = 15
sample = np.array([98, 102, 105, 95, 100, 97, 103, 101, 99, 104])
n = len(sample)
xbar = sample.mean()

# 95% confidence interval (Z-distribution)
z_critical = stats.norm.ppf(0.975)  # 1.96
margin_of_error = z_critical * sigma / np.sqrt(n)

ci_lower = xbar - margin_of_error
ci_upper = xbar + margin_of_error

print(f"Sample mean: {xbar:.2f}")
print(f"Z-critical (95%): {z_critical:.3f}")
print(f"Margin of error: {margin_of_error:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

### Confidence Interval for Mean (Unknown σ)

\[
\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
\]

```python
import numpy as np
from scipy import stats

# Unknown population std dev (use sample std dev)
sample = np.array([98, 102, 105, 95, 100, 97, 103, 101, 99, 104])
n = len(sample)
xbar = sample.mean()
s = sample.std(ddof=1)  # Sample std dev

# 95% confidence interval (t-distribution)
ci = stats.t.interval(
    confidence=0.95,
    df=n-1,
    loc=xbar,
    scale=stats.sem(sample)
)

print(f"Sample mean: {xbar:.2f}")
print(f"Sample std dev: {s:.2f}")
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

# Different confidence levels
for confidence in [0.90, 0.95, 0.99]:
    ci = stats.t.interval(confidence, df=n-1, loc=xbar, scale=stats.sem(sample))
    width = ci[1] - ci[0]
    print(f"{confidence:.0%} CI: [{ci[0]:.2f}, {ci[1]:.2f}] (width={width:.2f})")
```

### Confidence Interval for Proportion

\[
\hat{p} \pm Z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
\]

```python
import numpy as np
from scipy import stats

# Survey data: 540 out of 1000 support proposal
n = 1000
successes = 540
p_hat = successes / n

# 95% confidence interval
z_critical = stats.norm.ppf(0.975)
standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
margin_of_error = z_critical * standard_error

ci_lower = p_hat - margin_of_error
ci_upper = p_hat + margin_of_error

print(f"Sample proportion: {p_hat:.3f}")
print(f"Standard error: {standard_error:.4f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
```

## Hypothesis Testing

### Framework

1. **Null Hypothesis (H₀)**: Status quo, no effect
2. **Alternative Hypothesis (H₁)**: What we want to prove
3. **Test Statistic**: Measure of evidence
4. **P-value**: Probability of observing data if H₀ is true
5. **Decision**: Reject or fail to reject H₀

### P-Value Interpretation

- **p < 0.05**: Statistically significant (reject H₀)
- **p ≥ 0.05**: Not statistically significant (fail to reject H₀)

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Visualize p-value
z_stat = 2.5  # Test statistic
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Standard Normal')

# Shade rejection regions
plt.fill_between(x[x <= -abs(z_stat)], y[x <= -abs(z_stat)], 
                 alpha=0.3, color='red', label='Rejection Region')
plt.fill_between(x[x >= abs(z_stat)], y[x >= abs(z_stat)], 
                 alpha=0.3, color='red')

plt.axvline(z_stat, color='green', linestyle='--', linewidth=2, 
            label=f'Test Statistic = {z_stat}')
plt.axvline(-z_stat, color='green', linestyle='--', linewidth=2)

plt.xlabel('Z-score')
plt.ylabel('Probability Density')
plt.title(f'P-value = {p_value:.4f}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Test statistic: {z_stat}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Decision: Reject H₀ (statistically significant)")
else:
    print("Decision: Fail to reject H₀ (not significant)")
```

### One-Sample t-Test

**Test if sample mean differs from population mean**

```python
import numpy as np
from scipy import stats

# H₀: μ = 100
# H₁: μ ≠ 100

sample = np.array([105, 110, 98, 102, 107, 103, 99, 106, 101, 104])
population_mean = 100

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(sample, population_mean)

print("One-Sample t-Test:")
print(f"H₀: μ = {population_mean}")
print(f"Sample mean: {sample.mean():.2f}")
print(f"Sample std: {sample.std(ddof=1):.2f}")
print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"\nDecision: Reject H₀ (p={p_value:.4f} < {alpha})")
    print(f"Conclusion: Sample mean is significantly different from {population_mean}")
else:
    print(f"\nDecision: Fail to reject H₀ (p={p_value:.4f} ≥ {alpha})")
    print(f"Conclusion: No significant difference from {population_mean}")
```

### Two-Sample t-Test (Independent)

**Test if two groups have different means**

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# H₀: μ₁ = μ₂ (no difference)
# H₁: μ₁ ≠ μ₂ (difference exists)

# Control vs Treatment
control = np.array([85, 88, 90, 87, 89, 86, 91, 88, 87, 90])
treatment = np.array([92, 95, 93, 94, 96, 93, 95, 94, 92, 95])

# Two-sample t-test (assuming equal variances)
t_stat, p_value = stats.ttest_ind(control, treatment)

print("Two-Sample t-Test:")
print(f"Control: mean={control.mean():.2f}, std={control.std(ddof=1):.2f}")
print(f"Treatment: mean={treatment.mean():.2f}, std={treatment.std(ddof=1):.2f}")
print(f"\nDifference in means: {treatment.mean() - control.mean():.2f}")
print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"\nDecision: Reject H₀ (p={p_value:.6f} < 0.05)")
    print("Conclusion: Significant difference between groups")
else:
    print(f"\nDecision: Fail to reject H₀")
```

### Paired t-Test

**Test differences in matched pairs**

```python
import numpy as np
from scipy import stats

# Before and after measurements
before = np.array([120, 135, 128, 142, 138, 125, 140, 132, 145, 130])
after = np.array([118, 130, 125, 138, 135, 122, 136, 128, 140, 127])

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

differences = before - after

print("Paired t-Test:")
print(f"Before: mean={before.mean():.2f}")
print(f"After: mean={after.mean():.2f}")
print(f"Mean difference: {differences.mean():.2f}")
print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"\nDecision: Reject H₀")
    print("Conclusion: Significant change from before to after")
else:
    print(f"\nDecision: Fail to reject H₀")
```

### Chi-Square Test for Independence

**Test if two categorical variables are independent**

```python
import numpy as np
import pandas as pd
from scipy import stats

# Contingency table: Gender vs Product Preference
contingency_table = pd.DataFrame({
    'Product_A': [50, 30],
    'Product_B': [30, 50],
    'Product_C': [20, 20]
}, index=['Male', 'Female'])

print("Contingency Table:")
print(contingency_table)
print()

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("Chi-Square Test for Independence:")
print(f"Chi-square statistic: {chi2:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected frequencies:")
print(pd.DataFrame(expected, index=contingency_table.index, 
                   columns=contingency_table.columns))

if p_value < 0.05:
    print(f"\nDecision: Reject H₀ (p={p_value:.4f} < 0.05)")
    print("Conclusion: Gender and product preference are dependent")
else:
    print(f"\nDecision: Fail to reject H₀")
    print("Conclusion: No significant association")
```

## Type I and Type II Errors

```python
import pandas as pd

# Error types table
errors = pd.DataFrame({
    'H₀ True': ['Correct Decision\n(True Negative)', 'Type II Error (β)\n(False Negative)'],
    'H₀ False': ['Type I Error (α)\n(False Positive)', 'Correct Decision\n(True Positive)']
}, index=['Fail to Reject H₀', 'Reject H₀'])

print("Decision Matrix:")
print(errors)
print()
print("Type I Error (α): Reject H₀ when it's true (false alarm)")
print("Type II Error (β): Fail to reject H₀ when it's false (miss)")
print(f"\nSignificance level α = P(Type I Error)")
print(f"Power = 1 - β = P(Reject H₀ when false)")
```

## A/B Testing Example

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# A/B test data
control = {'visitors': 10000, 'conversions': 1200}
treatment = {'visitors': 10000, 'conversions': 1350}

# Conversion rates
p_control = control['conversions'] / control['visitors']
p_treatment = treatment['conversions'] / treatment['visitors']

print("A/B Test Results:")
print(f"Control: {control['conversions']}/{control['visitors']} = {p_control:.2%}")
print(f"Treatment: {treatment['conversions']}/{treatment['visitors']} = {p_treatment:.2%}")
print(f"Lift: {(p_treatment / p_control - 1):.1%}")

# Two-proportion z-test
p_pooled = (control['conversions'] + treatment['conversions']) / \
           (control['visitors'] + treatment['visitors'])

se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control['visitors'] + 1/treatment['visitors']))
z_stat = (p_treatment - p_control) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nStatistical Test:")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"\n✅ Result is statistically significant (p < 0.05)")
    print(f"Decision: Launch treatment variant")
else:
    print(f"\n❌ Result is not statistically significant")
    print(f"Decision: Keep control variant")

# Confidence interval for difference
ci_diff = stats.norm.interval(
    0.95,
    loc=p_treatment - p_control,
    scale=se
)

print(f"\n95% CI for difference: [{ci_diff[0]:.3%}, {ci_diff[1]:.3%}]")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
groups = ['Control', 'Treatment']
rates = [p_control, p_treatment]
colors = ['blue', 'green']

ax1.bar(groups, rates, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Conversion Rate')
ax1.set_title('A/B Test Conversion Rates')
ax1.set_ylim(0, max(rates) * 1.2)
for i, (group, rate) in enumerate(zip(groups, rates)):
    ax1.text(i, rate + 0.002, f'{rate:.2%}', ha='center', fontsize=12)
ax1.grid(alpha=0.3, axis='y')

# Confidence intervals
errors = [1.96 * np.sqrt(p * (1-p) / n) for p, n in 
          zip(rates, [control['visitors'], treatment['visitors']])]
ax2.errorbar(groups, rates, yerr=errors, fmt='o', markersize=10, 
             capsize=10, capthick=2, linewidth=2)
ax2.set_ylabel('Conversion Rate')
ax2.set_title('95% Confidence Intervals')
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## Statistical vs Practical Significance

```python
import numpy as np
from scipy import stats

# Large sample: tiny difference is statistically significant
n_large = 1000000
group1_large = np.random.normal(100, 15, n_large)
group2_large = np.random.normal(100.5, 15, n_large)  # 0.5 difference

t_stat, p_value = stats.ttest_ind(group1_large, group2_large)

print("Large Sample (n=1,000,000):")
print(f"Group 1 mean: {group1_large.mean():.3f}")
print(f"Group 2 mean: {group2_large.mean():.3f}")
print(f"Difference: {group2_large.mean() - group1_large.mean():.3f}")
print(f"P-value: {p_value:.10f}")
print(f"Statistically significant: {p_value < 0.05}")
print(f"Practically significant: No! (only 0.5 point difference)")

# Effect size (Cohen's d)
mean_diff = group2_large.mean() - group1_large.mean()
pooled_std = np.sqrt((group1_large.std()**2 + group2_large.std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
print("Interpretation: Negligible effect (d < 0.2)")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Confidence intervals** quantify uncertainty
2. **Hypothesis testing** provides framework for decisions
3. **P-value** ≠ probability that H₀ is true
4. **Statistical significance** ≠ practical importance
5. **Type I error (α)**: False positive (usually 5%)
6. **Type II error (β)**: False negative (power = 1 - β)
7. **Large samples** detect tiny differences
8. **Always report effect sizes** alongside p-values
:::

## Further Reading

- Wasserman, L. (2004). "All of Statistics", Chapters 8-10
- Casella, G. & Berger, R. (2002). "Statistical Inference" (2nd Edition)
- Nuzzo, R. (2014). "Statistical Errors", Nature 506
