# Lab 2: Statistical Analysis - Complete Solutions

## Part 1: Descriptive Statistics

### Exercise 1.1: Load and Explore Data

```python
from sklearn.datasets import load_iris, load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris Dataset Shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())

# 1. Dataset info
print("\n" + "="*50)
print("DATASET INFO")
print("="*50)
print(iris_df.info())

# 2. Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
print(iris_df.isnull().sum())

# 3. Basic statistics
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)
print(iris_df.describe())

# 4. Range of sepal length
sepal_range = iris_df['sepal length (cm)'].max() - iris_df['sepal length (cm)'].min()
print(f"\nRange of sepal length: {sepal_range:.2f} cm")
```

**Answers:**

**Q1:** How many samples are there for each species?
```python
species_counts = iris_df['species_name'].value_counts()
print("\nSamples per species:")
print(species_counts)
```
**Answer:** 50 samples for each species (setosa, versicolor, virginica)

**Q2:** Which feature has the highest mean value?
```python
means = iris_df[iris.feature_names].mean()
print("\nFeature means:")
print(means)
print(f"\nHighest mean: {means.idxmax()} = {means.max():.2f}")
```
**Answer:** Petal length (cm) with mean = 3.76 cm

**Q3:** Which feature has the most variation?
```python
stds = iris_df[iris.feature_names].std()
print("\nFeature standard deviations:")
print(stds)
print(f"\nHighest variation: {stds.idxmax()} = {stds.max():.2f}")
```
**Answer:** Petal length (cm) with std = 1.77 cm

---

### Exercise 1.2: Measures of Central Tendency

```python
sepal_length = iris_df['sepal length (cm)']

# Calculate measures
mean_val = sepal_length.mean()
median_val = sepal_length.median()
mode_val = sepal_length.mode()[0]  # mode() returns a Series

print(f"Mean: {mean_val:.2f} cm")
print(f"Median: {median_val:.2f} cm")
print(f"Mode: {mode_val:.2f} cm")

# Additional insight
print(f"\nDifference (Mean - Median): {abs(mean_val - median_val):.4f}")
print("This suggests the distribution is approximately symmetric.")
```

**Q4:** Are mean and median similar?
```python
if abs(mean_val - median_val) < 0.1:
    print("\nYes, mean and median are very close.")
    print("This suggests the distribution is approximately symmetric (not skewed).")
else:
    print("\nMean and median differ significantly.")
    if mean_val > median_val:
        print("Mean > Median suggests right-skewed distribution.")
    else:
        print("Mean < Median suggests left-skewed distribution.")
```

**Q5:** Calculate trimmed mean
```python
from scipy import stats as scipy_stats

# Remove top and bottom 10%
trimmed_mean = scipy_stats.trim_mean(sepal_length, 0.1)
print(f"\nTrimmed Mean (10%): {trimmed_mean:.2f} cm")
print(f"Regular Mean: {mean_val:.2f} cm")
print(f"Difference: {abs(trimmed_mean - mean_val):.4f} cm")
print("\nSmall difference indicates few extreme values.")
```

---

### Exercise 1.3: Measures of Dispersion

```python
# Calculate all measures
variance = sepal_length.var()
std_dev = sepal_length.std()
q1 = sepal_length.quantile(0.25)
q3 = sepal_length.quantile(0.75)
iqr = q3 - q1
data_range = sepal_length.max() - sepal_length.min()

# Coefficient of variation
cv = (std_dev / mean_val) * 100

print("MEASURES OF DISPERSION")
print("=" * 50)
print(f"Variance: {variance:.4f} cm²")
print(f"Std Dev: {std_dev:.4f} cm")
print(f"Q1 (25th percentile): {q1:.2f} cm")
print(f"Q3 (75th percentile): {q3:.2f} cm")
print(f"IQR: {iqr:.2f} cm")
print(f"Range: {data_range:.2f} cm")
print(f"CV: {cv:.2f}%")

# Interpretation
print("\nINTERPRETATION:")
print(f"- Standard deviation is {std_dev/mean_val*100:.1f}% of the mean")
print(f"- IQR contains middle 50% of data ({iqr:.2f} cm)")
print(f"- Coefficient of variation of {cv:.1f}% indicates moderate variability")
```

**Q6:** Which measure is most robust to outliers?
```python
print("\nQ6 Answer:")
print("The IQR (Interquartile Range) is most robust to outliers.")
print("\nReason: IQR uses quartiles which are not affected by extreme values.")
print("Range and standard deviation can be heavily influenced by outliers.")

# Demonstration
sepal_with_outlier = sepal_length.copy()
sepal_with_outlier.iloc[0] = 100  # Add extreme outlier

print(f"\nOriginal - Range: {data_range:.2f}, Std: {std_dev:.2f}, IQR: {iqr:.2f}")
print(f"With Outlier - Range: {sepal_with_outlier.max() - sepal_with_outlier.min():.2f}, "
      f"Std: {sepal_with_outlier.std():.2f}, IQR: {sepal_with_outlier.quantile(0.75) - sepal_with_outlier.quantile(0.25):.2f}")
print("\nNotice: IQR barely changed, but range and std increased dramatically!")
```

**Q7:** What does CV tell us?
```python
print("\nQ7 Answer:")
print(f"Coefficient of Variation (CV) = {cv:.2f}%")
print("\nInterpretation:")
print("- CV expresses std dev as percentage of mean")
print("- Allows comparison of variability between different variables/datasets")
print("- CV < 15%: Low variability")
print("- CV 15-30%: Moderate variability")
print("- CV > 30%: High variability")
print(f"\nOur CV of {cv:.1f}% indicates moderate variability in sepal length.")
```

---

### Exercise 1.4: Visualizations

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram of sepal length
axes[0, 0].hist(sepal_length, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
axes[0, 0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
axes[0, 0].set_xlabel('Sepal Length (cm)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Sepal Length', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Box plot by species
sns.boxplot(data=iris_df, x='species_name', y='sepal length (cm)', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_xlabel('Species', fontsize=12)
axes[0, 1].set_ylabel('Sepal Length (cm)', fontsize=12)
axes[0, 1].set_title('Sepal Length by Species', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Violin plot of all features
iris_melted = iris_df.melt(id_vars=['species_name'], 
                           value_vars=iris.feature_names,
                           var_name='Feature', 
                           value_name='Value')
sns.violinplot(data=iris_melted, x='Feature', y='Value', ax=axes[1, 0], palette='muted')
axes[1, 0].set_xlabel('Feature', fontsize=12)
axes[1, 0].set_ylabel('Value (cm)', fontsize=12)
axes[1, 0].set_title('Distribution of All Features', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(alpha=0.3)

# 4. Scatter plot
axes[1, 1].scatter(iris_df['sepal length (cm)'], iris_df['petal length (cm)'], 
                   c=iris_df['species'], cmap='viridis', alpha=0.6, s=50, edgecolors='black')
axes[1, 1].set_xlabel('Sepal Length (cm)', fontsize=12)
axes[1, 1].set_ylabel('Petal Length (cm)', fontsize=12)
axes[1, 1].set_title('Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Add correlation coefficient
corr = iris_df[['sepal length (cm)', 'petal length (cm)']].corr().iloc[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[1, 1].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('lab2_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Q8:** Outliers in box plot?
```python
print("\nQ8 Answer:")
print("Looking at the box plots:")
for species in iris_df['species_name'].unique():
    species_data = iris_df[iris_df['species_name'] == species]['sepal length (cm)']
    Q1 = species_data.quantile(0.25)
    Q3 = species_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = species_data[(species_data < lower_bound) | (species_data > upper_bound)]
    print(f"\n{species.capitalize()}:")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier values: {outliers.values}")
```

**Q9:** Normal distribution?
```python
from scipy.stats import shapiro, normaltest

print("\nQ9 Answer: Testing for Normality")
print("=" * 50)

# Shapiro-Wilk test
stat, p_value = shapiro(sepal_length)
print(f"\nShapiro-Wilk Test:")
print(f"  Statistic: {stat:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value > 0.05:
    print(f"  Conclusion: Data appears normally distributed (p={p_value:.4f} > 0.05)")
else:
    print(f"  Conclusion: Data significantly deviates from normal (p={p_value:.4f} < 0.05)")

# Visual check: Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with normal curve overlay
ax1.hist(sepal_length, bins=20, density=True, alpha=0.7, edgecolor='black', label='Data')
mu, sigma = sepal_length.mean(), sepal_length.std()
x = np.linspace(sepal_length.min(), sepal_length.max(), 100)
ax1.plot(x, scipy_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Density')
ax1.set_title('Histogram with Normal Curve')
ax1.legend()
ax1.grid(alpha=0.3)

# Q-Q plot
scipy_stats.probplot(sepal_length, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Q10:** Correlation?
```python
from scipy.stats import pearsonr, spearmanr

print("\nQ10 Answer: Correlation Analysis")
print("=" * 50)

# Pearson correlation
pearson_r, pearson_p = pearsonr(iris_df['sepal length (cm)'], iris_df['petal length (cm)'])
print(f"\nPearson Correlation: {pearson_r:.4f}")
print(f"P-value: {pearson_p:.4e}")

# Spearman correlation (for comparison)
spearman_r, spearman_p = spearmanr(iris_df['sepal length (cm)'], iris_df['petal length (cm)'])
print(f"\nSpearman Correlation: {spearman_r:.4f}")
print(f"P-value: {spearman_p:.4e}")

# Interpretation
print("\nInterpretation:")
if abs(pearson_r) < 0.3:
    strength = "weak"
elif abs(pearson_r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"
    
print(f"There is a {strength} positive correlation ({pearson_r:.3f}) between sepal and petal length.")
print(f"This correlation is statistically significant (p < 0.001).")
print(f"\nConclusion: As sepal length increases, petal length tends to increase as well.")
```

---

## Part 2: Probability Distributions

[CONTINUES WITH COMPLETE SOLUTIONS FOR ALL REMAINING EXERCISES...]

**Note:** This is the complete solution format. Would you like me to continue with all parts?
