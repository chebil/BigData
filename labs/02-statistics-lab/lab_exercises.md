# Lab 2: Statistical Analysis - Exercises

## Part 1: Descriptive Statistics (20 points)

### Exercise 1.1: Load and Explore Data (5 points)

```python
from sklearn.datasets import load_iris, load_wine
import pandas as pd

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris Dataset Shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())
```

**Tasks:**
1. Display the dataset info (data types, non-null counts)
2. Check for missing values
3. Display basic statistics using `.describe()`
4. What is the range of sepal length?

**Questions:**
- Q1: How many samples are there for each species?
- Q2: Which feature has the highest mean value?
- Q3: Which feature has the most variation (highest standard deviation)?

### Exercise 1.2: Calculate Measures of Central Tendency (5 points)

**Task:** For the 'sepal length (cm)' column:

```python
sepal_length = iris_df['sepal length (cm)']

# TODO: Calculate the following
mean_val = ??
median_val = ??
mode_val = ??

print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Mode: {mode_val:.2f}")
```

**Questions:**
- Q4: Are mean and median similar? What does this tell you about the distribution?
- Q5: Calculate the trimmed mean (remove top and bottom 10%)

### Exercise 1.3: Measures of Dispersion (5 points)

**Task:** Calculate variance, standard deviation, IQR, and range

```python
# TODO: Complete the calculations
variance = ??
std_dev = ??
q1 = ??
q3 = ??
iqr = ??
data_range = ??

# Coefficient of variation
cv = (std_dev / mean_val) * 100

print(f"Variance: {variance:.2f}")
print(f"Std Dev: {std_dev:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Range: {data_range:.2f}")
print(f"CV: {cv:.2f}%")
```

**Questions:**
- Q6: Which measure of spread is most robust to outliers?
- Q7: What does the coefficient of variation tell us?

### Exercise 1.4: Visualizations (5 points)

**Task:** Create the following plots:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# TODO: Create these visualizations
# 1. Histogram of sepal length (axes[0, 0])
# 2. Box plot of sepal length by species (axes[0, 1])
# 3. Violin plot of all features (axes[1, 0])
# 4. Scatter plot: sepal length vs petal length (axes[1, 1])

plt.tight_layout()
plt.show()
```

**Questions:**
- Q8: Are there any outliers visible in the box plot?
- Q9: Does sepal length appear normally distributed?
- Q10: Is there a correlation between sepal length and petal length?

---

## Part 2: Probability Distributions (20 points)

### Exercise 2.1: Normal Distribution (7 points)

**Scenario:** Heights of adult males are normally distributed with mean = 175 cm and std = 7 cm.

```python
from scipy import stats

mu = 175
sigma = 7

# TODO: Answer the following
# 1. What percentage of males are taller than 180 cm?
prob_taller_180 = ??

# 2. What percentage are between 170 and 180 cm?
prob_between = ??

# 3. What height corresponds to the 90th percentile?
height_90th = ??

# 4. Generate 1000 random samples and plot histogram
samples = ??

print(f"P(X > 180): {prob_taller_180:.2%}")
print(f"P(170 < X < 180): {prob_between:.2%}")
print(f"90th percentile: {height_90th:.2f} cm")
```

**Questions:**
- Q11: What is the probability of finding someone taller than 190 cm?
- Q12: Between what two heights do 95% of males fall? (Hint: 95% confidence interval)

### Exercise 2.2: Binomial Distribution (7 points)

**Scenario:** A multiple-choice test has 10 questions, each with 4 options. If a student guesses randomly:

```python
n = 10  # number of questions
p = 0.25  # probability of correct guess

# TODO: Calculate the following
# 1. Probability of getting exactly 5 correct
prob_5_correct = ??

# 2. Probability of getting at least 7 correct
prob_at_least_7 = ??

# 3. Expected number of correct answers
expected = ??

# 4. Standard deviation
std = ??

print(f"P(X = 5): {prob_5_correct:.4f}")
print(f"P(X >= 7): {prob_at_least_7:.4f}")
print(f"Expected: {expected:.2f}")
print(f"Std Dev: {std:.2f}")

# Plot PMF
x = range(0, 11)
pmf = [stats.binom.pmf(k, n, p) for k in x]

plt.figure(figsize=(10, 6))
plt.bar(x, pmf, alpha=0.7, edgecolor='black')
plt.xlabel('Number of Correct Answers')
plt.ylabel('Probability')
plt.title('Binomial Distribution: Random Guessing')
plt.grid(alpha=0.3)
plt.show()
```

**Questions:**
- Q13: Would you consider getting 7+ correct answers as evidence against random guessing? Why?
- Q14: What's the probability of passing (≥6 correct)?

### Exercise 2.3: Poisson Distribution (6 points)

**Scenario:** A website receives an average of 5 visitors per hour.

```python
lambda_rate = 5  # average per hour

# TODO: Calculate
# 1. Probability of exactly 3 visitors in next hour
prob_3 = ??

# 2. Probability of more than 8 visitors
prob_more_8 = ??

# 3. Probability of 0 visitors (no traffic)
prob_0 = ??

print(f"P(X = 3): {prob_3:.4f}")
print(f"P(X > 8): {prob_more_8:.4f}")
print(f"P(X = 0): {prob_0:.4f}")
```

**Questions:**
- Q15: What's the expected number of visitors in 2 hours?
- Q16: In a 30-minute period, what's the probability of at least 1 visitor?

---

## Part 3: Hypothesis Testing (30 points)

### Exercise 3.1: One-Sample t-test (10 points)

**Scenario:** A manufacturer claims their light bulbs last 1000 hours on average. You test 25 bulbs.

```python
np.random.seed(42)
# Simulated data (in reality, you'd have actual measurements)
bulb_lifetimes = np.random.normal(950, 100, 25)

print(f"Sample mean: {bulb_lifetimes.mean():.2f}")
print(f"Sample std: {bulb_lifetimes.std():.2f}")

# TODO: Perform one-sample t-test
# H0: μ = 1000 (manufacturer's claim is correct)
# H1: μ ≠ 1000 (manufacturer's claim is incorrect)

t_statistic, p_value = ??

print(f"\nt-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"\nReject H0: Evidence suggests bulbs don't last 1000 hours (p={p_value:.4f})")
else:
    print(f"\nFail to reject H0: No significant evidence against 1000 hours (p={p_value:.4f})")
```

**Questions:**
- Q17: What is your conclusion at α = 0.05?
- Q18: Calculate the 95% confidence interval for the true mean
- Q19: What assumptions does the t-test require?

### Exercise 3.2: Two-Sample t-test (10 points)

**Scenario:** Compare sepal length between two iris species.

```python
# Get data for two species
setosa = iris_df[iris_df['species_name'] == 'setosa']['sepal length (cm)']
versicolor = iris_df[iris_df['species_name'] == 'versicolor']['sepal length (cm)']

print(f"Setosa - Mean: {setosa.mean():.2f}, Std: {setosa.std():.2f}")
print(f"Versicolor - Mean: {versicolor.mean():.2f}, Std: {versicolor.std():.2f}")

# TODO: Perform two-sample t-test
# H0: μ_setosa = μ_versicolor
# H1: μ_setosa ≠ μ_versicolor

t_stat, p_val = ??

print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df[iris_df['species_name'].isin(['setosa', 'versicolor'])],
            x='species_name', y='sepal length (cm)')
plt.title('Sepal Length Comparison')
plt.ylabel('Sepal Length (cm)')
plt.show()
```

**Questions:**
- Q20: Is there a significant difference between the two species?
- Q21: What is the effect size (Cohen's d)?
- Q22: Would results change with a one-tailed test?

### Exercise 3.3: Chi-Square Test (10 points)

**Scenario:** Test if a die is fair.

```python
# Observed frequencies from 60 rolls
observed = np.array([12, 8, 15, 10, 7, 8])

# TODO: Perform chi-square goodness-of-fit test
# H0: Die is fair (all faces equally likely)
# H1: Die is not fair

expected = ??  # If fair, what do we expect?

chi2_stat, p_val = ??

print(f"Observed: {observed}")
print(f"Expected: {expected}")
print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(1, 7)
width = 0.35
ax.bar(x - width/2, observed, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, expected, width, label='Expected (Fair)', alpha=0.7)
ax.set_xlabel('Die Face')
ax.set_ylabel('Frequency')
ax.set_title('Die Fairness Test')
ax.legend()
plt.show()
```

**Questions:**
- Q23: Is the die fair at α = 0.05?
- Q24: Which face(s) deviate most from expectation?
- Q25: What are the degrees of freedom?

---

## Part 4: Real-World Case Study (30 points)

### Wine Quality Analysis

**Scenario:** Analyze wine quality data to understand factors affecting quality.

```python
# Load wine quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine_df = pd.read_csv(url, sep=';')

print("Dataset shape:", wine_df.shape)
print("\nFirst few rows:")
print(wine_df.head())
print("\nDataset info:")
print(wine_df.info())
```

### Task 4.1: Exploratory Analysis (10 points)

**TODO:**
1. Calculate descriptive statistics for all numeric variables
2. Create correlation matrix and heatmap
3. Identify which features correlate most with quality
4. Create distribution plots for key features
5. Check for outliers using box plots

```python
# Your code here
```

**Questions:**
- Q26: Which feature has the strongest correlation with quality?
- Q27: Are there any highly correlated features (potential multicollinearity)?
- Q28: Do you observe any outliers?

### Task 4.2: Group Comparisons (10 points)

**TODO:**
1. Create a binary quality variable (high quality: score ≥ 7, low quality: score < 7)
2. Compare alcohol content between high and low quality wines (t-test)
3. Compare pH levels between quality groups
4. Test if volatile acidity differs between groups

```python
# Create binary quality
wine_df['quality_category'] = wine_df['quality'].apply(
    lambda x: 'High' if x >= 7 else 'Low'
)

# Your comparison code here
```

**Questions:**
- Q29: Do high-quality wines have significantly different alcohol content?
- Q30: What other features show significant differences?

### Task 4.3: Multiple Hypotheses (10 points)

**TODO:**
Test the following hypotheses:

1. **H0:** Mean alcohol content is 10%
2. **H0:** Fixed acidity and volatile acidity are independent
3. **H0:** Quality distribution is uniform across ratings

```python
# Your hypothesis testing code here
```

**Final Report Questions:**
- Q31: Summarize your key findings about wine quality
- Q32: What recommendations would you make to wine producers?
- Q33: What are the limitations of this analysis?
- Q34: What additional data would be helpful?
- Q35: How would you prevent Type I errors when testing multiple hypotheses?

---

## Bonus Challenge (10 extra points)

### Power Analysis

**Task:** Conduct a power analysis to determine sample size needed to detect a difference of 0.5% in alcohol content with 80% power at α = 0.05.

```python
from statsmodels.stats.power import ttest_power

# Your code here
```

**Question:**
- How many samples would you need?

---

## Submission Checklist

- [ ] All code cells executed successfully
- [ ] All questions answered
- [ ] Visualizations included and labeled
- [ ] Final report with interpretations
- [ ] Code is well-commented
- [ ] Results are clearly presented

## Tips for Success

1. **Understand the Context:** Always interpret results in the context of the problem
2. **Check Assumptions:** Verify test assumptions before applying
3. **Visualize First:** Plot data before testing
4. **Report Completely:** Include test statistic, p-value, and conclusion
5. **Use α = 0.05** unless otherwise specified
6. **Two-tailed tests** are default unless stated otherwise
7. **Document Everything:** Comments and markdown explanations

## Resources

- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- Course lecture notes
- Office hours: [Schedule]

Good luck! 🎓
