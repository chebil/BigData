# Chapter 3: Statistical Foundations with Python

## Overview

This chapter introduces the statistical and programming foundations essential for data science work. You'll learn Python programming focused on data analysis, master key libraries (NumPy and Pandas), understand descriptive statistics, create effective visualizations, and perform hypothesis testing. This chapter adapts the R-based content from the textbook to Python, the predominant language in modern data science.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Write Python code** for data analysis using proper syntax and style
2. **Manipulate data efficiently** using NumPy arrays and Pandas DataFrames
3. **Calculate and interpret** descriptive statistics
4. **Create effective visualizations** using Matplotlib and Seaborn
5. **Understand probability distributions** and their applications
6. **Conduct hypothesis tests** including t-tests and ANOVA
7. **Interpret p-values** and make statistical inferences
8. **Apply exploratory data analysis** techniques to real datasets

## Topics Covered

### 1. Introduction to Python
- Python basics and syntax
- Variables and data types
- Control structures (if, for, while)
- Functions and modules
- List comprehensions
- Error handling

### 2. NumPy and Pandas
- NumPy arrays and operations
- Broadcasting and vectorization
- Pandas Series and DataFrames
- Data loading and saving
- Data selection and filtering
- Grouping and aggregation

### 3. Descriptive Statistics
- Measures of central tendency (mean, median, mode)
- Measures of dispersion (variance, standard deviation)
- Percentiles and quartiles
- Correlation and covariance
- Skewness and kurtosis

### 4. Data Visualization
- Principles of effective visualization
- Matplotlib fundamentals
- Seaborn for statistical graphics
- Common plot types (scatter, line, bar, histogram, box)
- Customization and styling
- Interactive visualizations with Plotly

### 5. Hypothesis Testing
- Null and alternative hypotheses
- Type I and Type II errors
- P-values and significance levels
- T-tests (one-sample, two-sample, paired)
- Wilcoxon rank-sum test
- Chi-square tests
- Power and sample size

### 6. ANOVA
- One-way ANOVA
- Assumptions and diagnostics
- Post-hoc tests
- Interpretation of results

## Chapter Sections

```{tableofcontents}
```

## Why Python for Data Science?

### Advantages
- **General-purpose language**: Not limited to statistics
- **Huge ecosystem**: Thousands of libraries available
- **Industry adoption**: Most widely used in data science jobs
- **Integration**: Easy to connect with databases, web services, etc.
- **Community**: Large, active community and resources

### Key Libraries

**NumPy**: Numerical computing
- Fast array operations
- Mathematical functions
- Linear algebra
- Random number generation

**Pandas**: Data manipulation
- DataFrame structure (like spreadsheets)
- Data cleaning and transformation
- Time series functionality
- Easy data I/O

**Matplotlib**: Basic plotting
- MATLAB-like interface
- Fine-grained control
- Publication-quality figures

**Seaborn**: Statistical visualization
- Built on Matplotlib
- Beautiful default styles
- Statistical plot types
- Easy to create complex visualizations

**SciPy**: Scientific computing
- Statistical functions
- Optimization
- Integration
- Signal processing

**Statsmodels**: Statistical modeling
- Regression models
- Time series analysis
- Statistical tests
- Summary statistics

## Python vs. R

The textbook uses R, but this course uses Python. Here's why:

| Aspect | Python | R |
|--------|--------|---|
| **Primary Use** | General programming | Statistical analysis |
| **Learning Curve** | Gentler for programmers | Steeper for non-statisticians |
| **Industry Use** | Very high | High in academia, growing in industry |
| **ML Libraries** | Extensive (scikit-learn, TensorFlow, PyTorch) | Good (caret, mlr) |
| **Production** | Easier to deploy | More challenging |
| **Visualization** | Good (Matplotlib, Seaborn, Plotly) | Excellent (ggplot2) |

**The Good News**: Concepts transfer easily between languages!

## Exploratory Data Analysis (EDA)

### The EDA Process

1. **Understand the Data**
   - What do the columns represent?
   - What are the data types?
   - How many observations?
   - What's the time range?

2. **Check Data Quality**
   - Missing values?
   - Outliers or anomalies?
   - Duplicates?
   - Inconsistent formatting?

3. **Explore Distributions**
   - Histograms for continuous variables
   - Bar charts for categorical variables
   - Box plots for identifying outliers

4. **Examine Relationships**
   - Scatter plots for two continuous variables
   - Correlation matrices
   - Grouped statistics

5. **Generate Hypotheses**
   - What patterns do you see?
   - What might explain them?
   - What should you investigate further?

### Anscombe's Quartet

A famous example showing why visualization is crucial:
- Four datasets with nearly identical statistics
- Very different underlying patterns
- Only visible through plotting

**Lesson**: Always visualize your data!

## Statistical Testing

### The Hypothesis Testing Framework

1. **State Hypotheses**
   - H₀ (null hypothesis): No effect or no difference
   - H₁ (alternative hypothesis): There is an effect or difference

2. **Choose Significance Level**
   - α = 0.05 is common (5% chance of Type I error)
   - Can adjust based on context

3. **Calculate Test Statistic**
   - Depends on the test (t, F, χ², etc.)
   - Measures strength of evidence against H₀

4. **Determine P-value**
   - Probability of observing data if H₀ is true
   - Lower p-value = stronger evidence against H₀

5. **Make Decision**
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀

6. **Interpret Results**
   - What does this mean in context?
   - What are the practical implications?

### Common Tests

**One-Sample T-Test**: Compare sample mean to a known value
- Example: Is average customer satisfaction > 7?

**Two-Sample T-Test**: Compare means of two groups
- Example: Do males and females differ in height?

**Paired T-Test**: Compare before/after measurements
- Example: Did training improve test scores?

**ANOVA**: Compare means of 3+ groups
- Example: Do different fertilizers affect crop yield?

**Chi-Square Test**: Test independence of categorical variables
- Example: Is disease incidence related to smoking?

### Type I and Type II Errors

|  | H₀ True | H₀ False |
|--|---------|----------|
| **Reject H₀** | Type I Error (α) | Correct Decision (Power = 1-β) |
| **Fail to Reject H₀** | Correct Decision | Type II Error (β) |

**Type I Error (α)**: False positive - Rejecting H₀ when it's true
**Type II Error (β)**: False negative - Failing to reject H₀ when it's false
**Power (1-β)**: Probability of correctly rejecting false H₀

## Hands-On Practice

### Associated Labs
- **[Lab 2: Python & Pandas](../labs/lab-02-python-pandas/README.md)** - Master data manipulation
- **[Lab 3: Statistics & Visualization](../labs/lab-03-statistics-visualization/README.md)** - Apply statistical methods

### Jupyter Notebooks
1. [Python Basics](notebooks/01-python-basics.ipynb) - Python fundamentals
2. [Pandas Fundamentals](notebooks/02-pandas-fundamentals.ipynb) - DataFrame operations
3. [Exploratory Analysis](notebooks/03-exploratory-analysis.ipynb) - EDA techniques
4. [Visualization Gallery](notebooks/04-visualization-gallery.ipynb) - Creating effective plots
5. [Statistical Tests](notebooks/05-statistical-tests.ipynb) - Hypothesis testing

## Best Practices

### Code Quality
- ✅ Use descriptive variable names
- ✅ Comment your code
- ✅ Follow PEP 8 style guide
- ✅ Write functions for repeated operations
- ✅ Handle errors gracefully

### Data Analysis
- ✅ Always explore data before analysis
- ✅ Check assumptions of statistical tests
- ✅ Use appropriate visualizations
- ✅ Document your analytical decisions
- ✅ Validate results with different approaches

### Visualization
- ✅ Choose appropriate chart types
- ✅ Label axes and add titles
- ✅ Use color purposefully
- ✅ Keep it simple
- ✅ Consider your audience

## Common Pitfalls

- ❌ P-hacking: Testing many hypotheses and reporting only significant ones
- ❌ Ignoring assumptions of statistical tests
- ❌ Confusing correlation with causation
- ❌ Not checking for outliers
- ❌ Using inappropriate visualizations
- ❌ Overcomplicating analyses

## Additional Resources

### Required Reading
- Textbook Chapter 3: "Review of Basic Data Analytic Methods Using R"
- EMC Education Services, pp. 63-115
- **Note**: Adapt R code examples to Python using course notebooks

### Python Learning
- [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney (Pandas creator)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake VanderPlas
- [Real Python Tutorials](https://realpython.com/)

### Statistics
- [Think Stats](https://greenteapress.com/thinkstats2/) - Python-based statistics
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer) - Excellent explanations
- [Statistics for Data Science Course](https://www.coursera.org/learn/statistics-for-data-science-python)

### Visualization
- [Python Graph Gallery](https://www.python-graph-gallery.com/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

## Summary

This chapter provides the statistical and programming foundation for the rest of the course. Python, with libraries like NumPy, Pandas, and Matplotlib, provides a powerful toolkit for data analysis. Understanding descriptive statistics, visualization, and hypothesis testing is essential for drawing valid conclusions from data.

Key takeaways:
- Python is ideal for data science due to its ecosystem and versatility
- Exploratory data analysis should always precede formal modeling
- Visualization is essential for understanding data
- Statistical tests help us make inferences but require careful interpretation
- Good code practices make analysis reproducible and maintainable

## Next Steps

1. Work through all five Jupyter notebooks
2. Complete [Lab 2: Python & Pandas](../labs/lab-02-python-pandas/README.md)
3. Complete [Lab 3: Statistics & Visualization](../labs/lab-03-statistics-visualization/README.md)
4. Prepare for Quiz 1 (Chapters 1-3) in Week 4
5. Move on to [Chapter 4: Clustering Analysis](../04-clustering/index.md)

---

**Checkpoint**: By now you should be comfortable with Python basics, Pandas DataFrames, creating visualizations, and conducting statistical tests. These skills will be used throughout the rest of the course!
