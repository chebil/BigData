# Chapter 1 Exercises

## Conceptual Questions

### Exercise 1.1: Understanding Big Data

**Question**: Explain the three Vs of Big Data (Volume, Velocity, Variety) with examples from a domain you're familiar with (e.g., healthcare, finance, retail, education).

**Answer Template**:
- Volume: _[Your example]_
- Velocity: _[Your example]_
- Variety: _[Your example]_

---

### Exercise 1.2: Data Type Classification

**Question**: Classify each of the following data sources as structured, semi-structured, quasi-structured, or unstructured. Justify your classification.

a) Customer purchase transactions in a retail database  
b) Twitter posts about a product launch  
c) Web server access logs  
d) Product reviews with star ratings on Amazon  
e) Sensor readings from IoT devices formatted as JSON  
f) Medical imaging files (X-rays, MRI scans)  
g) Email messages  
h) Stock market tick data  

---

### Exercise 1.3: Business Drivers

**Question**: For each business driver below, provide a specific example of how Big Data analytics could address it:

a) Optimize business operations  
b) Identify business risk  
c) Predict new business opportunities  
d) Comply with regulatory requirements  

---

### Exercise 1.4: Data Analytics Lifecycle

**Question**: 

a) Explain why the Data Analytics Lifecycle is iterative rather than a linear waterfall process.  
b) Give an example of a scenario where you would need to return from Phase 4 (Model Building) to Phase 2 (Data Preparation).  
c) Which phase typically consumes the most time in a data science project, and why?

---

## Practical Exercises

### Exercise 1.5: Data Source Inventory

**Scenario**: You're working for an e-commerce company that wants to predict customer churn. Create a data source inventory identifying:

a) At least 5 internal data sources you would want to analyze  
b) At least 3 external data sources that could enhance your analysis  
c) For each source, classify the data type (structured/semi-structured/unstructured)  
d) Identify any data sources that are available but not accessible, and explain why

**Deliverable**: Create a table with columns: Data Source, Data Type, Availability Status, Business Value

---

### Exercise 1.6: Hypothesis Formulation

**Scenario**: A mobile app company has noticed declining user engagement. During the Discovery phase, formulate:

a) 3 testable hypotheses about why engagement might be declining  
b) For each hypothesis, identify what data you would need to test it  
c) Prioritize your hypotheses based on:
   - Potential business impact
   - Data availability
   - Feasibility of testing

---

### Exercise 1.7: Python Data Exploration

**Task**: Given a sample dataset of customer transactions:

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = {
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'purchase_amount': np.random.uniform(10, 500, 100),
    'num_purchases': np.random.randint(1, 20, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}

df = pd.DataFrame(data)
```

Complete the following tasks:

a) Load the data and display basic information (shape, data types, first few rows)  
b) Calculate summary statistics for numerical columns  
c) Check for missing values  
d) Create a histogram showing the distribution of purchase amounts  
e) Calculate the average purchase amount by region  
f) Identify any potential outliers in the purchase_amount column

---

### Exercise 1.8: Clickstream Analysis

**Task**: Given the following clickstream data:

```python
clickstream = [
    {"user_id": 1, "timestamp": "2024-01-15 10:23:15", "url": "https://shop.com/home"},
    {"user_id": 1, "timestamp": "2024-01-15 10:24:32", "url": "https://shop.com/products/laptop"},
    {"user_id": 1, "timestamp": "2024-01-15 10:28:45", "url": "https://shop.com/cart"},
    {"user_id": 2, "timestamp": "2024-01-15 10:25:10", "url": "https://shop.com/home"},
    {"user_id": 2, "timestamp": "2024-01-15 10:26:20", "url": "https://shop.com/products/phone"},
]
```

Write Python code to:

a) Parse the URLs to extract the page category (home, products, cart)  
b) Calculate the time spent on each page for each user  
c) Identify the most common navigation path  
d) Determine which users completed a purchase (reached cart)

---

### Exercise 1.9: Data Quality Assessment

**Task**: You receive the following dataset with quality issues:

```python
import pandas as pd

data = {
    'customer_id': [1, 2, 3, 4, 5, 5, 6],
    'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve', 'Frank'],
    'age': [25, 30, 35, -5, 28, 28, 150],
    'email': ['alice@test.com', 'bob@test', 'carol@test.com', 'david@test.com', 'eve@test.com', 'eve@test.com', None],
    'purchase_date': ['2024-01-15', '2024/01/16', '2024-01-17', '15-01-2024', '2024-01-18', '2024-01-18', '2024-01-19']
}

df = pd.DataFrame(data)
```

Identify and fix:

a) Duplicate records  
b) Missing values  
c) Invalid age values  
d) Inconsistent date formats  
e) Invalid email addresses  

Provide the cleaned dataset and document all transformations applied.

---

### Exercise 1.10: Mini Project - Customer Segmentation Discovery

**Scenario**: You work for a subscription-based streaming service. The marketing team wants to understand customer segments to personalize marketing campaigns.

**Tasks**:

1. **Discovery Phase**:
   - Define 3-5 business questions to answer
   - List data sources you would need
   - Formulate initial hypotheses about customer segments

2. **Data Preparation Phase**:
   - Create a sample dataset with relevant customer attributes:
     - Demographics (age, location)
     - Behavior (watch time, genres watched, device usage)
     - Subscription info (plan type, tenure, payment history)
   - Generate at least 500 sample records using Python

3. **Exploratory Analysis**:
   - Calculate summary statistics
   - Create visualizations showing:
     - Distribution of key metrics
     - Relationships between variables
     - Potential customer groupings

4. **Documentation**:
   - Write a brief report (2-3 paragraphs) summarizing:
     - Key findings from exploration
     - Recommended next steps
     - Potential challenges

---

## Discussion Questions

### Exercise 1.11: BI vs. Data Science

**Question**: Compare Business Intelligence (BI) and Data Science approaches:

a) What types of questions does each approach answer?  
b) What are the main differences in tools and techniques used?  
c) When would you choose BI over Data Science, or vice versa?  
d) Can they complement each other? Provide an example.

---

### Exercise 1.12: Ethical Considerations

**Question**: Consider the Big Data ecosystem described in the chapter (data devices, data collectors, data aggregators, data buyers).

a) What ethical concerns arise from this ecosystem?  
b) Who should own the data generated by individuals?  
c) How can organizations balance business value with privacy protection?  
d) What role do regulations (GDPR, CCPA) play in this ecosystem?

---

### Exercise 1.13: Sandbox vs. Production

**Question**: 

a) Why is it important to have a separate analytic sandbox instead of running analyses directly on production databases?  
b) What are the potential risks of not having a sandbox environment?  
c) How large should an analytic sandbox be relative to the original data size, and why?  
d) Who should have access to the sandbox, and what governance policies should be in place?

---

## Coding Challenges

### Exercise 1.14: Build a Simple Data Pipeline

**Task**: Create a Python script that:

1. Simulates extracting data from multiple sources (CSV, JSON)  
2. Performs data quality checks  
3. Merges the datasets  
4. Generates a data quality report  
5. Saves the cleaned data

**Requirements**:
- Use pandas for data manipulation
- Create at least 2 sample datasets with intentional quality issues
- Document all transformations
- Generate visualizations showing before/after data quality metrics

---

### Exercise 1.15: Web Scraping and Analysis

**Task**: Choose a public website (e.g., news site, product reviews) and:

1. Scrape data using Python (BeautifulSoup or Scrapy)  
2. Structure the unstructured web data  
3. Perform basic text analysis  
4. Create visualizations of your findings

**Note**: Respect robots.txt and terms of service. Use ethical scraping practices.

---

## Solutions

Detailed solutions to selected exercises are available in the `solutions/` directory.

## Grading Rubric

For instructors evaluating these exercises:

| Criteria | Points |
|----------|--------|
| Correctness of analysis | 40% |
| Code quality and documentation | 25% |
| Clarity of explanations | 20% |
| Visualizations and presentation | 15% |

## Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Real Python - Data Science Tutorials](https://realpython.com/tutorials/data-science/)
