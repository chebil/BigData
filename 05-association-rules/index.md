# Chapter 5: Association Rules Mining

## Overview

Association rules mining discovers interesting relationships between variables in large databases. Most famously used for market basket analysis ("customers who bought X also bought Y"), this technique has applications across many domains. This chapter covers the Apriori algorithm, evaluation metrics, and practical applications of association rules.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand association rules** and their components (antecedent, consequent)
2. **Explain key metrics**: support, confidence, and lift
3. **Implement the Apriori algorithm** for finding frequent itemsets
4. **Generate association rules** from frequent itemsets
5. **Evaluate rules** using multiple metrics
6. **Apply association rules** to market basket analysis
7. **Interpret results** in business context
8. **Visualize association rules** effectively

## Topics Covered

### 1. Market Basket Analysis
- Transaction data structure
- Business applications
- Retail use cases
- Beyond retail: other applications

### 2. Apriori Algorithm
- Frequent itemset generation
- Apriori principle
- Candidate generation and pruning
- Algorithm steps
- Computational complexity

### 3. Evaluation Metrics
- Support
- Confidence
- Lift
- Conviction
- Leverage
- Choosing appropriate metrics

### 4. Association Rules Applications
- Product recommendations
- Cross-selling strategies
- Store layout optimization
- Web usage mining
- Healthcare diagnosis

## Chapter Sections

```{tableofcontents}
```

## Association Rules Fundamentals

### Rule Structure

An association rule has the form:

\[
\text{Antecedent} \Rightarrow \text{Consequent}
\]

Example: \(\{\text{Bread, Butter}\} \Rightarrow \{\text{Milk}\}\)

**Antecedent** (LHS): Items in the "if" part
**Consequent** (RHS): Items in the "then" part

### Key Metrics

#### Support

Proportion of transactions containing the itemset:

\[
\text{support}(X) = \frac{\text{count}(X)}{\text{total transactions}}
\]

Example: If 100 of 1000 transactions contain {Bread, Butter}:
- support({Bread, Butter}) = 100/1000 = 0.10 = 10%

**Interpretation**: How frequently does the itemset appear?

#### Confidence

Proportion of transactions with X that also contain Y:

\[
\text{confidence}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}
\]

Example: If 80 of 100 {Bread, Butter} transactions also have Milk:
- confidence({Bread, Butter} ⇒ {Milk}) = 80/100 = 0.80 = 80%

**Interpretation**: How often is the rule correct?

#### Lift

Ratio of observed to expected support:

\[
\text{lift}(X \Rightarrow Y) = \frac{\text{confidence}(X \Rightarrow Y)}{\text{support}(Y)}
\]

or equivalently:

\[
\text{lift}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X) \times \text{support}(Y)}
\]

**Interpretation**:
- Lift > 1: Positive correlation (buying X increases probability of buying Y)
- Lift = 1: No correlation (independent)
- Lift < 1: Negative correlation (buying X decreases probability of buying Y)

## Apriori Algorithm

### The Apriori Principle

**Key Insight**: If an itemset is frequent, all its subsets must also be frequent.

**Contrapositive**: If an itemset is infrequent, all its supersets must also be infrequent.

This allows pruning of candidate itemsets, making the algorithm efficient.

### Algorithm Steps

1. **Set minimum support threshold** (e.g., 1%)
2. **Find frequent 1-itemsets** (single items above threshold)
3. **Generate candidate 2-itemsets** from frequent 1-itemsets
4. **Prune candidates** using Apriori principle
5. **Count support** for remaining candidates
6. **Keep frequent 2-itemsets**
7. **Repeat** for 3-itemsets, 4-itemsets, etc. until no more frequent itemsets

### Example Walkthrough

**Transactions**:
```
T1: {Bread, Milk}
T2: {Bread, Diaper, Beer, Eggs}
T3: {Milk, Diaper, Beer, Cola}
T4: {Bread, Milk, Diaper, Beer}
T5: {Bread, Milk, Diaper, Cola}
```

**Minimum Support**: 60% (3/5 transactions)

**Step 1**: Frequent 1-itemsets
- {Bread}: 4/5 = 80% ✓
- {Milk}: 4/5 = 80% ✓
- {Diaper}: 4/5 = 80% ✓
- {Beer}: 3/5 = 60% ✓
- {Eggs}: 1/5 = 20% ✗
- {Cola}: 2/5 = 40% ✗

**Step 2**: Candidate 2-itemsets (combinations of frequent 1-itemsets)

**Step 3**: Count and prune...

### Computational Complexity

- Worst case: Exponential in number of items
- Apriori principle dramatically reduces search space
- Most efficient for sparse transaction data

## Rule Generation

Once frequent itemsets are found:

1. **For each frequent itemset**: Generate all possible rules
2. **Calculate confidence**: For each rule
3. **Filter by minimum confidence**: Keep only high-confidence rules
4. **Calculate additional metrics**: Lift, conviction, etc.
5. **Rank and filter**: Select most interesting rules

## Advanced Metrics

### Conviction

Measures departure from independence:

\[
\text{conviction}(X \Rightarrow Y) = \frac{1 - \text{support}(Y)}{1 - \text{confidence}(X \Rightarrow Y)}
\]

- Higher values indicate stronger association
- Conviction of 1.5 means X appears 1.5 times more often without Y if the rule didn't hold

### Leverage

Difference between observed and expected co-occurrence:

\[
\text{leverage}(X \Rightarrow Y) = \text{support}(X \cup Y) - \text{support}(X) \times \text{support}(Y)
\]

- Range: [-1, 1]
- Positive: Positive association
- Zero: Independence

## Applications

### Retail Market Basket Analysis

**Use Cases**:
- Product placement in stores
- Cross-selling recommendations
- Promotional bundling
- Inventory management

**Example**:
- Rule: {Diapers} ⇒ {Beer} (famous "beer and diapers" story)
- Action: Place beer near diapers, create combo deals

### E-commerce Recommendations

- "Customers who bought X also bought Y"
- Personalized product recommendations
- Shopping cart suggestions

### Web Usage Mining

- Page navigation patterns
- Click-through sequences
- User behavior analysis

### Healthcare

- Symptom associations
- Drug interactions
- Treatment pathways
- Disease comorbidities

## Visualization

### Scatter Plots

- X-axis: Support
- Y-axis: Confidence
- Size: Lift
- Shows trade-offs between metrics

### Network Graphs

- Nodes: Items
- Edges: Association rules
- Edge thickness: Confidence or lift
- Shows item relationships

### Parallel Coordinates

- Multiple metrics displayed simultaneously
- Each line represents a rule
- Good for comparing many rules

## Hands-On Practice

### Associated Lab
- **[Lab 5: Association Rules](../labs/lab-05-association-rules/README.md)** - Implement Apriori and analyze retail data

### Jupyter Notebooks
1. [Association Mining](notebooks/01-association-mining.ipynb) - Apriori algorithm implementation
2. [Retail Recommendations](notebooks/02-retail-recommendations.ipynb) - Real grocery store analysis

## Case Study: Grocery Store Analysis

### Dataset

- **Groceries dataset**: 9,835 transactions
- **Items**: 169 unique products
- **Time period**: 1 month

### Analysis Steps

1. **Data Preparation**: Convert to transaction format
2. **Frequent Itemsets**: Apriori with min_support=0.01
3. **Rule Generation**: min_confidence=0.5
4. **Evaluation**: Sort by lift
5. **Interpretation**: Domain knowledge application

### Interesting Rules Found

1. {Other vegetables, Root vegetables} ⇒ {Whole milk}
   - Support: 2.3%, Confidence: 48.5%, Lift: 1.9
   
2. {Butter, Whole milk} ⇒ {Other vegetables}
   - Support: 1.5%, Confidence: 53.1%, Lift: 2.7

3. {Tropical fruit, Yogurt} ⇒ {Whole milk}
   - Support: 1.5%, Confidence: 51.7%, Lift: 2.0

### Business Recommendations

- Place whole milk near vegetables section
- Create combo deals with identified associations
- Optimize shelf placement based on lift values

## Practical Considerations

### Setting Thresholds

**Minimum Support**:
- Too high: Miss rare but important patterns
- Too low: Too many rules, high computation
- Typical: 0.1% - 5% depending on dataset size

**Minimum Confidence**:
- Too high: Miss valid associations
- Too low: Many spurious rules
- Typical: 50% - 80%

**Lift**:
- Consider only lift > 1 for positive associations
- Higher lift = stronger association
- But also check support (high lift with low support may not be actionable)

### Limitations

- ❌ Assumes all items equally important
- ❌ Doesn't handle quantities
- ❌ Can generate many redundant rules
- ❌ Requires discrete/categorical data
- ❌ Correlation ≠ causation

## Common Pitfalls

- ❌ Setting thresholds too low (too many rules)
- ❌ Ignoring lift (confidence alone is misleading)
- ❌ Not considering business context
- ❌ Assuming causation from association
- ❌ Not validating on new data

## Additional Resources

### Required Reading
- Textbook Chapter 5: "Advanced Analytical Theory and Methods: Association Rules"
- EMC Education Services, pp. 137-160

### Recommended Reading
- "Introduction to Data Mining" by Tan, Steinbach, Kumar (Chapter 6)

### Python Libraries
- **mlxtend**: Apriori and rule generation
- **PyFIM**: Fast frequent itemset mining
- **Orange**: GUI-based association rules

### Videos
- [StatQuest: Association Rules](https://www.youtube.com/watch?v=WGlMlS_Yydk)
- [Apriori Algorithm Explained](https://www.youtube.com/watch?v=guVvtZ7ZClw)

## Summary

Association rules mining discovers interesting relationships in transaction data. Key concepts:

- **Apriori algorithm**: Efficiently finds frequent itemsets
- **Support**: How often items appear together
- **Confidence**: How often rule is correct
- **Lift**: Strength of association (most important!)

Applications extend beyond retail to web mining, healthcare, and any domain with transactional data.

## Next Steps

1. Work through Jupyter notebooks
2. Complete [Lab 5: Association Rules](../labs/lab-05-association-rules/README.md)
3. Apply to your own transactional data
4. Move on to [Chapter 6: Regression Analysis](../06-regression/index.md)

---

**Remember**: High confidence doesn't mean strong association - always check lift!
