# Association Rules Mining - Overview

## Learning Objectives

- Understand association rule mining concepts
- Learn key metrics: support, confidence, lift
- Identify applications of association rules
- Understand the market basket analysis problem
- Prepare data for association rule mining

## Introduction

Association rule mining discovers **interesting relationships** between items in large datasets. It's most famous for **market basket analysis** - finding products frequently purchased together.

## Core Concepts

### Transaction Database

```python
import pandas as pd

# Example: Grocery store transactions
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread']
]

df = pd.DataFrame({'items': transactions})
print("Transaction Database:")
for idx, items in enumerate(transactions, 1):
    print(f"T{idx}: {items}")

print(f"\nTotal transactions: {len(transactions)}")

# All unique items
all_items = set(item for transaction in transactions for item in transaction)
print(f"Unique items: {sorted(all_items)}")
```

### Itemsets

**Itemset**: Collection of one or more items

```python
print("\nExamples of Itemsets:")
print("1-itemset (single item): {Milk}")
print("2-itemset (pair): {Milk, Bread}")
print("3-itemset (triple): {Milk, Bread, Butter}")
```

### Association Rules

**Form**: `{Antecedent} → {Consequent}`

**Example**: `{Milk, Bread} → {Butter}`

**Interpretation**: "Customers who buy Milk and Bread are likely to also buy Butter"

```python
print("\nAssociation Rule Structure:")
print("If customer buys {Milk, Bread} [antecedent]")
print("  Then customer likely buys {Butter} [consequent]")
print("\nNotation: {Milk, Bread} → {Butter}")
```

## Key Metrics

### 1. Support

**Definition**: Frequency of itemset in database

\[
\text{support}(X) = \frac{\text{# transactions containing } X}{\text{total # transactions}}
\]

```python
import numpy as np

def calculate_support(itemset, transactions):
    """
    Calculate support for an itemset
    """
    itemset = set(itemset)
    count = sum(1 for transaction in transactions if itemset.issubset(set(transaction)))
    return count / len(transactions)

# Examples
print("Support Calculations:")
print(f"support({{Milk}}): {calculate_support(['Milk'], transactions):.2%}")
print(f"support({{Bread}}): {calculate_support(['Bread'], transactions):.2%}")
print(f"support({{Milk, Bread}}): {calculate_support(['Milk', 'Bread'], transactions):.2%}")
print(f"support({{Milk, Bread, Butter}}): {calculate_support(['Milk', 'Bread', 'Butter'], transactions):.2%}")

print("\nInterpretation:")
print("- support({Milk}) = 62.5% → Milk appears in 62.5% of transactions")
print("- support({Milk, Bread}) = 50% → Both appear together in 50% of transactions")
```

### 2. Confidence

**Definition**: Conditional probability of consequent given antecedent

\[
\text{confidence}(X \rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}
\]

```python
def calculate_confidence(antecedent, consequent, transactions):
    """
    Calculate confidence for a rule: antecedent → consequent
    """
    antecedent = set(antecedent)
    consequent = set(consequent)
    combined = antecedent.union(consequent)
    
    support_antecedent = calculate_support(antecedent, transactions)
    support_combined = calculate_support(combined, transactions)
    
    if support_antecedent == 0:
        return 0
    
    return support_combined / support_antecedent

# Examples
print("\nConfidence Calculations:")
conf1 = calculate_confidence(['Milk', 'Bread'], ['Butter'], transactions)
print(f"confidence({{Milk, Bread}} → {{Butter}}): {conf1:.2%}")

conf2 = calculate_confidence(['Bread'], ['Eggs'], transactions)
print(f"confidence({{Bread}} → {{Eggs}}): {conf2:.2%}")

print("\nInterpretation:")
print(f"- {conf1:.0%} of customers who buy Milk and Bread also buy Butter")
print(f"- {conf2:.0%} of customers who buy Bread also buy Eggs")
```

### 3. Lift

**Definition**: How much more likely is consequent when antecedent occurs?

\[
\text{lift}(X \rightarrow Y) = \frac{\text{confidence}(X \rightarrow Y)}{\text{support}(Y)}
\]

Or equivalently:
\[
\text{lift}(X \rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X) \times \text{support}(Y)}
\]

```python
def calculate_lift(antecedent, consequent, transactions):
    """
    Calculate lift for a rule: antecedent → consequent
    """
    antecedent = set(antecedent)
    consequent = set(consequent)
    combined = antecedent.union(consequent)
    
    support_antecedent = calculate_support(antecedent, transactions)
    support_consequent = calculate_support(consequent, transactions)
    support_combined = calculate_support(combined, transactions)
    
    if support_antecedent == 0 or support_consequent == 0:
        return 0
    
    return support_combined / (support_antecedent * support_consequent)

# Examples
print("\nLift Calculations:")
lift1 = calculate_lift(['Milk', 'Bread'], ['Butter'], transactions)
print(f"lift({{Milk, Bread}} → {{Butter}}): {lift1:.2f}")

lift2 = calculate_lift(['Bread'], ['Eggs'], transactions)
print(f"lift({{Bread}} → {{Eggs}}): {lift2:.2f}")

print("\nInterpretation:")
print("- lift > 1: Positive correlation (items occur together more than expected)")
print("- lift = 1: Independent (no correlation)")
print("- lift < 1: Negative correlation (items occur together less than expected)")

if lift1 > 1:
    print(f"\n✓ Milk+Bread and Butter are positively correlated (lift={lift1:.2f})")
if lift2 > 1:
    print(f"✓ Bread and Eggs are positively correlated (lift={lift2:.2f})")
```

### 4. Additional Metrics

#### Leverage

**Definition**: Difference between observed and expected support

\[
\text{leverage}(X \rightarrow Y) = \text{support}(X \cup Y) - \text{support}(X) \times \text{support}(Y)
\]

```python
def calculate_leverage(antecedent, consequent, transactions):
    antecedent = set(antecedent)
    consequent = set(consequent)
    combined = antecedent.union(consequent)
    
    support_antecedent = calculate_support(antecedent, transactions)
    support_consequent = calculate_support(consequent, transactions)
    support_combined = calculate_support(combined, transactions)
    
    return support_combined - (support_antecedent * support_consequent)

leverage = calculate_leverage(['Milk', 'Bread'], ['Butter'], transactions)
print(f"\nleverage({{Milk, Bread}} → {{Butter}}): {leverage:.3f}")
print("Positive leverage → positive association")
```

#### Conviction

**Definition**: Measures dependency

\[
\text{conviction}(X \rightarrow Y) = \frac{1 - \text{support}(Y)}{1 - \text{confidence}(X \rightarrow Y)}
\]

```python
def calculate_conviction(antecedent, consequent, transactions):
    support_consequent = calculate_support(consequent, transactions)
    confidence = calculate_confidence(antecedent, consequent, transactions)
    
    if confidence == 1:
        return float('inf')
    
    return (1 - support_consequent) / (1 - confidence)

conviction = calculate_conviction(['Milk', 'Bread'], ['Butter'], transactions)
print(f"\nconviction({{Milk, Bread}} → {{Butter}}): {conviction:.2f}")
print("Higher conviction → stronger rule")
```

## Metrics Summary Table

```python
import pandas as pd

# Calculate all metrics for example rule
rule = ({'Milk', 'Bread'}, {'Butter'})
antecedent, consequent = rule

metrics = {
    'Metric': ['Support', 'Confidence', 'Lift', 'Leverage', 'Conviction'],
    'Value': [
        calculate_support(antecedent.union(consequent), transactions),
        calculate_confidence(antecedent, consequent, transactions),
        calculate_lift(antecedent, consequent, transactions),
        calculate_leverage(antecedent, consequent, transactions),
        calculate_conviction(antecedent, consequent, transactions)
    ],
    'Interpretation': [
        'Frequency of itemset',
        'P(Butter | Milk+Bread)',
        'Correlation strength',
        'Observed vs expected',
        'Rule dependency'
    ]
}

metrics_df = pd.DataFrame(metrics)
print("\n" + "="*60)
print(f"METRICS FOR RULE: {{Milk, Bread}} → {{Butter}}")
print("="*60)
print(metrics_df.to_string(index=False))
```

## The Association Rule Mining Problem

**Input**: 
- Transaction database
- Minimum support threshold (min_sup)
- Minimum confidence threshold (min_conf)

**Output**: 
- All association rules with:
  - support ≥ min_sup
  - confidence ≥ min_conf

**Two-Step Process**:

1. **Frequent Itemset Mining**: Find all itemsets with support ≥ min_sup
2. **Rule Generation**: Generate rules from frequent itemsets with confidence ≥ min_conf

```python
print("""
ASSOCIATION RULE MINING PROCESS:

Step 1: FREQUENT ITEMSET MINING
  Input: Transactions, min_support
  Output: All itemsets with support ≥ min_support
  Algorithms: Apriori, FP-Growth

Step 2: RULE GENERATION
  Input: Frequent itemsets, min_confidence
  Output: Rules with confidence ≥ min_confidence
  
Example:
  min_support = 30%
  min_confidence = 60%
  
  → Find itemsets appearing in ≥30% of transactions
  → Generate rules with ≥60% confidence
""")
```

## Applications

### 1. Market Basket Analysis

```python
print("""
MARKET BASKET ANALYSIS:

Use Cases:
  • Product placement in stores
  • Cross-selling recommendations
  • Promotional bundling
  • Inventory management
  
Example Rules:
  {Diapers} → {Beer}
  {Milk, Bread} → {Eggs}
  {Chips, Salsa} → {Soda}
  
Business Actions:
  • Place related items nearby
  • Bundle products in promotions
  • Recommend complementary items
""")
```

### 2. Web Usage Mining

```python
print("""
WEB USAGE MINING:

Use Cases:
  • Website navigation patterns
  • Page recommendations
  • Content organization
  
Example Rules:
  {Homepage, Products} → {Checkout}
  {Article A} → {Article B}
  {Video 1, Video 2} → {Video 3}
""")
```

### 3. Healthcare

```python
print("""
HEALTHCARE APPLICATIONS:

Use Cases:
  • Disease co-occurrence
  • Treatment patterns
  • Drug interactions
  
Example Rules:
  {Diabetes, Hypertension} → {Kidney Disease}
  {Drug A, Drug B} → {Side Effect}
  {Symptom X, Symptom Y} → {Diagnosis Z}
""")
```

## Data Preparation

### Transaction Format

```python
import pandas as pd

# Method 1: List of lists
transactions_list = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs', 'Butter']
]

print("Format 1: List of Lists")
for t in transactions_list:
    print(t)

# Method 2: DataFrame (one-hot encoded)
data = {
    'Transaction': [1, 2, 3],
    'Milk': [1, 0, 1],
    'Bread': [1, 1, 1],
    'Butter': [1, 0, 1],
    'Eggs': [0, 1, 1]
}

df_onehot = pd.DataFrame(data)
print("\nFormat 2: One-Hot Encoded DataFrame")
print(df_onehot)

# Method 3: DataFrame (long format)
data_long = {
    'Transaction': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'Item': ['Milk', 'Bread', 'Butter', 'Bread', 'Eggs', 
             'Milk', 'Bread', 'Eggs', 'Butter']
}

df_long = pd.DataFrame(data_long)
print("\nFormat 3: Long Format DataFrame")
print(df_long)
```

### Converting Between Formats

```python
import pandas as pd

# Long format → List of lists
def long_to_transactions(df, transaction_col='Transaction', item_col='Item'):
    return df.groupby(transaction_col)[item_col].apply(list).tolist()

transactions = long_to_transactions(df_long)
print("\nConverted to transaction list:")
for i, t in enumerate(transactions, 1):
    print(f"T{i}: {t}")

# One-hot → List of lists
def onehot_to_transactions(df, exclude_cols=['Transaction']):
    transactions = []
    for idx, row in df.iterrows():
        items = [col for col in df.columns 
                if col not in exclude_cols and row[col] == 1]
        transactions.append(items)
    return transactions

transactions_from_onehot = onehot_to_transactions(df_onehot)
print("\nConverted from one-hot:")
for i, t in enumerate(transactions_from_onehot, 1):
    print(f"T{i}: {t}")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Association rules** discover relationships in transaction data
2. **Support**: How often items appear together
3. **Confidence**: Conditional probability of consequent
4. **Lift**: Correlation strength (>1 = positive, <1 = negative)
5. **Two-step process**: Find frequent itemsets → Generate rules
6. **Applications**: Retail, web, healthcare, and more
7. **Thresholds** (min_sup, min_conf) control output size
8. **Higher thresholds** → fewer but stronger rules
9. **Lower thresholds** → more but weaker rules
10. **Interpretability** is a key advantage
:::

## Further Reading

- Agrawal, R. & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
- Han, J. et al. (2000). "Mining Frequent Patterns without Candidate Generation"
- Tan, P. et al. (2005). "Selecting the Right Objective Measure for Association Analysis"
