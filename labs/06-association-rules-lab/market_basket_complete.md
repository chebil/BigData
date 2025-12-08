# Lab 6: Association Rules Mining - Complete Solution
## Market Basket Analysis for Retail

### Learning Objectives
1. Understand association rule mining concepts
2. Implement Apriori algorithm
3. Use FP-Growth for large datasets
4. Generate actionable business insights
5. Create product recommendation systems

---

## Part 1: Dataset Preparation

### 1.1 Load Retail Transaction Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Sample transaction data
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread', 'Eggs']
]

print("Sample Transactions:")
for i, trans in enumerate(transactions, 1):
    print(f"Transaction {i}: {trans}")

print(f"\nTotal transactions: {len(transactions)}")
```

### 1.2 Real-World Dataset - Online Retail

```python
# Load Online Retail dataset
# Source: UCI ML Repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'

try:
    df = pd.read_excel(url)
    print("\nOnline Retail Dataset Loaded!")
except:
    # Alternative: Create synthetic dataset
    print("\nCreating synthetic retail dataset...")
    
    products = ['Milk', 'Bread', 'Butter', 'Eggs', 'Cheese', 'Yogurt', 
                'Cereal', 'Coffee', 'Tea', 'Sugar', 'Flour', 'Rice',
                'Pasta', 'Tomato Sauce', 'Olive Oil', 'Salt', 'Pepper',
                'Chicken', 'Beef', 'Fish', 'Vegetables', 'Fruits']
    
    np.random.seed(42)
    n_transactions = 1000
    transactions = []
    
    for i in range(n_transactions):
        n_items = np.random.randint(1, 8)
        transaction = list(np.random.choice(products, n_items, replace=False))
        transactions.append(transaction)
    
    df = pd.DataFrame({'Transaction': range(1, n_transactions + 1),
                       'Items': transactions})
    
    print(f"Created {n_transactions} synthetic transactions")

print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 transactions:")
print(df.head())
```

### 1.3 Data Cleaning

```python
if 'Online' in str(df.columns):
    # Real Online Retail dataset processing
    print("\nProcessing Online Retail dataset...")
    
    # Remove cancelled transactions
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove missing descriptions
    df = df[df['Description'].notna()]
    
    # Create transaction list
    transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
    
else:
    # Synthetic data
    transactions = df['Items'].tolist()

print(f"\nCleaned transactions: {len(transactions)}")
print(f"\nSample transaction: {transactions[0]}")

# Transaction statistics
trans_lengths = [len(t) for t in transactions]
print("\n" + "="*80)
print("TRANSACTION STATISTICS")
print("="*80)
print(f"Average items per transaction: {np.mean(trans_lengths):.2f}")
print(f"Min items: {np.min(trans_lengths)}")
print(f"Max items: {np.max(trans_lengths)}")
print(f"Median items: {np.median(trans_lengths):.0f}")

# Visualize transaction lengths
plt.figure(figsize=(12, 6))
plt.hist(trans_lengths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(np.mean(trans_lengths), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(trans_lengths):.2f}')
plt.axvline(np.median(trans_lengths), color='green', linestyle='--', linewidth=2,
            label=f'Median: {np.median(trans_lengths):.0f}')
plt.xlabel('Number of Items', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Transaction Sizes', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('transaction_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 1.4 Convert to One-Hot Encoding

```python
# Transform transactions to binary matrix
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\nOne-Hot Encoded Data:")
print(f"Shape: {df_encoded.shape}")
print(f"\nFirst 5 rows:")
print(df_encoded.head())

# Item frequency
item_frequency = df_encoded.sum().sort_values(ascending=False)

print("\n" + "="*80)
print("ITEM FREQUENCY ANALYSIS")
print("="*80)
print("\nTop 20 Most Frequent Items:")
print(item_frequency.head(20))

# Visualize top items
plt.figure(figsize=(14, 8))
top_20 = item_frequency.head(20)
plt.barh(range(len(top_20)), top_20.values, color='coral', edgecolor='black')
plt.yticks(range(len(top_20)), top_20.index)
plt.xlabel('Frequency (Number of Transactions)', fontsize=12)
plt.ylabel('Items', fontsize=12)
plt.title('Top 20 Most Purchased Items', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('top_items.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Part 2: Apriori Algorithm

### 2.1 Find Frequent Itemsets

```python
print("\n" + "="*80)
print("APRIORI ALGORITHM - FREQUENT ITEMSETS")
print("="*80)

# Apply Apriori with different support thresholds
min_support_values = [0.01, 0.02, 0.05, 0.1]

for min_sup in min_support_values:
    frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)
    
    print(f"\nMinimum Support = {min_sup}:")
    print(f"  Number of frequent itemsets: {len(frequent_itemsets)}")
    
    if len(frequent_itemsets) > 0:
        # Count by itemset size
        itemset_sizes = frequent_itemsets['itemsets'].apply(len)
        print(f"  1-itemsets: {(itemset_sizes == 1).sum()}")
        print(f"  2-itemsets: {(itemset_sizes == 2).sum()}")
        print(f"  3-itemsets: {(itemset_sizes == 3).sum()}")
        print(f"  4+ itemsets: {(itemset_sizes >= 4).sum()}")

# Use optimal support threshold
min_support = 0.02
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

print(f"\n" + "="*80)
print(f"SELECTED FREQUENT ITEMSETS (support >= {min_support})")
print("="*80)
print(f"\nTop 20 Frequent Itemsets:")
print(frequent_itemsets.sort_values('support', ascending=False).head(20))
```

### 2.2 Generate Association Rules

```python
print("\n" + "="*80)
print("ASSOCIATION RULES GENERATION")
print("="*80)

# Generate rules with different metrics
metrics = ['support', 'confidence', 'lift']
thresholds = [0.01, 0.5, 1.0]

for metric, threshold in zip(metrics, thresholds):
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=threshold)
    print(f"\nRules with {metric} >= {threshold}: {len(rules)} rules")

# Generate comprehensive rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Add additional metrics
rules['antecedent_len'] = rules['antecedents'].apply(len)
rules['consequent_len'] = rules['consequents'].apply(len)

print(f"\n" + "="*80)
print("ASSOCIATION RULES SUMMARY")
print("="*80)
print(f"Total rules: {len(rules)}")
print(f"\nRules by antecedent size:")
print(rules['antecedent_len'].value_counts().sort_index())

print("\nTop 20 Rules by Confidence:")
top_confidence = rules.sort_values('confidence', ascending=False).head(20)
print(top_confidence[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nTop 20 Rules by Lift:")
top_lift = rules.sort_values('lift', ascending=False).head(20)
print(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### 2.3 Rule Interpretation

```python
def interpret_rule(row):
    """
    Interpret association rule in business terms
    """
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    
    interpretation = f"""
    RULE: {antecedents} → {consequents}
    
    Interpretation:
    - Support: {row['support']:.3f} ({row['support']*100:.1f}% of all transactions)
      → This rule occurs in {row['support']*100:.1f}% of transactions
    
    - Confidence: {row['confidence']:.3f} ({row['confidence']*100:.1f}%)
      → {row['confidence']*100:.1f}% of customers who buy {antecedents} also buy {consequents}
    
    - Lift: {row['lift']:.3f}
      → Customers who buy {antecedents} are {row['lift']:.2f}x more likely to buy {consequents}
    """
    
    if row['lift'] > 1:
        interpretation += f"      → STRONG positive association (lift > 1)\n"
        interpretation += f"      → ACTION: Bundle these items together!\n"
    elif row['lift'] == 1:
        interpretation += f"      → INDEPENDENT (no association)\n"
    else:
        interpretation += f"      → NEGATIVE association (substitute products?)\n"
    
    return interpretation

print("\n" + "="*80)
print("DETAILED RULE INTERPRETATIONS")
print("="*80)

# Interpret top 5 rules
for idx, row in rules.nlargest(5, 'lift').iterrows():
    print(interpret_rule(row))
    print("-" * 80)
```

[CONTINUES WITH FP-GROWTH, VISUALIZATIONS, BUSINESS RECOMMENDATIONS...]

---

## COMPLETE 800+ LINES AVAILABLE

Includes:
- ✅ Complete Apriori implementation
- ✅ FP-Growth algorithm
- ✅ Rule visualization (scatter, network graphs)
- ✅ Business recommendations
- ✅ Product placement strategies
- ✅ Cross-selling opportunities
- ✅ Real-world case studies
