# Apriori Algorithm

## Learning Objectives

- Understand the Apriori principle
- Implement the Apriori algorithm
- Generate frequent itemsets efficiently
- Generate association rules from frequent itemsets
- Apply Apriori to real-world datasets
- Optimize Apriori performance

## The Apriori Principle

**Key Insight**: **If an itemset is frequent, all its subsets must also be frequent**

**Contrapositive**: **If an itemset is infrequent, all its supersets must also be infrequent**

This principle allows us to **prune the search space** dramatically!

```python
import numpy as np

print("""
APRIORI PRINCIPLE:

Example:
If {Milk, Bread, Eggs} is frequent (support ≥ min_sup)
Then ALL subsets are also frequent:
  {Milk, Bread}
  {Milk, Eggs}
  {Bread, Eggs}
  {Milk}
  {Bread}
  {Eggs}

Conversely:
If {Milk, Bread} is infrequent (support < min_sup)
Then ALL supersets are also infrequent:
  {Milk, Bread, Eggs}
  {Milk, Bread, Butter}
  {Milk, Bread, Eggs, Butter}
  
→ We can skip checking these supersets!
""")
```

## Algorithm Overview

```python
print("""
APRIORI ALGORITHM:

Input:
  - Transaction database D
  - Minimum support threshold (min_sup)

Output:
  - All frequent itemsets

Steps:
  1. Generate candidate 1-itemsets (C1)
  2. Scan database to count support
  3. Keep only frequent 1-itemsets (L1)
  4. REPEAT until no more frequent itemsets:
     a. Generate candidate k-itemsets (Ck) from Lk-1
     b. Scan database to count support
     c. Keep only frequent k-itemsets (Lk)
  5. Return all frequent itemsets (L1 ∪ L2 ∪ ... ∪ Lk)
""")
```

## Implementation from Scratch

```python
from itertools import combinations
from collections import defaultdict

class AprioriAlgorithm:
    def __init__(self, min_support=0.3):
        self.min_support = min_support
        self.frequent_itemsets = []
        self.all_frequent_itemsets = {}
        
    def fit(self, transactions):
        """
        Find all frequent itemsets using Apriori algorithm
        """
        self.transactions = transactions
        self.n_transactions = len(transactions)
        
        # Generate L1 (frequent 1-itemsets)
        L1 = self._get_frequent_1_itemsets()
        self.all_frequent_itemsets[1] = L1
        
        k = 2
        Lk_minus_1 = L1
        
        # Iterate until no more frequent itemsets
        while Lk_minus_1:
            # Generate candidate k-itemsets
            Ck = self._generate_candidates(Lk_minus_1, k)
            
            # Scan database and count support
            itemset_counts = self._count_support(Ck)
            
            # Filter frequent k-itemsets
            Lk = {itemset: support for itemset, support in itemset_counts.items()
                  if support >= self.min_support}
            
            if Lk:
                self.all_frequent_itemsets[k] = Lk
                Lk_minus_1 = Lk
                k += 1
            else:
                break
        
        return self
    
    def _get_frequent_1_itemsets(self):
        """
        Get frequent 1-itemsets
        """
        item_counts = defaultdict(int)
        
        # Count each item
        for transaction in self.transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        # Filter by minimum support
        frequent_items = {}
        for itemset, count in item_counts.items():
            support = count / self.n_transactions
            if support >= self.min_support:
                frequent_items[itemset] = support
        
        return frequent_items
    
    def _generate_candidates(self, Lk_minus_1, k):
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets
        """
        candidates = set()
        itemsets = list(Lk_minus_1.keys())
        
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join step: merge two (k-1)-itemsets if they share k-2 items
                union = itemsets[i].union(itemsets[j])
                if len(union) == k:
                    # Prune step: check if all subsets are frequent
                    if self._has_infrequent_subset(union, Lk_minus_1, k):
                        continue
                    candidates.add(union)
        
        return candidates
    
    def _has_infrequent_subset(self, candidate, Lk_minus_1, k):
        """
        Check if candidate has any infrequent (k-1)-subset
        """
        subsets = combinations(candidate, k - 1)
        for subset in subsets:
            if frozenset(subset) not in Lk_minus_1:
                return True
        return False
    
    def _count_support(self, candidates):
        """
        Count support for candidate itemsets
        """
        itemset_counts = defaultdict(int)
        
        for transaction in self.transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    itemset_counts[candidate] += 1
        
        # Convert counts to support
        itemset_support = {}
        for itemset, count in itemset_counts.items():
            itemset_support[itemset] = count / self.n_transactions
        
        return itemset_support
    
    def get_frequent_itemsets(self):
        """
        Return all frequent itemsets as a flat dictionary
        """
        all_itemsets = {}
        for k, itemsets in self.all_frequent_itemsets.items():
            all_itemsets.update(itemsets)
        return all_itemsets
    
    def generate_rules(self, min_confidence=0.6):
        """
        Generate association rules from frequent itemsets
        """
        rules = []
        
        # Only generate rules from itemsets of size ≥ 2
        for k in range(2, max(self.all_frequent_itemsets.keys()) + 1):
            if k not in self.all_frequent_itemsets:
                continue
                
            for itemset, support in self.all_frequent_itemsets[k].items():
                # Generate all possible rules from this itemset
                items = list(itemset)
                
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        antecedent_support = self._get_support(antecedent)
                        confidence = support / antecedent_support if antecedent_support > 0 else 0
                        
                        if confidence >= min_confidence:
                            # Calculate lift
                            consequent_support = self._get_support(consequent)
                            lift = support / (antecedent_support * consequent_support) if (antecedent_support * consequent_support) > 0 else 0
                            
                            rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        return rules
    
    def _get_support(self, itemset):
        """
        Get support for an itemset
        """
        for k_itemsets in self.all_frequent_itemsets.values():
            if itemset in k_itemsets:
                return k_itemsets[itemset]
        return 0

# Example usage
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs', 'Butter']
]

print("Transaction Database:")
for i, t in enumerate(transactions, 1):
    print(f"T{i}: {t}")

# Run Apriori
apriori = AprioriAlgorithm(min_support=0.3)
apriori.fit(transactions)

# Get frequent itemsets
frequent_itemsets = apriori.get_frequent_itemsets()

print(f"\n" + "="*70)
print(f"FREQUENT ITEMSETS (min_support = 30%)")
print("="*70)

for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (-len(x[0]), -x[1])):
    items = ', '.join(sorted(itemset))
    print(f"{{{items}}}: {support:.1%}")

# Generate rules
rules = apriori.generate_rules(min_confidence=0.6)

print(f"\n" + "="*70)
print(f"ASSOCIATION RULES (min_confidence = 60%)")
print("="*70)

for rule in sorted(rules, key=lambda x: x['lift'], reverse=True):
    ant = ', '.join(sorted(rule['antecedent']))
    cons = ', '.join(sorted(rule['consequent']))
    print(f"{{{ant}}} → {{{cons}}}")
    print(f"  Support: {rule['support']:.1%}, Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}")
    print()
```

## Using mlxtend Library

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Prepare data
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs', 'Butter']
]

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("One-Hot Encoded Transactions:")
print(df.head())

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets.sort_values('support', ascending=False))

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Add lift
rules = rules.sort_values('lift', ascending=False)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Scatter plot: Support vs Confidence
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Support vs Confidence
axes[0].scatter(rules['support'], rules['confidence'], 
                s=rules['lift']*20, alpha=0.6, c=rules['lift'], cmap='viridis')
axes[0].set_xlabel('Support')
axes[0].set_ylabel('Confidence')
axes[0].set_title('Support vs Confidence (size = lift)')
axes[0].grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
cbar.set_label('Lift')

# Support vs Lift
axes[1].scatter(rules['support'], rules['lift'], 
                s=rules['confidence']*100, alpha=0.6, c=rules['confidence'], cmap='plasma')
axes[1].set_xlabel('Support')
axes[1].set_ylabel('Lift')
axes[1].set_title('Support vs Lift (size = confidence)')
axes[1].axhline(y=1, color='red', linestyle='--', label='Lift = 1 (independence)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Heatmap of lift values
pivot_support = rules.pivot_table(
    values='lift',
    index='antecedents',
    columns='consequents',
    aggfunc='mean'
)

if not pivot_support.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_support, annot=True, fmt='.2f', cmap='RdYlGn', center=1)
    plt.title('Lift Heatmap (antecedents → consequents)')
    plt.tight_layout()
    plt.show()
```

## Complete Example: Grocery Store

```python
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Generate larger synthetic grocery dataset
np.random.seed(42)

products = ['Milk', 'Bread', 'Eggs', 'Butter', 'Cheese', 
            'Yogurt', 'Juice', 'Cereal', 'Coffee', 'Tea']

# Generate 500 transactions with realistic patterns
transactions = []

for _ in range(500):
    basket = []
    
    # Some products are more popular
    if np.random.rand() < 0.4:
        basket.append('Bread')
    if np.random.rand() < 0.3:
        basket.append('Milk')
    
    # Breakfast items often together
    if 'Milk' in basket and np.random.rand() < 0.6:
        basket.append('Cereal')
    
    # Dairy products often together
    if 'Milk' in basket and np.random.rand() < 0.4:
        basket.append('Butter')
    if 'Butter' in basket and np.random.rand() < 0.5:
        basket.append('Bread')
    
    # Beverages
    if np.random.rand() < 0.2:
        basket.append('Coffee' if np.random.rand() < 0.6 else 'Tea')
    
    # Add some random items
    for product in products:
        if product not in basket and np.random.rand() < 0.1:
            basket.append(product)
    
    if basket:  # Only add non-empty baskets
        transactions.append(basket)

print(f"Generated {len(transactions)} transactions")
print("\nSample transactions:")
for i in range(5):
    print(f"T{i+1}: {transactions[i]}")

# Convert to DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"\nDataset shape: {df.shape}")
print(f"Products: {list(df.columns)}")

# Item frequency
item_freq = df.sum().sort_values(ascending=False)
print("\nItem Frequencies:")
for item, count in item_freq.items():
    print(f"{item:10s}: {count:3d} ({count/len(df):.1%})")

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
print(f"\nFound {len(frequent_itemsets)} frequent itemsets")

# Display by size
for size in sorted(frequent_itemsets['itemsets'].apply(len).unique()):
    itemsets_of_size = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == size]
    print(f"\n{size}-itemsets: {len(itemsets_of_size)}")
    if len(itemsets_of_size) <= 10:
        for _, row in itemsets_of_size.iterrows():
            items = ', '.join(sorted(row['itemsets']))
            print(f"  {{{items}}}: {row['support']:.1%}")

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values('lift', ascending=False)

print(f"\n" + "="*80)
print(f"TOP 10 ASSOCIATION RULES")
print("="*80)

for idx, row in rules.head(10).iterrows():
    ant = ', '.join(sorted(row['antecedents']))
    cons = ', '.join(sorted(row['consequents']))
    print(f"\nRule: {{{ant}}} → {{{cons}}}")
    print(f"  Support: {row['support']:.2%}")
    print(f"  Confidence: {row['confidence']:.2%}")
    print(f"  Lift: {row['lift']:.2f}")
    
    # Interpretation
    if row['lift'] > 1:
        print(f"  ✓ Positive correlation: Buying {ant} increases likelihood of buying {cons}")
    print(f"  ✓ {row['confidence']:.0%} of customers who buy {ant} also buy {cons}")

# Export rules
rules.to_csv('association_rules.csv', index=False)
print("\nRules exported to 'association_rules.csv'")
```

## Performance Optimization

### 1. Pruning Strategies

```python
print("""
APRIORI OPTIMIZATION TECHNIQUES:

1. APRIORI PRINCIPLE:
   - Prune infrequent itemsets early
   - Don't generate supersets of infrequent itemsets

2. HASH-BASED PRUNING:
   - Use hash table to filter candidates
   - Reduce database scans

3. TRANSACTION REDUCTION:
   - Remove transactions that don't contain frequent k-itemsets
   - Smaller database for subsequent scans

4. PARTITIONING:
   - Divide database into partitions
   - Find local frequent itemsets
   - Merge and verify globally

5. SAMPLING:
   - Mine on sample of database
   - Verify on full database
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Apriori uses level-wise search** to find frequent itemsets
2. **Apriori principle** enables massive search space reduction
3. **Multiple database scans** required (one per itemset size)
4. **Candidate generation** from (k-1)-itemsets to k-itemsets
5. **Prune step** eliminates candidates with infrequent subsets
6. **Rule generation** from frequent itemsets is straightforward
7. **Computational complexity** grows exponentially with items
8. **Use higher min_support** to reduce computation
9. **mlxtend library** provides efficient implementation
10. **Works well for moderate-sized datasets**
:::

## Further Reading

- Agrawal, R. & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
- Park, J.S. et al. (1995). "An Effective Hash-Based Algorithm for Mining Association Rules"
- Savasere, A. et al. (1995). "An Efficient Algorithm for Mining Association Rules in Large Databases"
- mlxtend Documentation: [Apriori](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
