# FP-Growth Algorithm

## Learning Objectives

- Understand FP-Growth algorithm and FP-tree structure
- Compare FP-Growth with Apriori
- Implement FP-Growth for frequent pattern mining
- Build and traverse FP-trees
- Apply FP-Growth to large datasets efficiently

## Introduction

**FP-Growth** (Frequent Pattern Growth) is a faster alternative to Apriori that:
- **Avoids candidate generation**
- **Requires only 2 database scans**
- **Uses compressed tree structure** (FP-tree)
- **More efficient for large datasets**

## FP-Growth vs Apriori

```python
import pandas as pd

comparison = pd.DataFrame({
    'Aspect': ['Database Scans', 'Candidate Generation', 'Memory Usage', 
               'Speed', 'Best For'],
    'Apriori': ['Multiple (k scans for k-itemsets)', 'Yes (generates candidates)', 
                'Lower', 'Slower for large datasets', 'Small-medium datasets'],
    'FP-Growth': ['Two (build tree + mine)', 'No (pattern growth)', 
                  'Higher (tree structure)', 'Faster for large datasets', 
                  'Large datasets']
})

print("APRIORI vs FP-GROWTH:")
print(comparison.to_string(index=False))

print("""
\nKEY DIFFERENCES:

1. CANDIDATE GENERATION:
   - Apriori: Generates and tests candidates
   - FP-Growth: Grows patterns directly

2. DATABASE SCANS:
   - Apriori: k+1 scans for k-itemsets
   - FP-Growth: Only 2 scans

3. APPROACH:
   - Apriori: Breadth-first (level-wise)
   - FP-Growth: Depth-first (divide-and-conquer)
""")
```

## FP-Tree Structure

### Example

```python
print("""
FP-TREE STRUCTURE:

Transaction Database:
  T1: {Bread, Milk}
  T2: {Bread, Eggs, Milk, Butter}
  T3: {Milk, Eggs}
  T4: {Bread, Milk, Butter}
  T5: {Bread, Eggs}

Item Frequencies (sorted by frequency):
  Bread: 4, Milk: 4, Eggs: 3, Butter: 2

FP-Tree (items ordered by frequency):

                Root
                  |
           Bread:4 ── Milk:2
              |           |
           Milk:2      Eggs:2
              |           |
           Eggs:1    Butter:1
              |
          Butter:1

Header Table (links to nodes):
  Bread → Bread:4
  Milk  → Milk:2 → Milk:2
  Eggs  → Eggs:1 → Eggs:2
  Butter → Butter:1 → Butter:1
""")
```

## FP-Growth Algorithm

```python
print("""
FP-GROWTH ALGORITHM:

Input:
  - Transaction database D
  - Minimum support threshold (min_sup)

Output:
  - All frequent itemsets

Phase 1: BUILD FP-TREE
  1. Scan database to find frequent 1-itemsets
  2. Sort items by frequency (descending)
  3. Scan database again to build FP-tree:
     - For each transaction:
       a. Sort items by frequency order
       b. Insert into tree (share common prefixes)

Phase 2: MINE FP-TREE
  1. For each frequent item (bottom-up):
     a. Find conditional pattern base
     b. Build conditional FP-tree
     c. Mine conditional FP-tree recursively
  2. Combine results to get all frequent patterns
""")
```

## Basic Implementation

```python
from collections import defaultdict, Counter

class FPNode:
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # Link to next node with same item

class FPTree:
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.transactions = transactions
        self.header_table = {}
        
        # Build the tree
        self.root = FPNode(None, 0)
        self._build_tree()
    
    def _build_tree(self):
        # First scan: count item frequencies
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        n_trans = len(self.transactions)
        frequent_items = {item: count for item, count in item_counts.items()
                         if count / n_trans >= self.min_support}
        
        if not frequent_items:
            return
        
        # Initialize header table
        for item in frequent_items:
            self.header_table[item] = None
        
        # Second scan: build tree
        for transaction in self.transactions:
            # Filter and sort transaction by frequency
            filtered = [item for item in transaction if item in frequent_items]
            sorted_items = sorted(filtered, 
                                key=lambda x: frequent_items[x], 
                                reverse=True)
            
            if sorted_items:
                self._insert_transaction(sorted_items, self.root)
    
    def _insert_transaction(self, items, node):
        if not items:
            return
        
        first_item = items[0]
        
        if first_item in node.children:
            # Increment count
            node.children[first_item].count += 1
        else:
            # Create new node
            new_node = FPNode(first_item, 1, node)
            node.children[first_item] = new_node
            
            # Update header table
            if self.header_table[first_item] is None:
                self.header_table[first_item] = new_node
            else:
                # Add to linked list
                current = self.header_table[first_item]
                while current.link is not None:
                    current = current.link
                current.link = new_node
        
        # Recursively insert remaining items
        if len(items) > 1:
            self._insert_transaction(items[1:], node.children[first_item])
    
    def mine_patterns(self):
        """
        Mine frequent patterns from FP-tree
        """
        patterns = {}
        
        # Mine patterns for each item (bottom-up by frequency)
        items = sorted(self.header_table.keys(),
                      key=lambda x: self._get_item_support(x))
        
        for item in items:
            # Get conditional pattern base
            patterns_with_item = self._find_patterns(item)
            patterns.update(patterns_with_item)
        
        return patterns
    
    def _get_item_support(self, item):
        """Calculate total support for an item"""
        count = 0
        node = self.header_table[item]
        while node is not None:
            count += node.count
            node = node.link
        return count / len(self.transactions)
    
    def _find_patterns(self, item):
        """
        Find all patterns containing given item
        """
        patterns = {}
        
        # Get all paths containing this item
        paths = []
        node = self.header_table[item]
        
        while node is not None:
            path = []
            parent = node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                paths.append((path[::-1], node.count))
            node = node.link
        
        # The item itself is a pattern
        support = self._get_item_support(item)
        patterns[frozenset([item])] = support
        
        return patterns

# Example usage
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Eggs', 'Milk', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Milk', 'Butter'],
    ['Bread', 'Eggs']
]

print("Transactions:")
for i, t in enumerate(transactions, 1):
    print(f"T{i}: {t}")

# Build FP-tree
fp_tree = FPTree(transactions, min_support=0.4)

print("\nHeader Table (frequent items):")
for item, node in fp_tree.header_table.items():
    support = fp_tree._get_item_support(item)
    print(f"{item}: {support:.1%}")

# Mine patterns
patterns = fp_tree.mine_patterns()

print("\nFrequent Patterns:")
for pattern, support in sorted(patterns.items(), key=lambda x: (-len(x[0]), -x[1])):
    items = ', '.join(sorted(pattern))
    print(f"{{{items}}}: {support:.1%}")
```

## Using pyfpgrowth Library

```python
import pyfpgrowth

# Transactions
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

print("Using pyfpgrowth library:")
print(f"Transactions: {len(transactions)}")

# Find frequent patterns
min_support = int(0.3 * len(transactions))  # 30%
patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)

print(f"\nFrequent Patterns (min_support={min_support}):")
for pattern, count in sorted(patterns.items(), key=lambda x: (-len(x[0]), -x[1])):
    support = count / len(transactions)
    items = ', '.join(sorted(pattern))
    print(f"{{{items}}}: count={count}, support={support:.1%}")

# Generate rules
rules = pyfpgrowth.generate_association_rules(patterns, min_confidence=0.6)

print(f"\nAssociation Rules (min_confidence=60%):")
for antecedent, consequents in rules.items():
    for consequent, confidence in consequents.items():
        ant = ', '.join(sorted(antecedent))
        cons = ', '.join(sorted(consequent))
        print(f"{{{ant}}} → {{{cons}}}: confidence={confidence:.1%}")
```

## Using mlxtend (FP-Growth option)

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time

# Larger dataset for comparison
import numpy as np
np.random.seed(42)

# Generate 1000 transactions
products = ['Milk', 'Bread', 'Eggs', 'Butter', 'Cheese', 
            'Yogurt', 'Juice', 'Cereal', 'Coffee', 'Tea',
            'Sugar', 'Flour', 'Rice', 'Pasta', 'Sauce']

transactions = []
for _ in range(1000):
    n_items = np.random.randint(2, 8)
    basket = list(np.random.choice(products, n_items, replace=False))
    transactions.append(basket)

print(f"Generated {len(transactions)} transactions")

# Convert to DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Compare Apriori vs FP-Growth
from mlxtend.frequent_patterns import apriori

min_support = 0.05

# Apriori
start = time.time()
frequent_apriori = apriori(df, min_support=min_support, use_colnames=True)
apriori_time = time.time() - start

# FP-Growth
start = time.time()
frequent_fpgrowth = fpgrowth(df, min_support=min_support, use_colnames=True)
fpgrowth_time = time.time() - start

print(f"\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"Apriori:    {len(frequent_apriori)} itemsets in {apriori_time:.3f}s")
print(f"FP-Growth:  {len(frequent_fpgrowth)} itemsets in {fpgrowth_time:.3f}s")
print(f"Speedup:    {apriori_time/fpgrowth_time:.2f}x faster with FP-Growth")

# Generate rules
rules_fpgrowth = association_rules(frequent_fpgrowth, metric="confidence", min_threshold=0.5)
rules_fpgrowth = rules_fpgrowth.sort_values('lift', ascending=False)

print(f"\nGenerated {len(rules_fpgrowth)} rules")
print("\nTop 5 Rules by Lift:")
for idx, row in rules_fpgrowth.head().iterrows():
    ant = ', '.join(sorted(row['antecedents']))
    cons = ', '.join(sorted(row['consequents']))
    print(f"{{{ant}}} → {{{cons}}}")
    print(f"  Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.1%}")
```

## Conditional Pattern Base Example

```python
print("""
CONDITIONAL PATTERN BASE:

Example: Mining patterns for item 'Butter'

FP-Tree paths containing 'Butter':
  1. Bread:Milk:Butter (count=2)
  2. Bread:Eggs:Butter (count=1)

Conditional Pattern Base for 'Butter':
  {Bread:Milk} : 2
  {Bread:Eggs} : 1

Conditional FP-Tree for 'Butter':
  Only 'Bread' is frequent (count=3 ≥ min_support)
  
  Root
    |
  Bread:3

Frequent Patterns with 'Butter':
  {Butter} : support from original tree
  {Bread, Butter} : support=3
""")
```

## Scalability Comparison

```python
import matplotlib.pyplot as plt
import numpy as np
import time
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def benchmark_algorithms(n_transactions_list, n_items=20):
    apriori_times = []
    fpgrowth_times = []
    
    for n_trans in n_transactions_list:
        # Generate data
        np.random.seed(42)
        items = [f"Item_{i}" for i in range(n_items)]
        transactions = []
        for _ in range(n_trans):
            n = np.random.randint(3, 8)
            basket = list(np.random.choice(items, n, replace=False))
            transactions.append(basket)
        
        # Convert to DataFrame
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Benchmark Apriori
        start = time.time()
        apriori(df, min_support=0.05, use_colnames=True)
        apriori_times.append(time.time() - start)
        
        # Benchmark FP-Growth
        start = time.time()
        fpgrowth(df, min_support=0.05, use_colnames=True)
        fpgrowth_times.append(time.time() - start)
    
    return apriori_times, fpgrowth_times

# Benchmark
n_transactions_list = [100, 500, 1000, 2000, 5000]
apriori_times, fpgrowth_times = benchmark_algorithms(n_transactions_list)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_transactions_list, apriori_times, 'o-', label='Apriori', linewidth=2)
plt.plot(n_transactions_list, fpgrowth_times, 's-', label='FP-Growth', linewidth=2)
plt.xlabel('Number of Transactions')
plt.ylabel('Time (seconds)')
plt.title('Scalability: Apriori vs FP-Growth')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\nScalability Results:")
for n, ap_t, fp_t in zip(n_transactions_list, apriori_times, fpgrowth_times):
    speedup = ap_t / fp_t
    print(f"n={n:5d}: Apriori={ap_t:.3f}s, FP-Growth={fp_t:.3f}s, Speedup={speedup:.2f}x")
```

## When to Use FP-Growth

```python
print("""
CHOOSING BETWEEN APRIORI AND FP-GROWTH:

USE APRIORI WHEN:
  ✓ Small to medium datasets (< 10,000 transactions)
  ✓ Low number of unique items (< 100)
  ✓ Memory is limited
  ✓ Simple implementation preferred
  ✓ Need to understand algorithm easily

USE FP-GROWTH WHEN:
  ✓ Large datasets (> 10,000 transactions)
  ✓ Many unique items
  ✓ Low minimum support threshold
  ✓ Performance is critical
  ✓ Memory available for tree structure
  ✓ Dense transaction data

BOTTOM LINE:
  - FP-Growth is generally 10-100x faster than Apriori
  - FP-Growth uses more memory
  - Both find the same frequent itemsets
  - For Big Data applications, use FP-Growth
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **FP-Growth is faster than Apriori** for large datasets
2. **Only 2 database scans** vs multiple for Apriori
3. **No candidate generation** - grows patterns directly
4. **FP-tree compresses database** into compact structure
5. **Divide-and-conquer approach** via conditional trees
6. **Higher memory usage** for tree structure
7. **Same results as Apriori** - just faster
8. **Best for large, dense datasets**
9. **Scales better** with increasing transactions
10. **Preferred for Big Data** applications
:::

## Further Reading

- Han, J. et al. (2000). "Mining Frequent Patterns without Candidate Generation"
- Han, J. et al. (2004). "Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach"
- Grahne, G. & Zhu, J. (2005). "Fast Algorithms for Frequent Itemset Mining Using FP-Trees"
- mlxtend FP-Growth: [fpgrowth](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
