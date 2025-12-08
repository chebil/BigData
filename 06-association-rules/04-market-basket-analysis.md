# Market Basket Analysis - Real-World Applications

## Learning Objectives

- Apply association rules to retail data
- Implement end-to-end market basket analysis
- Generate actionable business insights
- Create product recommendation systems
- Optimize store layouts and promotions
- Handle real-world data challenges

## Introduction

Market Basket Analysis (MBA) uses association rules to discover **which products customers buy together**. This enables:

- **Cross-selling**: Recommend complementary products
- **Store layout optimization**: Place related items near each other
- **Promotional bundling**: Create attractive product bundles
- **Inventory management**: Stock complementary items together

## Complete Retail Analysis

### Dataset Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Load retail transaction data
# Format: TransactionID, Items (comma-separated)
data = {
    'TransactionID': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5,
                      6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10],
    'Item': ['Milk', 'Bread', 'Butter', 'Beer', 'Diapers', 
             'Milk', 'Bread', 'Butter', 'Eggs', 'Bread', 'Butter',
             'Milk', 'Bread', 'Eggs', 'Beer', 'Chips', 'Salsa',
             'Milk', 'Eggs', 'Bread', 'Milk', 'Butter',
             'Beer', 'Diapers', 'Milk', 'Bread', 'Eggs']
}

df_long = pd.DataFrame(data)

print("Retail Transaction Data (Long Format):")
print(df_long.head(10))

print(f"\nDataset Statistics:")
print(f"Total transactions: {df_long['TransactionID'].nunique()}")
print(f"Total items sold: {len(df_long)}")
print(f"Unique products: {df_long['Item'].nunique()}")

# Convert to transaction list
transactions = df_long.groupby('TransactionID')['Item'].apply(list).tolist()

print("\nTransaction List Format:")
for i, transaction in enumerate(transactions[:5], 1):
    print(f"Transaction {i}: {transaction}")

# Item frequency analysis
item_freq = df_long['Item'].value_counts()

print("\nItem Frequency:")
print(item_freq)

# Visualize item frequency
plt.figure(figsize=(10, 6))
item_freq.plot(kind='barh', color='skyblue', edgecolor='black')
plt.xlabel('Frequency')
plt.title('Product Purchase Frequency')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Exploratory Data Analysis

```python
# Transaction size distribution
trans_sizes = df_long.groupby('TransactionID').size()

print("\nTransaction Size Statistics:")
print(trans_sizes.describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(trans_sizes, bins=range(1, trans_sizes.max()+2), 
             edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Items per Transaction')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Transaction Sizes')
axes[0].grid(alpha=0.3)

# Cumulative distribution
axes[1].hist(trans_sizes, bins=range(1, trans_sizes.max()+2),
             cumulative=True, density=True, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Items per Transaction')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Cumulative Distribution of Transaction Sizes')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Average basket size
avg_basket_size = trans_sizes.mean()
print(f"\nAverage basket size: {avg_basket_size:.2f} items")
```

### Mining Association Rules

```python
# Convert to one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("One-Hot Encoded Transactions:")
print(df_encoded.head())

# Find frequent itemsets using FP-Growth (faster)
frequent_itemsets = fpgrowth(df_encoded, min_support=0.2, use_colnames=True)

print(f"\n" + "="*70)
print(f"FREQUENT ITEMSETS (min_support = 20%)")
print("="*70)
print(frequent_itemsets.sort_values('support', ascending=False))

# Analyze by itemset size
for size in sorted(frequent_itemsets['itemsets'].apply(len).unique()):
    itemsets_of_size = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == size]
    print(f"\n{size}-itemsets: {len(itemsets_of_size)}")
    for _, row in itemsets_of_size.head(5).iterrows():
        items = ', '.join(sorted(row['itemsets']))
        print(f"  {{{items}}}: {row['support']:.1%}")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Add additional metrics
rules['antecedent_len'] = rules['antecedents'].apply(len)
rules['consequent_len'] = rules['consequents'].apply(len)

# Sort by lift
rules = rules.sort_values('lift', ascending=False)

print(f"\n" + "="*70)
print(f"ASSOCIATION RULES (min_confidence = 50%)")
print("="*70)
print(f"Total rules: {len(rules)}")

# Display top rules
print("\nTop 10 Rules by Lift:")
for idx, row in rules.head(10).iterrows():
    ant = ', '.join(sorted(row['antecedents']))
    cons = ', '.join(sorted(row['consequents']))
    print(f"\n{idx+1}. {{{ant}}} → {{{cons}}}")
    print(f"   Support: {row['support']:.2%}")
    print(f"   Confidence: {row['confidence']:.2%}")
    print(f"   Lift: {row['lift']:.2f}")
    
    # Business interpretation
    if row['lift'] > 1:
        print(f"   ✓ Customers who buy {ant} are {row['lift']:.2f}x more likely to buy {cons}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter plot: Support vs Confidence (sized by lift)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Support vs Confidence
scatter = axes[0, 0].scatter(rules['support'], rules['confidence'], 
                             s=rules['lift']*50, alpha=0.6, 
                             c=rules['lift'], cmap='viridis')
axes[0, 0].set_xlabel('Support', fontsize=12)
axes[0, 0].set_ylabel('Confidence', fontsize=12)
axes[0, 0].set_title('Support vs Confidence (bubble size = lift)', fontsize=14)
axes[0, 0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 0], label='Lift')

# 2. Support vs Lift
axes[0, 1].scatter(rules['support'], rules['lift'], 
                   s=rules['confidence']*200, alpha=0.6,
                   c=rules['confidence'], cmap='plasma')
axes[0, 1].set_xlabel('Support', fontsize=12)
axes[0, 1].set_ylabel('Lift', fontsize=12)
axes[0, 1].set_title('Support vs Lift (bubble size = confidence)', fontsize=14)
axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Lift = 1')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Lift distribution
axes[1, 0].hist(rules['lift'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Lift = 1')
axes[1, 0].set_xlabel('Lift', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Lift Values', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Confidence distribution
axes[1, 1].hist(rules['confidence'], bins=30, edgecolor='black', 
                alpha=0.7, color='green')
axes[1, 1].set_xlabel('Confidence', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution of Confidence Values', fontsize=14)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('market_basket_analysis.png', dpi=300)
plt.show()
```

### Network Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create network graph of top rules
top_rules = rules.head(15)

G = nx.DiGraph()

for idx, row in top_rules.iterrows():
    antecedents = ', '.join(sorted(row['antecedents']))
    consequents = ', '.join(sorted(row['consequents']))
    
    G.add_edge(antecedents, consequents, 
               weight=row['lift'],
               confidence=row['confidence'])

# Draw network
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, iterations=50)

# Node sizes based on degree
node_sizes = [3000 + 1000 * G.degree(node) for node in G.nodes()]

# Edge widths based on lift
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)
widths = [5 * (w / max_weight) for w in weights]

# Draw
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                       node_color='lightblue', alpha=0.9, edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, 
                       edge_color='gray', arrows=True, arrowsize=20)

plt.title('Product Association Network\n(Node size = centrality, Edge width = lift)', 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('product_network.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nMost Central Products (hub products):")
centrality = nx.degree_centrality(G)
for node, cent in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node}: {cent:.3f}")
```

## Business Applications

### 1. Product Recommendations

```python
class ProductRecommender:
    def __init__(self, rules):
        self.rules = rules
    
    def recommend(self, basket, top_n=5):
        """
        Recommend products based on current basket
        """
        basket_set = set(basket)
        recommendations = []
        
        for _, rule in self.rules.iterrows():
            # Check if antecedents are in basket
            if rule['antecedents'].issubset(basket_set):
                # Recommend consequents not already in basket
                new_items = rule['consequents'] - basket_set
                if new_items:
                    for item in new_items:
                        recommendations.append({
                            'item': item,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'reason': f"Often bought with {', '.join(sorted(rule['antecedents']))}"
                        })
        
        # Remove duplicates and sort by confidence * lift
        recommendations_df = pd.DataFrame(recommendations)
        if not recommendations_df.empty:
            recommendations_df['score'] = recommendations_df['confidence'] * recommendations_df['lift']
            recommendations_df = recommendations_df.drop_duplicates('item')
            recommendations_df = recommendations_df.sort_values('score', ascending=False)
            return recommendations_df.head(top_n)
        return pd.DataFrame()

# Example usage
recommender = ProductRecommender(rules)

current_basket = ['Milk', 'Bread']
print(f"\nCustomer's current basket: {current_basket}")
print("\nRecommended products:")

recommendations = recommender.recommend(current_basket, top_n=3)
if not recommendations.empty:
    for idx, rec in recommendations.iterrows():
        print(f"\n{idx+1}. {rec['item']}")
        print(f"   Confidence: {rec['confidence']:.1%}")
        print(f"   Lift: {rec['lift']:.2f}")
        print(f"   Reason: {rec['reason']}")
else:
    print("No recommendations available")
```

### 2. Store Layout Optimization

```python
def analyze_product_placement(rules, min_lift=1.2):
    """
    Suggest product placements based on association rules
    """
    strong_rules = rules[rules['lift'] >= min_lift].copy()
    
    # Create placement groups
    placement_groups = {}
    
    for _, rule in strong_rules.iterrows():
        items = rule['antecedents'].union(rule['consequents'])
        items_sorted = tuple(sorted(items))
        
        if items_sorted not in placement_groups:
            placement_groups[items_sorted] = {
                'items': items,
                'avg_lift': [],
                'rules_count': 0
            }
        
        placement_groups[items_sorted]['avg_lift'].append(rule['lift'])
        placement_groups[items_sorted]['rules_count'] += 1
    
    # Calculate average lift for each group
    for group in placement_groups.values():
        group['avg_lift'] = np.mean(group['avg_lift'])
    
    # Sort by average lift
    sorted_groups = sorted(placement_groups.items(), 
                          key=lambda x: x[1]['avg_lift'], 
                          reverse=True)
    
    print("\n" + "="*70)
    print("STORE LAYOUT RECOMMENDATIONS")
    print("="*70)
    print("\nPlace these products near each other:")
    
    for idx, (items_tuple, info) in enumerate(sorted_groups[:10], 1):
        items_str = ', '.join(sorted(info['items']))
        print(f"\n{idx}. {{{items_str}}}")
        print(f"   Average Lift: {info['avg_lift']:.2f}")
        print(f"   Supporting Rules: {info['rules_count']}")
        print(f"   Action: Create product cluster or place in same aisle")

analyze_product_placement(rules, min_lift=1.5)
```

### 3. Promotional Bundling

```python
def create_bundles(rules, min_confidence=0.7, min_lift=1.5):
    """
    Create product bundles for promotions
    """
    bundle_candidates = rules[
        (rules['confidence'] >= min_confidence) & 
        (rules['lift'] >= min_lift)
    ].copy()
    
    bundles = []
    
    for _, rule in bundle_candidates.iterrows():
        bundle = {
            'products': list(rule['antecedents'].union(rule['consequents'])),
            'confidence': rule['confidence'],
            'lift': rule['lift'],
            'support': rule['support'],
            'expected_sales': rule['support']  # Percentage of customers
        }
        bundles.append(bundle)
    
    # Remove duplicate bundles
    unique_bundles = []
    seen = set()
    
    for bundle in bundles:
        bundle_key = tuple(sorted(bundle['products']))
        if bundle_key not in seen:
            seen.add(bundle_key)
            unique_bundles.append(bundle)
    
    # Sort by lift * confidence
    unique_bundles.sort(key=lambda x: x['lift'] * x['confidence'], reverse=True)
    
    print("\n" + "="*70)
    print("PROMOTIONAL BUNDLE SUGGESTIONS")
    print("="*70)
    
    for idx, bundle in enumerate(unique_bundles[:5], 1):
        products = ', '.join(sorted(bundle['products']))
        print(f"\nBundle {idx}: {{{products}}}")
        print(f"  Expected Purchase Rate: {bundle['expected_sales']:.1%}")
        print(f"  Customer Likelihood: {bundle['confidence']:.1%}")
        print(f"  Lift: {bundle['lift']:.2f}")
        print(f"  Recommended Discount: {min(15, 5 * bundle['lift']):.0f}%")
        print(f"  Action: Create 'Buy Together, Save More' promotion")

create_bundles(rules, min_confidence=0.6, min_lift=1.3)
```

### 4. Inventory Management

```python
def inventory_insights(rules, frequent_itemsets):
    """
    Provide inventory management insights
    """
    print("\n" + "="*70)
    print("INVENTORY MANAGEMENT INSIGHTS")
    print("="*70)
    
    # High-frequency items
    high_freq = frequent_itemsets[frequent_itemsets['support'] >= 0.4]
    print("\n1. HIGH-DEMAND ITEMS (Stock generously):")
    for _, row in high_freq.iterrows():
        if len(row['itemsets']) == 1:
            item = list(row['itemsets'])[0]
            print(f"   - {item}: {row['support']:.1%} of transactions")
    
    # Frequently bundled items
    strong_pairs = rules[
        (rules['antecedent_len'] == 1) & 
        (rules['consequent_len'] == 1) &
        (rules['lift'] > 1.5)
    ].head(5)
    
    print("\n2. COMPLEMENTARY PAIRS (Stock together):")
    for _, rule in strong_pairs.iterrows():
        ant = list(rule['antecedents'])[0]
        cons = list(rule['consequents'])[0]
        print(f"   - {ant} + {cons}: Lift={rule['lift']:.2f}")
        print(f"     Action: When {ant} stock is low, check {cons} inventory")
    
    # Seasonal/promotional items
    print("\n3. PROMOTIONAL OPPORTUNITY (Bundle for sales):")
    promo_rules = rules[rules['lift'] > 2].head(3)
    for _, rule in promo_rules.iterrows():
        items = list(rule['antecedents'].union(rule['consequents']))
        print(f"   - {', '.join(items)}")
        print(f"     Lift: {rule['lift']:.2f}")
        print(f"     Action: Create promotion to boost sales")

inventory_insights(rules, frequent_itemsets)
```

## Big Data Considerations

### Scaling to Large Datasets

```python
print("""
SCALING MARKET BASKET ANALYSIS:

1. DATA SAMPLING:
   - Use representative sample for algorithm tuning
   - Validate on full dataset
   - Stratified sampling by transaction size

2. DISTRIBUTED COMPUTING:
   - Use Spark MLlib for massive datasets
   - Parallel FP-Growth implementation
   - Distribute across cluster

3. OPTIMIZATION TECHNIQUES:
   - Increase min_support to reduce itemsets
   - Filter items by minimum frequency first
   - Use FP-Growth instead of Apriori
   - Prune rules aggressively

4. INCREMENTAL UPDATES:
   - Don't recompute from scratch daily
   - Update rules incrementally
   - Use sliding time windows

5. STORAGE:
   - Store frequent itemsets in database
   - Cache commonly accessed rules
   - Use compression for transaction logs
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Market basket analysis** discovers product purchase patterns
2. **Cross-selling opportunities** identified through association rules
3. **Store layout** optimized by placing associated items together
4. **Bundle promotions** created from high-lift rules
5. **Inventory management** improved through complementary product insights
6. **Recommendation systems** built using association rules
7. **Lift > 1** indicates positive association (buy together)
8. **High confidence** rules are reliable for recommendations
9. **Business context** essential for interpreting rules
10. **Actionable insights** more valuable than statistical significance
:::

## Further Reading

- Berry, M. & Linoff, G. (2004). "Data Mining Techniques" - Chapter on Market Basket Analysis
- Kaur, M. & Kang, S. (2016). "Market Basket Analysis: Identify the Changing Trends of Market Data"
- Retail Analytics: [Best Practices](https://www.ibm.com/topics/market-basket-analysis)
- Real-world Case Studies: Amazon, Walmart, Target
