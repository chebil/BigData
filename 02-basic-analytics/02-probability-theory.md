# Probability Theory Basics

## Learning Objectives

- Understand fundamental probability concepts
- Calculate conditional probabilities and apply Bayes' theorem
- Work with probability rules and laws
- Apply probability to real-world data problems
- Understand independence and dependence

## Introduction

Probability theory provides the mathematical foundation for statistical inference, machine learning, and data analysis. Understanding probability is essential for interpreting uncertainty, making predictions, and building probabilistic models.

## Fundamental Concepts

### Probability Definition

**Classical (Theoretical) Probability**:
\[
P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
\]

**Empirical (Frequentist) Probability**:
\[
P(A) = \lim_{n \to \infty} \frac{\text{Number of times A occurs}}{n}
\]

**Subjective Probability**: Personal degree of belief

```python
import numpy as np
import matplotlib.pyplot as plt

# Classical: Fair die
die_outcomes = [1, 2, 3, 4, 5, 6]
P_rolling_4 = 1 / 6
print(f"Classical P(rolling 4) = {P_rolling_4:.4f}")

# Empirical: Simulate die rolls
np.random.seed(42)
rolls = np.random.randint(1, 7, size=10000)
P_rolling_4_empirical = np.mean(rolls == 4)
print(f"Empirical P(rolling 4) = {P_rolling_4_empirical:.4f}")

# Visualize convergence
n_trials = np.arange(1, 10001)
empirical_probs = np.array([np.mean(rolls[:n] == 4) for n in n_trials])

plt.figure(figsize=(10, 6))
plt.plot(n_trials, empirical_probs, linewidth=0.5, alpha=0.7)
plt.axhline(y=1/6, color='r', linestyle='--', label='Theoretical (1/6)')
plt.xlabel('Number of Rolls')
plt.ylabel('P(Rolling 4)')
plt.title('Law of Large Numbers: Empirical Probability Converges to Theoretical')
plt.xscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Sample Space and Events

**Sample Space (Ω)**: Set of all possible outcomes

**Event**: Subset of sample space

```python
# Example: Two coin flips
sample_space = ['HH', 'HT', 'TH', 'TT']
print(f"Sample space: {sample_space}")
print(f"Size: {len(sample_space)}")

# Event A: At least one head
event_A = ['HH', 'HT', 'TH']
P_A = len(event_A) / len(sample_space)
print(f"\nP(at least one head) = {P_A} = {P_A:.1%}")

# Event B: Both same
event_B = ['HH', 'TT']
P_B = len(event_B) / len(sample_space)
print(f"P(both same) = {P_B} = {P_B:.1%}")
```

## Probability Rules

### 1. Range Rule

\[
0 \leq P(A) \leq 1
\]

### 2. Sum Rule (Complement Rule)

\[
P(A) + P(A^c) = 1
\]

where \(A^c\) is the complement of A ("not A")

```python
# Example: Probability of rain
P_rain = 0.3
P_no_rain = 1 - P_rain
print(f"P(rain) = {P_rain}")
print(f"P(no rain) = {P_no_rain}")
```

### 3. Addition Rule

**For mutually exclusive events**:
\[
P(A \cup B) = P(A) + P(B)
\]

**For non-mutually exclusive events**:
\[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
\]

```python
import numpy as np

# Simulate card draws
np.random.seed(42)

# Deck of cards
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [(rank, suit) for suit in suits for rank in ranks]

print(f"Deck size: {len(deck)}")

# Event A: Draw a heart
hearts = [(r, s) for r, s in deck if s == 'Hearts']
P_heart = len(hearts) / len(deck)

# Event B: Draw a king
kings = [(r, s) for r, s in deck if r == 'K']
P_king = len(kings) / len(deck)

# Event A ∩ B: King of hearts
king_of_hearts = [(r, s) for r, s in deck if r == 'K' and s == 'Hearts']
P_king_and_heart = len(king_of_hearts) / len(deck)

# Addition rule
P_heart_or_king = P_heart + P_king - P_king_and_heart

print(f"\nP(Heart) = {P_heart:.4f}")
print(f"P(King) = {P_king:.4f}")
print(f"P(King AND Heart) = {P_king_and_heart:.4f}")
print(f"P(Heart OR King) = {P_heart_or_king:.4f}")

# Verify
heart_or_king = [(r, s) for r, s in deck if s == 'Hearts' or r == 'K']
P_verify = len(heart_or_king) / len(deck)
print(f"Verified: {P_verify:.4f}")
```

### 4. Multiplication Rule

**For independent events**:
\[
P(A \cap B) = P(A) \times P(B)
\]

**For dependent events**:
\[
P(A \cap B) = P(A) \times P(B|A)
\]

```python
# Independent: Two coin flips
P_heads_flip1 = 0.5
P_heads_flip2 = 0.5
P_both_heads = P_heads_flip1 * P_heads_flip2
print(f"P(HH) = {P_both_heads}")

# Dependent: Drawing cards without replacement
P_first_ace = 4 / 52
P_second_ace_given_first = 3 / 51  # Only 3 aces left in 51 cards
P_two_aces = P_first_ace * P_second_ace_given_first
print(f"\nP(two aces without replacement) = {P_two_aces:.4f}")
```

## Conditional Probability

**Definition**: Probability of A given B has occurred

\[
P(A|B) = \frac{P(A \cap B)}{P(B)}
\]

```python
import pandas as pd

# Example: Medical test
data = pd.DataFrame({
    'Disease': ['Yes', 'Yes', 'No', 'No'],
    'Test_Result': ['Positive', 'Negative', 'Positive', 'Negative'],
    'Count': [95, 5, 50, 850]
})

print(data)
print()

total = data['Count'].sum()

# P(Disease)
P_disease = data[data['Disease'] == 'Yes']['Count'].sum() / total
print(f"P(Disease) = {P_disease:.3f}")

# P(Positive test)
P_positive = data[data['Test_Result'] == 'Positive']['Count'].sum() / total
print(f"P(Positive test) = {P_positive:.3f}")

# P(Disease AND Positive)
P_disease_and_positive = data[(data['Disease'] == 'Yes') & 
                              (data['Test_Result'] == 'Positive')]['Count'].sum() / total
print(f"P(Disease AND Positive) = {P_disease_and_positive:.3f}")

# P(Disease | Positive) - Given positive test, probability of disease
P_disease_given_positive = P_disease_and_positive / P_positive
print(f"\nP(Disease | Positive test) = {P_disease_given_positive:.3f}")

# P(Positive | Disease) - Sensitivity
P_positive_given_disease = P_disease_and_positive / P_disease
print(f"P(Positive | Disease) = {P_positive_given_disease:.3f}")
```

## Bayes' Theorem

**Formula**:
\[
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
\]

**Expanded form**:
\[
P(A|B) = \frac{P(B|A) \times P(A)}{P(B|A) \times P(A) + P(B|A^c) \times P(A^c)}
\]

### Classic Example: Medical Diagnosis

```python
# Disease prevalence
P_disease = 0.01  # 1% of population has disease

# Test characteristics
sensitivity = 0.95  # P(Positive | Disease) = 95%
specificity = 0.90  # P(Negative | No Disease) = 90%

# Derived probabilities
P_no_disease = 1 - P_disease
P_positive_given_disease = sensitivity
P_positive_given_no_disease = 1 - specificity  # False positive rate

# P(Positive) - Total probability
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_no_disease * P_no_disease)

# Bayes' Theorem: P(Disease | Positive)
P_disease_given_positive = (
    P_positive_given_disease * P_disease / P_positive
)

print("Medical Test Analysis:")
print(f"Disease prevalence: {P_disease:.1%}")
print(f"Test sensitivity: {sensitivity:.1%}")
print(f"Test specificity: {specificity:.1%}")
print(f"\nP(Positive test) = {P_positive:.3f}")
print(f"\nP(Disease | Positive test) = {P_disease_given_positive:.1%}")
print(f"\nInsight: Even with 95% sensitivity, only {P_disease_given_positive:.1%} ")
print("of positive tests indicate actual disease (low prevalence)")
```

### Spam Filter Example

```python
import numpy as np

# Prior probabilities
P_spam = 0.3  # 30% of emails are spam
P_ham = 0.7   # 70% are legitimate

# Likelihoods: P(word | class)
# Word "free" appears in:
P_free_given_spam = 0.8  # 80% of spam emails
P_free_given_ham = 0.1   # 10% of legitimate emails

# Calculate P(free)
P_free = P_free_given_spam * P_spam + P_free_given_ham * P_ham

# Bayes: P(Spam | "free")
P_spam_given_free = (P_free_given_spam * P_spam) / P_free

print("Spam Filter Analysis:")
print(f"P(Spam) = {P_spam:.1%}")
print(f'P("free" | Spam) = {P_free_given_spam:.1%}')
print(f'P("free" | Ham) = {P_free_given_ham:.1%}')
print(f'\nP(Spam | "free") = {P_spam_given_free:.1%}')

# Naive Bayes with multiple words
words_in_email = ['free', 'winner', 'click']

# Likelihoods for each word
likelihoods_spam = {'free': 0.8, 'winner': 0.7, 'click': 0.6}
likelihoods_ham = {'free': 0.1, 'winner': 0.05, 'click': 0.15}

# Naive assumption: words are independent
P_words_given_spam = np.prod([likelihoods_spam[w] for w in words_in_email])
P_words_given_ham = np.prod([likelihoods_ham[w] for w in words_in_email])

# Unnormalized posteriors
posterior_spam = P_words_given_spam * P_spam
posterior_ham = P_words_given_ham * P_ham

# Normalize
P_spam_given_words = posterior_spam / (posterior_spam + posterior_ham)

print(f"\nWith multiple words {words_in_email}:")
print(f"P(Spam | words) = {P_spam_given_words:.1%}")
```

## Independence

**Definition**: Events A and B are independent if:
\[
P(A \cap B) = P(A) \times P(B)
\]

Equivalently:
\[
P(A|B) = P(A)
\]

```python
import numpy as np

# Test independence
def test_independence(P_A, P_B, P_A_and_B, tolerance=0.01):
    expected = P_A * P_B
    independent = abs(P_A_and_B - expected) < tolerance
    
    print(f"P(A) = {P_A:.3f}")
    print(f"P(B) = {P_B:.3f}")
    print(f"P(A ∩ B) = {P_A_and_B:.3f}")
    print(f"P(A) × P(B) = {expected:.3f}")
    print(f"Independent: {independent}\n")
    
    return independent

# Example 1: Two coin flips (independent)
print("Example 1: Two coin flips")
test_independence(P_A=0.5, P_B=0.5, P_A_and_B=0.25)

# Example 2: Drawing cards with replacement (independent)
print("Example 2: Cards with replacement")
P_first_heart = 13/52
P_second_heart = 13/52
P_both_hearts = (13/52) * (13/52)
test_independence(P_first_heart, P_second_heart, P_both_hearts)

# Example 3: Drawing cards without replacement (dependent)
print("Example 3: Cards without replacement")
P_first_heart = 13/52
P_second_heart = 13/52
P_both_hearts = (13/52) * (12/51)  # Dependent!
test_independence(P_first_heart, P_second_heart, P_both_hearts)
```

## Law of Total Probability

**Partition of sample space**: \(B_1, B_2, \ldots, B_n\) are mutually exclusive and exhaustive

\[
P(A) = \sum_{i=1}^{n} P(A|B_i) \times P(B_i)
\]

```python
# Example: Manufacturing defects
# Three factories produce products
factories = {
    'Factory A': {'proportion': 0.5, 'defect_rate': 0.02},
    'Factory B': {'proportion': 0.3, 'defect_rate': 0.03},
    'Factory C': {'proportion': 0.2, 'defect_rate': 0.01}
}

# Overall defect rate
P_defect = sum(
    factory['proportion'] * factory['defect_rate']
    for factory in factories.values()
)

print("Manufacturing Analysis:")
for name, factory in factories.items():
    print(f"{name}: {factory['proportion']:.0%} of products, "
          f"{factory['defect_rate']:.1%} defect rate")

print(f"\nOverall defect rate: {P_defect:.2%}")

# If we find a defect, which factory likely made it?
for name, factory in factories.items():
    P_factory_given_defect = (
        factory['defect_rate'] * factory['proportion'] / P_defect
    )
    print(f"P({name} | Defect) = {P_factory_given_defect:.1%}")
```

## Practical Applications

### A/B Testing with Probability

```python
import scipy.stats as stats

# Conversion rates from A/B test
control = {'visitors': 1000, 'conversions': 120}
treatment = {'visitors': 1000, 'conversions': 145}

# Empirical probabilities
P_convert_control = control['conversions'] / control['visitors']
P_convert_treatment = treatment['conversions'] / treatment['visitors']

print("A/B Test Results:")
print(f"Control conversion rate: {P_convert_control:.1%}")
print(f"Treatment conversion rate: {P_convert_treatment:.1%}")
print(f"Lift: {(P_convert_treatment / P_convert_control - 1):.1%}")

# Statistical significance (chi-square test)
contingency = [
    [control['conversions'], control['visitors'] - control['conversions']],
    [treatment['conversions'], treatment['visitors'] - treatment['conversions']]
]

chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\nStatistical test:")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result is statistically significant! (p < 0.05)")
else:
    print("Result is not statistically significant")
```

### Recommendation System Probability

```python
import numpy as np
import pandas as pd

# User-item interactions
data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'item_id': ['A', 'B', 'C', 'A', 'D', 'B', 'C', 'D', 'E']
})

# Calculate probabilities
total_interactions = len(data)

# P(item)
item_probs = data['item_id'].value_counts() / total_interactions
print("Item Popularity:")
print(item_probs)

# Conditional probability: P(item | user)
for user in data['user_id'].unique():
    user_data = data[data['user_id'] == user]
    print(f"\nUser {user} preferences:")
    probs = user_data['item_id'].value_counts() / len(user_data)
    print(probs)
```

## Monte Carlo Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_samples):
    """
    Estimate π using Monte Carlo simulation
    """
    # Random points in unit square [0,1] x [0,1]
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # Check if inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    
    # Estimate π
    pi_estimate = 4 * np.mean(inside_circle)
    
    return pi_estimate, x, y, inside_circle

# Run simulation
np.random.seed(42)
n_samples = 10000
pi_est, x, y, inside = estimate_pi(n_samples)

print(f"Estimated π: {pi_est:.4f}")
print(f"Actual π: {np.pi:.4f}")
print(f"Error: {abs(pi_est - np.pi):.4f}")

# Visualize
plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5, label='Inside circle')
plt.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5, label='Outside circle')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Monte Carlo Estimation of π: {pi_est:.4f}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Common Probability Mistakes

### 1. Gambler's Fallacy

```python
# Misconception: After 5 heads, tails is "due"
flips = ['H', 'H', 'H', 'H', 'H']

print("After 5 heads in a row:")
print(f"P(next flip is H) = 0.5 (still!)")
print(f"P(next flip is T) = 0.5 (not higher!)")
print("\nEach flip is independent!")
```

### 2. Conjunction Fallacy

```python
# P(A and B) cannot exceed P(A) or P(B)
P_A = 0.6
P_B = 0.4
P_A_and_B_max = min(P_A, P_B)

print(f"P(A) = {P_A}")
print(f"P(B) = {P_B}")
print(f"P(A and B) ≤ {P_A_and_B_max}")
print("\nP(A and B) ≤ min(P(A), P(B)) always!")
```

### 3. Base Rate Neglect

```python
# Ignoring prior probability (base rate)
print("Medical test with 99% accuracy:")
print("Disease prevalence: 0.1%")
print("\nMost positive tests are FALSE positives!")
print("(See Bayes' Theorem example above)")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Probability ranges from 0 to 1**
2. **Complement rule**: P(A) + P(not A) = 1
3. **Addition rule**: P(A or B) = P(A) + P(B) - P(A and B)
4. **Multiplication rule**: P(A and B) = P(A) × P(B|A)
5. **Conditional probability**: P(A|B) = P(A ∩ B) / P(B)
6. **Bayes' theorem** updates probabilities with new evidence
7. **Independence**: P(A|B) = P(A)
8. **Law of total probability** partitions sample space
:::

## Exercises

See `exercises/chapter-02-probability-exercises.md` for practice problems.

## Further Reading

- Ross, S. (2014). "A First Course in Probability" (9th Edition)
- Blitzstein, J. & Hwang, J. (2019). "Introduction to Probability" (2nd Edition)
- Downey, A. (2013). "Think Bayes: Bayesian Statistics in Python"
