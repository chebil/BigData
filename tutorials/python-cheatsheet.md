# Python for Data Science Cheatsheet

## NumPy

```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
rand = np.random.rand(5)

# Operations
arr.mean()
arr.std()
arr.sum()
arr.reshape(3, 1)
```

## Pandas

```python
import pandas as pd

# DataFrames
df = pd.read_csv('file.csv')
df.head()
df.info()
df.describe()
df['column'].value_counts()
df.groupby('col').mean()
df.fillna(0)
df.drop_duplicates()
```

## Matplotlib

```python
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.scatter(x, y)
plt.hist(data, bins=20)
plt.xlabel('Label')
plt.title('Title')
plt.show()
```

## Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
