# Common Errors and Solutions

## Python Errors

### ImportError

**Error**: `ImportError: No module named 'pandas'`

**Solution**:
```bash
pip install pandas
# or
conda install pandas
```

### KeyError

**Error**: `KeyError: 'column_name'`

**Solution**:
```python
# Check column names
print(df.columns)
# Use correct column name or check for typos
```

### ValueError

**Error**: `ValueError: could not convert string to float`

**Solution**:
```python
# Check data types
df['column'] = pd.to_numeric(df['column'], errors='coerce')
```

## Scikit-learn Errors

### Shape Mismatch

**Error**: `ValueError: X has 10 features but model expects 8`

**Solution**:
```python
# Ensure same preprocessing for train and test
# Use fit on training data, transform on both
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Memory Error

**Error**: `MemoryError`

**Solution**:
```python
# Use batch processing
# Reduce dataset size
# Use more efficient data types
df['column'] = df['column'].astype('int32')
```

## Jupyter Errors

### Kernel Dies

**Solution**:
1. Restart kernel
2. Clear output
3. Check memory usage
4. Reduce dataset size

### Module Not Found in Jupyter

**Solution**:
```bash
# Install in correct environment
python -m pip install package_name

# Or use magic command in notebook
!pip install package_name
```
