# Stationarity

## Introduction

Stationarity is the foundation of time series modeling. Most forecasting methods, including ARIMA, assume the underlying statistical properties of the data remain constant over time.

**Why stationarity matters**:
- Makes patterns easier to model
- Enables reliable parameter estimation
- Ensures forecasts are meaningful
- Prevents spurious regression

## Definition

A time series \(\{y_t\}\) is **strictly stationary** if the joint distribution of \((y_{t_1}, y_{t_2}, ..., y_{t_n})\) is identical to \((y_{t_1+h}, y_{t_2+h}, ..., y_{t_n+h})\) for all \(t_1, t_2, ..., t_n\) and all lag \(h\).

**Practical definition** (**Weak stationarity** or **Covariance stationarity**):

A time series \(\{y_t\}\) is stationary if:

1. **Constant mean**: \(E[y_t] = \mu\) for all \(t\)
2. **Constant variance**: \(Var(y_t) = \sigma^2\) for all \(t\)
3. **Autocovariance depends only on lag**: \(Cov(y_t, y_{t+h})\) depends only on \(h\), not on \(t\)

## Visual Inspection

### Stationary Series

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate stationary series
np.random.seed(42)
n = 200
stationary = np.random.normal(0, 1, n)
for i in range(1, n):
    stationary[i] = 0.5 * stationary[i-1] + np.random.normal(0, 1)

plt.figure(figsize=(12, 4))
plt.plot(stationary)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title('Stationary Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()
```

**Characteristics**:
- Fluctuates around constant mean
- Constant variance over time
- No obvious trend or seasonality

### Non-Stationary Series

```python
# Simulate non-stationary series (random walk with drift)
nonstationary = np.zeros(n)
nonstationary[0] = 0
for i in range(1, n):
    nonstationary[i] = nonstationary[i-1] + 0.5 + np.random.normal(0, 1)

plt.figure(figsize=(12, 4))
plt.plot(nonstationary)
plt.title('Non-Stationary Time Series (Random Walk with Drift)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()
```

**Characteristics**:
- Upward or downward trend
- Mean changes over time
- Variance may change

## Statistical Tests

### Augmented Dickey-Fuller (ADF) Test

**Most common stationarity test**

**Null hypothesis (H₀)**: Series has unit root (non-stationary)
**Alternative hypothesis (H₁)**: Series is stationary

**Decision rule**:
- If p-value < 0.05: Reject H₀ → Series is stationary
- If p-value ≥ 0.05: Fail to reject H₀ → Series is non-stationary

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f'ADF Test Results for {name}:')
    print(f'  ADF Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print(f"  => Reject null hypothesis")
        print(f"  => Series is STATIONARY")
    else:
        print(f"  => Fail to reject null hypothesis")
        print(f"  => Series is NON-STATIONARY")
    print()
    
    return result[1] <= 0.05

# Test on stationary series
stationary_series = pd.Series(stationary)
adf_test(stationary_series, 'Stationary Series')

# Test on non-stationary series
nonstationary_series = pd.Series(nonstationary)
adf_test(nonstationary_series, 'Non-Stationary Series')
```

Output:
```
ADF Test Results for Stationary Series:
  ADF Statistic: -5.2341
  p-value: 0.0000
  Critical Values:
    1%: -3.4639
    5%: -2.8763
    10%: -2.5746
  => Reject null hypothesis
  => Series is STATIONARY

ADF Test Results for Non-Stationary Series:
  ADF Statistic: -1.0234
  p-value: 0.7456
  Critical Values:
    1%: -3.4639
    5%: -2.8763
    10%: -2.5746
  => Fail to reject null hypothesis
  => Series is NON-STATIONARY
```

### KPSS Test

**Kwiatkowski-Phillips-Schmidt-Shin test**

**Null hypothesis (H₀)**: Series is stationary
**Alternative hypothesis (H₁)**: Series has unit root (non-stationary)

**Note**: KPSS is opposite of ADF!

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name=''):
    """
    Perform KPSS test
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    print(f'KPSS Test Results for {name}:')
    print(f'  KPSS Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Values:')
    for key, value in result[3].items():
        print(f'    {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print(f"  => Reject null hypothesis")
        print(f"  => Series is NON-STATIONARY")
    else:
        print(f"  => Fail to reject null hypothesis")
        print(f"  => Series is STATIONARY")
    print()
    
    return result[1] > 0.05

kpss_test(stationary_series, 'Stationary Series')
kpss_test(nonstationary_series, 'Non-Stationary Series')
```

### Combined Interpretation

| ADF Result | KPSS Result | Interpretation |
|------------|-------------|----------------|
| Stationary | Stationary | **Stationary** ✓ |
| Non-stationary | Stationary | Trend stationary |
| Stationary | Non-stationary | Difference stationary |
| Non-stationary | Non-stationary | **Non-stationary** ✗ |

**Best practice**: Use both tests for confirmation

## Types of Non-Stationarity

### 1. Trend Non-Stationarity

**Mean increases/decreases over time**

```python
# Linear trend
t = np.arange(200)
trend_series = 0.5 * t + np.random.normal(0, 5, 200)

plt.figure(figsize=(12, 4))
plt.plot(trend_series)
plt.title('Trend Non-Stationary Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()
```

### 2. Heteroscedastic (Non-Constant Variance)

**Variance changes over time**

```python
# Increasing variance
hetero = np.zeros(200)
for i in range(200):
    hetero[i] = np.random.normal(0, i/50 + 1)

plt.figure(figsize=(12, 4))
plt.plot(hetero)
plt.title('Heteroscedastic Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()
```

### 3. Seasonal Non-Stationarity

**Periodic patterns that repeat**

```python
# Seasonal pattern
t = np.arange(200)
seasonal = 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, 200)

plt.figure(figsize=(12, 4))
plt.plot(seasonal)
plt.title('Seasonal Non-Stationary Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()
```

## Transformations to Achieve Stationarity

### 1. Differencing

**Most common transformation**

**First-order differencing**:
\[
\nabla y_t = y_t - y_{t-1}
\]

```python
# Difference the non-stationary series
differenced = np.diff(nonstationary)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(nonstationary)
axes[0].set_title('Original Series (Non-Stationary)')
axes[0].grid(alpha=0.3)

axes[1].plot(differenced)
axes[1].set_title('Differenced Series (Stationary)')
axes[1].set_xlabel('Time')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Test stationarity
adf_test(pd.Series(differenced), 'Differenced Series')
```

**Second-order differencing** (difference of differences):
\[
\nabla^2 y_t = \nabla y_t - \nabla y_{t-1} = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})
\]

```python
# Second difference
second_diff = np.diff(nonstationary, n=2)

plt.figure(figsize=(12, 4))
plt.plot(second_diff)
plt.title('Second-Order Differenced Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.show()
```

**Seasonal differencing** (lag s):
\[
\nabla_s y_t = y_t - y_{t-s}
\]

```python
# Seasonal difference (period = 12)
def seasonal_difference(series, lag=12):
    return series[lag:] - series[:-lag]

seasonal_diff = seasonal_difference(seasonal, lag=12)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(seasonal)
axes[0].set_title('Original Seasonal Series')
axes[0].grid(alpha=0.3)

axes[1].plot(seasonal_diff)
axes[1].set_title('Seasonally Differenced Series')
axes[1].set_xlabel('Time')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. Detrending

**Remove trend by regression**

```python
from scipy import signal

# Linear detrending
detrended = signal.detrend(trend_series)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(trend_series)
axes[0].set_title('Original Series with Trend')
axes[0].grid(alpha=0.3)

axes[1].plot(detrended)
axes[1].set_title('Detrended Series')
axes[1].set_xlabel('Time')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Logarithmic Transformation

**Stabilize variance (for multiplicative seasonality)**

```python
# Series with increasing variance
exponential = np.exp(np.random.normal(0, 0.1, 200).cumsum())

# Log transform
log_transform = np.log(exponential)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(exponential)
axes[0].set_title('Original Series (Increasing Variance)')
axes[0].grid(alpha=0.3)

axes[1].plot(log_transform)
axes[1].set_title('Log-Transformed Series (Stable Variance)')
axes[1].set_xlabel('Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4. Box-Cox Transformation

**General power transformation**

\[
y_t^{(\lambda)} = \begin{cases}
\frac{y_t^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y_t) & \text{if } \lambda = 0
\end{cases}
\]

```python
from scipy.stats import boxcox

# Box-Cox transformation (requires positive values)
positive_series = exponential
transformed, lmbda = boxcox(positive_series)

print(f"Optimal lambda: {lmbda:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(positive_series)
axes[0].set_title('Original Series')
axes[0].grid(alpha=0.3)

axes[1].plot(transformed)
axes[1].set_title(f'Box-Cox Transformed (λ={lmbda:.2f})')
axes[1].set_xlabel('Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Determining Order of Differencing

### Method 1: Visual Inspection

**Check if series "looks" stationary after differencing**

### Method 2: ADF Test After Each Difference

```python
def find_optimal_difference(series, max_diff=3):
    """
    Find optimal number of differences using ADF test
    """
    for d in range(max_diff + 1):
        if d == 0:
            test_series = series
        else:
            test_series = np.diff(series, n=d)
        
        result = adfuller(test_series)
        print(f"d={d}: ADF Statistic={result[0]:.4f}, p-value={result[1]:.4f}")
        
        if result[1] <= 0.05:
            print(f"  => Series is stationary with d={d}")
            return d
    
    print(f"  => Series may need d > {max_diff} or other transformations")
    return max_diff

optimal_d = find_optimal_difference(nonstationary)
```

Output:
```
d=0: ADF Statistic=-1.0234, p-value=0.7456
d=1: ADF Statistic=-12.3456, p-value=0.0000
  => Series is stationary with d=1
```

### Method 3: NDIFFS (Automated)

```python
from pmdarima.arima import ndiffs

# Automated difference selection
d_adf = ndiffs(nonstationary, test='adf', max_d=3)
d_kpss = ndiffs(nonstationary, test='kpss', max_d=3)

print(f"ADF test suggests d={d_adf}")
print(f"KPSS test suggests d={d_kpss}")
print(f"Use d={max(d_adf, d_kpss)} to be conservative")
```

## Over-Differencing

**Warning**: Too much differencing can:
- Introduce unnecessary autocorrelation
- Increase variance
- Reduce forecast accuracy

```python
# Example of over-differencing
over_diff = np.diff(stationary, n=2)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(stationary)
axes[0].set_title('Original Stationary Series')
axes[0].grid(alpha=0.3)

axes[1].plot(over_diff)
axes[1].set_title('Over-Differenced Series (Increased Variance)')
axes[1].set_xlabel('Time')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Original variance: {np.var(stationary):.4f}")
print(f"Over-differenced variance: {np.var(over_diff):.4f}")
```

## Practical Guidelines

### Differencing Rules of Thumb

1. **d = 0**: Series is already stationary
2. **d = 1**: Most common, removes linear trend
3. **d = 2**: Rarely needed, removes quadratic trend
4. **d > 2**: Almost never needed, likely over-differencing

### Seasonal Differencing

- **D = 1**: One seasonal difference (most common)
- **D = 2**: Very rare, extreme seasonality

### Transformation Order

**Recommended sequence**:
1. **Log or Box-Cox** (if variance increases with level)
2. **Seasonal differencing** (if seasonal pattern)
3. **First-order differencing** (if still non-stationary)
4. **Check stationarity** (ADF test)
5. **Stop** when stationary

## Real-World Example

```python
# Load example data
from statsmodels.datasets import get_rdataset
air = get_rdataset('AirPassengers').data
air['time'] = pd.date_range('1949-01', periods=len(air), freq='M')
air.set_index('time', inplace=True)
ts = air['value']

print("Step 1: Original series")
adf_test(ts, 'Air Passengers')

# Take log to stabilize variance
ts_log = np.log(ts)
print("\nStep 2: After log transformation")
adf_test(ts_log, 'Log(Air Passengers)')

# First difference
ts_log_diff = ts_log.diff().dropna()
print("\nStep 3: After first differencing")
adf_test(ts_log_diff, 'Differenced Log(Air Passengers)')

# Seasonal difference
ts_log_diff_seasonal = ts_log_diff.diff(12).dropna()
print("\nStep 4: After seasonal differencing")
adf_test(ts_log_diff_seasonal, 'Seasonally Differenced')

# Plot all stages
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
ts.plot(ax=axes[0], title='Original')
ts_log.plot(ax=axes[1], title='Log Transformed')
ts_log_diff.plot(ax=axes[2], title='First Differenced')
ts_log_diff_seasonal.plot(ax=axes[3], title='Seasonally Differenced')

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Summary

**Key Takeaways**:

✅ **Always test stationarity** before modeling
✅ **Use both ADF and KPSS** for robust testing
✅ **Difference conservatively**: Start with d=1
✅ **Transform first**: Log/Box-Cox before differencing
✅ **Check residuals**: Even after model fitting
✅ **Avoid over-differencing**: More is not always better

**Decision Flow**:
1. Plot data → Visual inspection
2. ADF + KPSS tests → Statistical confirmation
3. Transform if needed → Stabilize variance
4. Difference → Remove trend/seasonality
5. Re-test → Confirm stationarity
6. Proceed to modeling → ACF/PACF analysis

## Next Section

Continue to [Autocorrelation](02-autocorrelation.md) to learn how to identify ARIMA components using ACF and PACF plots.