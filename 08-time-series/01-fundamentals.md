# Time Series Analysis - Fundamentals

## Learning Objectives

- Understand time series data characteristics
- Identify trends, seasonality, and cycles
- Test for stationarity
- Perform time series decomposition
- Handle missing values and outliers
- Prepare time series data for modeling

## Introduction

**Time Series**: Sequence of data points indexed in time order

**Examples**:
- Stock prices
- Weather data
- Sales figures
- Website traffic
- Sensor readings
- Economic indicators

## Time Series Components

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("""
TIME SERIES COMPONENTS:

1. TREND (T):
   - Long-term increase or decrease
   - Overall direction of the series
   
2. SEASONALITY (S):
   - Regular, periodic fluctuations
   - Fixed and known period (day, week, month, year)
   
3. CYCLICAL (C):
   - Long-term oscillations
   - No fixed period
   - Often related to economic cycles
   
4. IRREGULAR/NOISE (I):
   - Random variations
   - Unpredictable component

ADDITIVE MODEL: Y(t) = T(t) + S(t) + C(t) + I(t)
MULTIPLICATIVE MODEL: Y(t) = T(t) × S(t) × C(t) × I(t)
""")

# Generate synthetic time series
np.random.seed(42)
n_points = 365 * 3  # 3 years of daily data

dates = pd.date_range(start='2021-01-01', periods=n_points, freq='D')

# Components
trend = np.linspace(100, 200, n_points)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365)  # Yearly
weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly
noise = np.random.normal(0, 5, n_points)

# Combine
y = trend + seasonality + weekly_seasonality + noise

# Create DataFrame
ts_df = pd.DataFrame({
    'date': dates,
    'value': y,
    'trend': trend,
    'seasonality': seasonality + weekly_seasonality,
    'noise': noise
})
ts_df.set_index('date', inplace=True)

print("\nTime Series Data:")
print(ts_df.head(10))

# Visualize components
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(ts_df.index, ts_df['value'], linewidth=1)
axes[0].set_ylabel('Original Series')
axes[0].set_title('Time Series Components', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(ts_df.index, ts_df['trend'], color='red', linewidth=2)
axes[1].set_ylabel('Trend')
axes[1].grid(alpha=0.3)

axes[2].plot(ts_df.index, ts_df['seasonality'], color='green', linewidth=1)
axes[2].set_ylabel('Seasonality')
axes[2].grid(alpha=0.3)

axes[3].plot(ts_df.index, ts_df['noise'], color='orange', linewidth=0.5, alpha=0.7)
axes[3].set_ylabel('Noise')
axes[3].set_xlabel('Date')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Stationarity

**Stationary Time Series**: Statistical properties (mean, variance) constant over time

**Why Important**: Most time series models assume stationarity

```python
print("""
STATIONARITY REQUIREMENTS:

1. Constant mean over time
2. Constant variance over time
3. Constant autocovariance for each lag

NON-STATIONARY INDICATORS:
  ✗ Trend
  ✗ Seasonality
  ✗ Changing variance
  ✗ Unit roots
""")

# Visual stationarity check
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Non-stationary (with trend)
axes[0, 0].plot(ts_df['value'])
axes[0, 0].set_title('Non-Stationary (Trend + Seasonality)')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(alpha=0.3)

# Stationary (differenced)
ts_diff = ts_df['value'].diff().dropna()
axes[0, 1].plot(ts_diff)
axes[0, 1].set_title('Stationary (First Difference)')
axes[0, 1].set_ylabel('Difference')
axes[0, 1].grid(alpha=0.3)

# Rolling statistics for non-stationary
rolling_mean = ts_df['value'].rolling(window=30).mean()
rolling_std = ts_df['value'].rolling(window=30).std()

axes[1, 0].plot(ts_df['value'], label='Original', alpha=0.5)
axes[1, 0].plot(rolling_mean, label='Rolling Mean', color='red', linewidth=2)
axes[1, 0].plot(rolling_std, label='Rolling Std', color='green', linewidth=2)
axes[1, 0].set_title('Non-Stationary: Rolling Statistics')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Rolling statistics for stationary
rolling_mean_diff = ts_diff.rolling(window=30).mean()
rolling_std_diff = ts_diff.rolling(window=30).std()

axes[1, 1].plot(ts_diff, label='Differenced', alpha=0.5)
axes[1, 1].plot(rolling_mean_diff, label='Rolling Mean', color='red', linewidth=2)
axes[1, 1].plot(rolling_std_diff, label='Rolling Std', color='green', linewidth=2)
axes[1, 1].set_title('Stationary: Rolling Statistics')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Augmented Dickey-Fuller Test

**Statistical test for stationarity**

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test
    """
    result = adfuller(series.dropna())
    
    print(f"\nADF Test Results for {name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  ✓ Series is STATIONARY (reject H0, p < 0.05)")
    else:
        print(f"  ✗ Series is NON-STATIONARY (fail to reject H0, p ≥ 0.05)")
    
    return result

# Test original series
adf_test(ts_df['value'], 'Original Series')

# Test differenced series
adf_test(ts_diff, 'Differenced Series')
```

## Time Series Decomposition

### Additive Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
decomposition = seasonal_decompose(ts_df['value'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

decomposition.observed.plot(ax=axes[0], title='Observed')
axes[0].grid(alpha=0.3)

decomposition.trend.plot(ax=axes[1], title='Trend')
axes[1].grid(alpha=0.3)

decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
axes[2].grid(alpha=0.3)

decomposition.resid.plot(ax=axes[3], title='Residual')
axes[3].grid(alpha=0.3)

plt.suptitle('Additive Decomposition', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# Statistics of residuals
print("\nResidual Statistics:")
print(f"Mean: {decomposition.resid.mean():.4f}")
print(f"Std: {decomposition.resid.std():.4f}")
print(f"Min: {decomposition.resid.min():.4f}")
print(f"Max: {decomposition.resid.max():.4f}")
```

### Multiplicative Decomposition

```python
# Create multiplicative series
trend_mult = np.linspace(10, 20, n_points)
seasonality_mult = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_points) / 365)
noise_mult = 1 + np.random.normal(0, 0.05, n_points)

y_mult = trend_mult * seasonality_mult * noise_mult

ts_mult = pd.Series(y_mult, index=dates)

# Multiplicative decomposition
decomp_mult = seasonal_decompose(ts_mult, model='multiplicative', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

decomp_mult.observed.plot(ax=axes[0], title='Observed')
axes[0].grid(alpha=0.3)

decomp_mult.trend.plot(ax=axes[1], title='Trend')
axes[1].grid(alpha=0.3)

decomp_mult.seasonal.plot(ax=axes[2], title='Seasonal')
axes[2].grid(alpha=0.3)

decomp_mult.resid.plot(ax=axes[3], title='Residual')
axes[3].grid(alpha=0.3)

plt.suptitle('Multiplicative Decomposition', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
```

## Autocorrelation

### ACF (Autocorrelation Function)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF
plot_acf(ts_df['value'], lags=50, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
axes[0].grid(alpha=0.3)

# ACF of differenced series
plot_acf(ts_diff, lags=50, ax=axes[1])
axes[1].set_title('ACF of Differenced Series')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
INTERPRETING ACF:

• Significant spikes: Correlation at those lags
• Slow decay: Non-stationarity (trend)
• Periodic spikes: Seasonality
• Quick decay to zero: Stationarity
• Blue area: 95% confidence interval
""")
```

### PACF (Partial Autocorrelation Function)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PACF
plot_pacf(ts_df['value'], lags=50, ax=axes[0])
axes[0].set_title('Partial Autocorrelation Function (PACF)')
axes[0].grid(alpha=0.3)

# PACF of differenced series
plot_pacf(ts_diff, lags=50, ax=axes[1], method='ywm')
axes[1].set_title('PACF of Differenced Series')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
ACF vs PACF:

ACF:
  - Shows total correlation at each lag
  - Includes indirect effects
  - Used to identify MA order

PACF:
  - Shows direct correlation at each lag
  - Removes indirect effects
  - Used to identify AR order
""")
```

## Handling Missing Values

```python
# Introduce missing values
ts_missing = ts_df['value'].copy()
missing_idx = np.random.choice(ts_missing.index, size=50, replace=False)
ts_missing.loc[missing_idx] = np.nan

print(f"\nMissing values: {ts_missing.isna().sum()}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original with missing
axes[0].plot(ts_missing, linewidth=1)
axes[0].scatter(missing_idx, [ts_df.loc[idx, 'value'] for idx in missing_idx], 
                color='red', s=50, label='Missing', zorder=5)
axes[0].set_title('Series with Missing Values')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Forward fill
ts_ffill = ts_missing.fillna(method='ffill')
axes[1].plot(ts_ffill, linewidth=1, color='green')
axes[1].set_title('Forward Fill')
axes[1].grid(alpha=0.3)

# Interpolation
ts_interp = ts_missing.interpolate(method='linear')
axes[2].plot(ts_interp, linewidth=1, color='purple')
axes[2].set_title('Linear Interpolation')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
MISSING VALUE STRATEGIES:

1. FORWARD/BACKWARD FILL:
   ✓ Simple and fast
   ✗ Not suitable for long gaps
   
2. LINEAR INTERPOLATION:
   ✓ Smooth transition
   ✓ Works well for short gaps
   ✗ May not capture patterns
   
3. SPLINE INTERPOLATION:
   ✓ Smooth curves
   ✗ Can overshoot
   
4. SEASONAL INTERPOLATION:
   ✓ Uses seasonal patterns
   ✓ Best for regular seasonality
""")
```

## Data Transformations

```python
# Different transformations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(ts_df['value'])
axes[0, 0].set_title('Original')
axes[0, 0].grid(alpha=0.3)

# Log transformation (for multiplicative effects)
ts_log = np.log(ts_df['value'] - ts_df['value'].min() + 1)
axes[0, 1].plot(ts_log)
axes[0, 1].set_title('Log Transform')
axes[0, 1].grid(alpha=0.3)

# First difference (remove trend)
ts_diff1 = ts_df['value'].diff().dropna()
axes[1, 0].plot(ts_diff1)
axes[1, 0].set_title('First Difference')
axes[1, 0].grid(alpha=0.3)

# Seasonal difference (remove seasonality)
ts_diff_seasonal = ts_df['value'].diff(periods=7).dropna()  # Weekly
axes[1, 1].plot(ts_diff_seasonal)
axes[1, 1].set_title('Seasonal Difference (7-day)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("""
COMMON TRANSFORMATIONS:

1. LOG: Stabilize variance, convert multiplicative to additive
2. SQUARE ROOT: Moderate variance stabilization
3. BOX-COX: Automatic transformation selection
4. DIFFERENCING: Remove trend and seasonality
5. STANDARDIZATION: Zero mean, unit variance
""")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Time series** have temporal dependencies
2. **Components**: Trend, seasonality, cyclical, noise
3. **Stationarity** required for most models
4. **ADF test** checks for stationarity
5. **Decomposition** separates components
6. **ACF/PACF** identify autocorrelation patterns
7. **Differencing** removes trend and seasonality
8. **Transformations** stabilize variance
9. **Missing values** require careful handling
10. **Visual analysis** essential for understanding patterns
:::

## Further Reading

- Box, G. & Jenkins, G. (1970). "Time Series Analysis: Forecasting and Control"
- Hyndman, R.J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice"
- Statsmodels Documentation: [Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
