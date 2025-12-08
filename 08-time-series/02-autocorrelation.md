# Autocorrelation and Partial Autocorrelation

## Introduction

Autocorrelation measures how a time series is related to its past values. Understanding autocorrelation patterns is crucial for:
- Identifying appropriate ARIMA model orders
- Detecting seasonality
- Validating model residuals
- Understanding temporal dependencies

## Autocorrelation Function (ACF)

### Definition

**Autocorrelation** at lag \(k\) measures the correlation between \(y_t\) and \(y_{t-k}\):

\[
\rho_k = \text{Corr}(y_t, y_{t-k}) = \frac{\text{Cov}(y_t, y_{t-k})}{\sigma^2}
\]

Where:
- \(\rho_k\): Autocorrelation at lag \(k\)
- \(\sigma^2\): Variance of the time series
- \(-1 \leq \rho_k \leq 1\)

### Sample Autocorrelation

For observed data \(y_1, y_2, ..., y_n\), the sample autocorrelation is:

\[
r_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n}(y_t - \bar{y})^2}
\]

### Properties

1. **\(\rho_0 = 1\)**: Perfect correlation with itself
2. **Symmetric**: \(\rho_k = \rho_{-k}\)
3. **Bounded**: \(|\rho_k| \leq 1\)
4. **Decay**: For stationary series, \(\rho_k \to 0\) as \(k \to \infty\)

## ACF Plot

### Creating ACF Plot

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# Simulate AR(1) process: y_t = 0.7 * y_{t-1} + e_t
np.random.seed(42)
n = 200
y = np.zeros(n)
y[0] = np.random.normal(0, 1)

for t in range(1, n):
    y[t] = 0.7 * y[t-1] + np.random.normal(0, 1)

# Plot time series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y)
axes[0].set_title('AR(1) Process: y_t = 0.7 * y_{t-1} + e_t')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].grid(alpha=0.3)

# Plot ACF
plot_acf(y, lags=40, ax=axes[1], alpha=0.05)
axes[1].set_title('Autocorrelation Function (ACF)')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')

plt.tight_layout()
plt.show()

# Get ACF values
acf_values, confint = acf(y, nlags=20, alpha=0.05)
print("ACF values:")
for lag, val in enumerate(acf_values[:10]):
    print(f"  Lag {lag}: {val:.4f}")
```

### Confidence Intervals

**95% confidence interval** for white noise:

\[
\pm \frac{1.96}{\sqrt{n}}
\]

- If \(|r_k|\) exceeds these bounds, autocorrelation is statistically significant
- Indicates that lag \(k\) should be considered in the model

**Example**:
- \(n = 100\): Bounds = \(\pm 0.196\)
- \(n = 200\): Bounds = \(\pm 0.139\)
- \(n = 500\): Bounds = \(\pm 0.088\)

## Partial Autocorrelation Function (PACF)

### Definition

**Partial autocorrelation** at lag \(k\) is the correlation between \(y_t\) and \(y_{t-k}\) **after removing** the linear dependence on \(y_{t-1}, y_{t-2}, ..., y_{t-k+1}\).

\[
\text{PACF}(k) = \text{Corr}(y_t - \hat{y}_t, y_{t-k} - \hat{y}_{t-k})
\]

Where \(\hat{y}_t\) is the linear prediction based on \(y_{t-1}, ..., y_{t-k+1}\).

### Intuition

**Question**: Does \(y_{t-k}\) provide additional information beyond \(y_{t-1}, ..., y_{t-k+1}\)?

**Example for lag 3**:
- ACF(3): Total correlation between \(y_t\) and \(y_{t-3}\)
- PACF(3): Direct correlation, removing effect of \(y_{t-1}\) and \(y_{t-2}\)

### PACF Plot

```python
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf

# Plot PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(y, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF')

plot_pacf(y, lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

# Get PACF values
pacf_values, confint = pacf(y, nlags=20, alpha=0.05)
print("\nPACF values:")
for lag, val in enumerate(pacf_values[:10]):
    print(f"  Lag {lag}: {val:.4f}")
```

## Identifying ARIMA Orders from ACF/PACF

### Pattern Recognition

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|-------------|
| **AR(p)** | Exponential decay or damped sine wave | Cuts off after lag p |
| **MA(q)** | Cuts off after lag q | Exponential decay or damped sine wave |
| **ARMA(p,q)** | Exponential decay | Exponential decay |

### AR(p) Process

**Autoregressive of order p**:

\[
y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \varepsilon_t
\]

**ACF**: Gradual decay (exponential or sinusoidal)
**PACF**: Cuts off after lag \(p\)

```python
# Simulate AR(2) process
np.random.seed(42)
n = 500
y_ar2 = np.zeros(n)

for t in range(2, n):
    y_ar2[t] = 0.6 * y_ar2[t-1] - 0.3 * y_ar2[t-2] + np.random.normal(0, 1)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(y_ar2)
axes[0].set_title('AR(2) Process: y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + e_t')
axes[0].grid(alpha=0.3)

plot_acf(y_ar2, lags=30, ax=axes[1])
axes[1].set_title('ACF: Gradual Decay')

plot_pacf(y_ar2, lags=30, ax=axes[2], method='ywm')
axes[2].set_title('PACF: Cuts off after lag 2')

plt.tight_layout()
plt.show()
```

**Interpretation**: 
- ACF shows gradual decay
- PACF shows significant values at lags 1 and 2, then cuts off
- **Conclusion**: AR(2) model

### MA(q) Process

**Moving Average of order q**:

\[
y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}
\]

**ACF**: Cuts off after lag \(q\)
**PACF**: Gradual decay (exponential or sinusoidal)

```python
# Simulate MA(2) process
from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(42)
ar_params = np.array([1])  # No AR component
ma_params = np.array([1, 0.5, 0.3])  # MA(2): theta_1=0.5, theta_2=0.3

y_ma2 = arma_generate_sample(ar_params, ma_params, nsample=500, scale=1)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(y_ma2)
axes[0].set_title('MA(2) Process: y_t = e_t + 0.5*e_{t-1} + 0.3*e_{t-2}')
axes[0].grid(alpha=0.3)

plot_acf(y_ma2, lags=30, ax=axes[1])
axes[1].set_title('ACF: Cuts off after lag 2')

plot_pacf(y_ma2, lags=30, ax=axes[2], method='ywm')
axes[2].set_title('PACF: Gradual Decay')

plt.tight_layout()
plt.show()
```

**Interpretation**:
- ACF shows significant values at lags 1 and 2, then cuts off
- PACF shows gradual decay
- **Conclusion**: MA(2) model

### ARMA(p,q) Process

**Combined Autoregressive Moving Average**:

\[
y_t = \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}
\]

**ACF**: Gradual decay
**PACF**: Gradual decay

```python
# Simulate ARMA(1,1) process
ar_params = np.array([1, -0.6])  # AR(1): phi_1=0.6
ma_params = np.array([1, 0.4])   # MA(1): theta_1=0.4

y_arma = arma_generate_sample(ar_params, ma_params, nsample=500, scale=1)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(y_arma)
axes[0].set_title('ARMA(1,1) Process')
axes[0].grid(alpha=0.3)

plot_acf(y_arma, lags=30, ax=axes[1])
axes[1].set_title('ACF: Gradual Decay')

plot_pacf(y_arma, lags=30, ax=axes[2], method='ywm')
axes[2].set_title('PACF: Gradual Decay')

plt.tight_layout()
plt.show()
```

**Interpretation**:
- Both ACF and PACF show gradual decay
- **Conclusion**: ARMA model (both components needed)

## Seasonal Patterns in ACF/PACF

### Identifying Seasonality

**Seasonal pattern**: Significant spikes at seasonal lags

```python
# Simulate seasonal data
np.random.seed(42)
n = 144  # 12 years of monthly data
t = np.arange(n)

# Trend + Seasonal + Noise
trend = 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, n)

y_seasonal = trend + seasonal + noise

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(y_seasonal)
axes[0].set_title('Time Series with Seasonal Pattern (Period=12)')
axes[0].set_xlabel('Time')
axes[0].grid(alpha=0.3)

plot_acf(y_seasonal, lags=48, ax=axes[1])
axes[1].set_title('ACF: Notice spikes at lags 12, 24, 36, 48')

plot_pacf(y_seasonal, lags=48, ax=axes[2], method='ywm')
axes[2].set_title('PACF')

plt.tight_layout()
plt.show()
```

**Key observations**:
- Spikes in ACF at lags 12, 24, 36, 48
- Indicates seasonal period \(s = 12\)
- Requires seasonal differencing or SARIMA model

## Practical Guidelines

### Step-by-Step Process

**1. Plot the time series**
- Visual inspection for trend, seasonality

**2. Check stationarity**
- ADF test
- Transform if needed (log, difference)

**3. Plot ACF and PACF**
- Use at least 2-3 seasonal periods for lags
- Monthly data: 36-48 lags
- Quarterly data: 12-16 lags

**4. Identify patterns**

| ACF | PACF | Tentative Model |
|-----|------|----------------|
| Decay | Cutoff at p | AR(p) |
| Cutoff at q | Decay | MA(q) |
| Decay | Decay | ARMA(p,q) |
| Seasonal spikes | Various | Add seasonal terms |

**5. Consider multiple models**
- Don't rely solely on visual inspection
- Try several candidate models
- Use information criteria (AIC, BIC)

### Common Mistakes

❌ **Ignoring non-stationarity**: Always difference first if needed
❌ **Using too few lags**: Miss seasonal patterns
❌ **Over-interpreting**: Not every spike is significant
❌ **Forgetting confidence intervals**: Use the bands!
❌ **Single model focus**: Always compare alternatives

## Real-World Example

```python
# Load airline passengers data
from statsmodels.datasets import get_rdataset

data = get_rdataset('AirPassengers').data
data['time'] = pd.date_range('1949-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

print("Step 1: Plot original series")
fig, axes = plt.subplots(4, 2, figsize=(14, 14))

# Original
ts.plot(ax=axes[0,0], title='Original Series')
plot_acf(ts, lags=48, ax=axes[0,1])
axes[0,1].set_title('ACF: Non-stationary (slow decay)')

# Log transform
ts_log = np.log(ts)
ts_log.plot(ax=axes[1,0], title='Log Transformed')
plot_acf(ts_log, lags=48, ax=axes[1,1])
axes[1,1].set_title('ACF: Still non-stationary')

# First difference
ts_diff = ts_log.diff().dropna()
ts_diff.plot(ax=axes[2,0], title='First Difference')
plot_acf(ts_diff, lags=48, ax=axes[2,1])
axes[2,1].set_title('ACF: Seasonal pattern at lag 12')

# Seasonal difference
ts_seasonal = ts_diff.diff(12).dropna()
ts_seasonal.plot(ax=axes[3,0], title='Seasonal Difference (lag 12)')
plot_acf(ts_seasonal, lags=48, ax=axes[3,1])
axes[3,1].set_title('ACF: Nearly stationary')

plt.tight_layout()
plt.show()

print("\nStep 2: Examine ACF and PACF of transformed series")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(ts_seasonal, lags=24, ax=axes[0])
axes[0].set_title('ACF: Spike at lag 1, small spike at lag 12')

plot_pacf(ts_seasonal, lags=24, ax=axes[1], method='ywm')
axes[1].set_title('PACF: Spike at lag 1, small spike at lag 12')

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("- Both ACF and PACF show spikes at lag 1")
print("- Small spikes at seasonal lag 12")
print("- Suggests: SARIMA(1,1,1)(1,1,1)[12]")
print("  or try: SARIMA(0,1,1)(0,1,1)[12]")
```

## Ljung-Box Test

**Statistical test** for autocorrelation:

**Null hypothesis**: No autocorrelation up to lag \(h\)

\[
Q = n(n+2) \sum_{k=1}^{h} \frac{r_k^2}{n-k}
\]

- \(Q\) follows \(\chi^2\) distribution with \(h\) degrees of freedom
- Reject H₀ if p-value < 0.05

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test for autocorrelation
result = acorr_ljungbox(ts_seasonal, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test Results:")
print(result)

if (result['lb_pvalue'] < 0.05).any():
    print("\n=> Significant autocorrelation detected")
    print("=> Model needs AR or MA components")
else:
    print("\n=> No significant autocorrelation")
    print("=> Series resembles white noise")
```

## Summary

**Key Takeaways**:

✅ **ACF measures** total correlation at each lag
✅ **PACF measures** direct correlation, removing intermediate effects
✅ **Pattern recognition** helps identify ARIMA orders:
   - AR(p): PACF cuts off at lag p
   - MA(q): ACF cuts off at lag q
   - ARMA: Both decay gradually
✅ **Seasonal spikes** indicate need for seasonal terms
✅ **Confidence intervals** determine significance
✅ **Always transform** to stationarity first
✅ **Consider multiple models** - don't rely on single pattern

**Decision Flow**:
1. Transform to stationarity
2. Plot ACF and PACF
3. Identify patterns
4. Propose candidate models
5. Fit and compare models
6. Validate with residual diagnostics

## Next Section

Continue to [ARIMA Models](03-arima-models.md) to learn the mathematical formulation and implementation of ARIMA models.