# ARIMA Models

## Introduction

ARIMA (AutoRegressive Integrated Moving Average) combines three components to model time series:
- **AR(p)**: Autoregressive - uses p past values
- **I(d)**: Integrated - applies d differences
- **MA(q)**: Moving Average - uses q past errors

**ARIMA(p,d,q)** is the workhorse of time series forecasting.

## Autoregressive (AR) Models

### AR(1) Model

Simplest case: current value depends on previous value

\[
y_t = \phi_1 y_{t-1} + \epsilon_t
\]

Where:
- \(y_t\): Value at time t
- \(\phi_1\): AR coefficient
- \(\epsilon_t\): White noise error \(\sim N(0, \sigma^2)\)

**Interpretation**: Each value is a fraction of the previous value plus random shock

### General AR(p) Model

\[
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
\]

Where:
- \(c\): Constant term (related to mean)
- \(\phi_i\): AR coefficients for lag i
- \(p\): Order (number of lags)

### AR Model Characteristics

**Stationarity Condition**: 
- Roots of characteristic equation must lie outside unit circle
- For AR(1): \(|\phi_1| < 1\)

**Mean**:
\[
E[y_t] = \frac{c}{1 - \phi_1 - \phi_2 - ... - \phi_p}
\]

**ACF**: Decays exponentially or with damped oscillations

**PACF**: Cuts off after lag p

### Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ar(phi, n=200, sigma=1):
    """
    Simulate AR process
    phi: list of AR coefficients
    """
    p = len(phi)
    y = np.zeros(n)
    epsilon = np.random.normal(0, sigma, n)
    
    for t in range(p, n):
        y[t] = sum(phi[i] * y[t-i-1] for i in range(p)) + epsilon[t]
    
    return y

# AR(1) with phi = 0.7
ar1 = simulate_ar([0.7], n=200)

# AR(2) with phi = [0.5, 0.3]
ar2 = simulate_ar([0.5, 0.3], n=200)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(ar1)
axes[0].set_title('AR(1): $y_t = 0.7y_{t-1} + \epsilon_t$')
axes[0].grid(alpha=0.3)

axes[1].plot(ar2)
axes[1].set_title('AR(2): $y_t = 0.5y_{t-1} + 0.3y_{t-2} + \epsilon_t$')
axes[1].set_xlabel('Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Identifying AR Order

**Use PACF (Partial Autocorrelation Function)**:

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_acf(ar1, lags=20, ax=axes[0,0], title='ACF of AR(1)')
plot_pacf(ar1, lags=20, ax=axes[0,1], title='PACF of AR(1)')
plot_acf(ar2, lags=20, ax=axes[1,0], title='ACF of AR(2)')
plot_pacf(ar2, lags=20, ax=axes[1,1], title='PACF of AR(2)')

plt.tight_layout()
plt.show()
```

**Pattern**: 
- **ACF**: Gradual decay
- **PACF**: Cuts off sharply after lag p
  - AR(1): Spike at lag 1 only
  - AR(2): Spikes at lags 1 and 2 only

## Moving Average (MA) Models

### MA(1) Model

\[
y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1}
\]

Where:
- \(\mu\): Mean
- \(\epsilon_t\): Current error
- \(\theta_1\): MA coefficient
- \(\epsilon_{t-1}\): Previous error

**Interpretation**: Current value is weighted sum of current and past shocks

### General MA(q) Model

\[
y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
\]

Where:
- \(q\): Order (number of lagged error terms)
- \(\theta_i\): MA coefficients

### MA Model Characteristics

**Always Stationary**: Finite weighted sum of white noise

**Invertibility Condition**: 
- Roots of characteristic equation outside unit circle
- For MA(1): \(|\theta_1| < 1\)

**ACF**: Cuts off after lag q

**PACF**: Decays exponentially

### Simulation

```python
def simulate_ma(theta, n=200, sigma=1, mu=0):
    """
    Simulate MA process
    theta: list of MA coefficients
    """
    q = len(theta)
    epsilon = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    
    for t in range(n):
        y[t] = mu + epsilon[t]
        for i in range(min(q, t)):
            y[t] += theta[i] * epsilon[t-i-1]
    
    return y

# MA(1) with theta = 0.7
ma1 = simulate_ma([0.7], n=200)

# MA(2) with theta = [0.5, 0.3]
ma2 = simulate_ma([0.5, 0.3], n=200)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(ma1)
axes[0].set_title('MA(1): $y_t = \epsilon_t + 0.7\epsilon_{t-1}$')
axes[0].grid(alpha=0.3)

axes[1].plot(ma2)
axes[1].set_title('MA(2): $y_t = \epsilon_t + 0.5\epsilon_{t-1} + 0.3\epsilon_{t-2}$')
axes[1].set_xlabel('Time')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Identifying MA Order

**Use ACF**:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_acf(ma1, lags=20, ax=axes[0,0], title='ACF of MA(1)')
plot_pacf(ma1, lags=20, ax=axes[0,1], title='PACF of MA(1)')
plot_acf(ma2, lags=20, ax=axes[1,0], title='ACF of MA(2)')
plot_pacf(ma2, lags=20, ax=axes[1,1], title='PACF of MA(2)')

plt.tight_layout()
plt.show()
```

**Pattern**:
- **ACF**: Cuts off sharply after lag q
  - MA(1): Spike at lag 1 only
  - MA(2): Spikes at lags 1 and 2 only
- **PACF**: Gradual decay

## ARMA Models

### ARMA(p,q)

Combines AR and MA components:

\[
y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}
\]

**Compact notation**:
\[
\phi(B) y_t = \theta(B) \epsilon_t
\]

Where B is the backshift operator: \(B y_t = y_{t-1}\)

### ARMA Characteristics

**ACF**: Decays (doesn't cut off sharply)

**PACF**: Decays (doesn't cut off sharply)

**Both decay** → suggests ARMA model

### Simulation

```python
from statsmodels.tsa.arima_process import ArmaProcess

# ARMA(1,1): phi=0.7, theta=0.5
ar_params = np.array([0.7])
ma_params = np.array([0.5])

arma11_process = ArmaProcess.from_coeffs(ar_params, ma_params)
arma11 = arma11_process.generate_sample(nsample=200)

plt.figure(figsize=(12, 4))
plt.plot(arma11)
plt.title('ARMA(1,1): $y_t = 0.7y_{t-1} + \epsilon_t + 0.5\epsilon_{t-1}$')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(arma11, lags=20, ax=axes[0], title='ACF of ARMA(1,1)')
plot_pacf(arma11, lags=20, ax=axes[1], title='PACF of ARMA(1,1)')
plt.tight_layout()
plt.show()
```

## ARIMA Models

### ARIMA(p,d,q)

**ARMA on differenced data**:

1. Difference the series d times: \(y_t' = \nabla^d y_t\)
2. Apply ARMA(p,q) to \(y_t'\)

**Full model**:
\[
\phi(B) \nabla^d y_t = \theta(B) \epsilon_t
\]

### Components

- **p**: Order of AR component
- **d**: Number of differences (integration order)
- **q**: Order of MA component

### Common ARIMA Models

| Model | Meaning | Use Case |
|-------|---------|----------|
| ARIMA(1,0,0) | AR(1) | Short-term autocorrelation |
| ARIMA(0,1,1) | Random walk + MA(1) | Non-seasonal, trending data |
| ARIMA(1,1,1) | General purpose | Most common starting point |
| ARIMA(0,1,0) | Random walk | Stock prices |
| ARIMA(0,0,1) | MA(1) | White noise + shock |

### Model Selection Rules

**ACF and PACF Patterns**:

| ACF Pattern | PACF Pattern | Suggested Model |
|-------------|--------------|------------------|
| Cuts off after q | Decays | MA(q) |
| Decays | Cuts off after p | AR(p) |
| Decays | Decays | ARMA(p,q) |
| All zero | All zero | White noise |

**After Differencing**:
1. If still not stationary: Increase d
2. If ACF cuts off: MA model
3. If PACF cuts off: AR model
4. If both decay: ARMA model

## Estimation Methods

### Maximum Likelihood Estimation (MLE)

**Most common method**

Maximizes log-likelihood:
\[
\log L(\phi, \theta, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^n \epsilon_t^2
\]

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(1,1,1)
model = ARIMA(ts, order=(1, 1, 1))
results = model.fit()

print(results.summary())
print(f"\nAR coefficient: {results.arparams}")
print(f"MA coefficient: {results.maparams}")
print(f"Log-likelihood: {results.llf}")
print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
```

### Information Criteria

**AIC (Akaike Information Criterion)**:
\[
AIC = -2\log L + 2k
\]

**BIC (Bayesian Information Criterion)**:
\[
BIC = -2\log L + k\log(n)
\]

Where:
- \(L\): Maximum likelihood
- \(k\): Number of parameters
- \(n\): Sample size

**Lower is better**: Balances fit and complexity

**BIC penalizes complexity more** than AIC

## Implementation

### Manual ARIMA Fitting

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Example: Monthly temperatures
np.random.seed(42)
t = np.arange(120)
trend = 0.1 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, 120)
temps = 20 + trend + seasonal + noise

ts = pd.Series(temps, index=pd.date_range('2015-01', periods=120, freq='M'))

# Step 1: Plot
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Monthly Temperature Data')
plt.ylabel('Temperature (°C)')
plt.grid(alpha=0.3)
plt.show()

# Step 2: Check stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
print(f"ADF p-value: {result[1]:.4f}")

# Step 3: Difference if needed
ts_diff = ts.diff().dropna()
result = adfuller(ts_diff)
print(f"After differencing, p-value: {result[1]:.4f}")

# Step 4: Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts_diff, lags=20, ax=axes[0])
plot_pacf(ts_diff, lags=20, ax=axes[1])
plt.tight_layout()
plt.show()

# Step 5: Fit model (starting with ARIMA(1,1,1))
model = ARIMA(ts, order=(1, 1, 1))
results = model.fit()
print(results.summary())

# Step 6: Diagnostics
results.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### Automated Model Selection

```python
import pmdarima as pm

# Auto ARIMA (finds best p, d, q)
auto_model = pm.auto_arima(
    ts,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # Let it find optimal d
    seasonal=True,
    m=12,  # Seasonal period
    stepwise=True,  # Faster
    suppress_warnings=True,
    error_action='ignore',
    trace=True  # Print search progress
)

print(auto_model.summary())
print(f"\nBest model: ARIMA{auto_model.order}")
```

### Grid Search

```python
import itertools

def find_best_arima(ts, p_range, d_range, q_range):
    """
    Grid search for best ARIMA model
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    
    # Generate all combinations
    orders = list(itertools.product(p_range, d_range, q_range))
    
    results = []
    
    for order in orders:
        try:
            model = ARIMA(ts, order=order)
            fitted = model.fit()
            
            results.append({
                'order': order,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'mse': np.mean(fitted.resid**2)
            })
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_model = fitted
                
        except:
            continue
    
    results_df = pd.DataFrame(results).sort_values('aic')
    print("Top 5 models by AIC:")
    print(results_df.head())
    
    print(f"\nBest model: ARIMA{best_order}")
    print(f"AIC: {best_aic:.2f}")
    
    return best_model, results_df

# Search over reasonable ranges
best_model, results = find_best_arima(
    ts,
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3)
)
```

## Model Diagnostics

### Residual Analysis

**Residuals should be white noise**:

```python
residuals = results.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series plot
axes[0, 0].plot(residuals)
axes[0, 0].axhline(0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].grid(alpha=0.3)

# Histogram
axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_title('Histogram of Residuals')
axes[0, 1].set_xlabel('Residual')

# ACF
plot_acf(residuals, lags=20, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Statistical Tests

**Ljung-Box Test** (residuals are white noise):

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(lb_test)

if (lb_test['lb_pvalue'] > 0.05).all():
    print("\n✓ Residuals are white noise")
else:
    print("\n✗ Residuals show autocorrelation")
```

**Jarque-Bera Test** (normality):

```python
from scipy.stats import jarque_bera

jb_stat, jb_pvalue = jarque_bera(residuals)
print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_pvalue:.4f}")

if jb_pvalue > 0.05:
    print("✓ Residuals are normally distributed")
else:
    print("✗ Residuals are not normally distributed")
```

## Summary

**Model Identification Guide**:

| Step | Action | Tool |
|------|--------|------|
| 1 | Plot series | Visual inspection |
| 2 | Test stationarity | ADF test |
| 3 | Difference if needed | \(d = 1\) or \(d = 2\) |
| 4 | Plot ACF/PACF | Identify p and q |
| 5 | Fit model | MLE |
| 6 | Check diagnostics | Residual tests |
| 7 | Compare models | AIC/BIC |
| 8 | Forecast | If adequate |

**Key Takeaways**:

✅ **AR**: PACF cuts off at p
✅ **MA**: ACF cuts off at q
✅ **ARMA**: Both decay gradually
✅ **Difference first**: Make stationary before identifying p,q
✅ **Start simple**: ARIMA(1,1,1) is good baseline
✅ **Use AIC/BIC**: For model comparison
✅ **Check residuals**: Must be white noise

## Next Section

Continue to [Forecasting](06-forecasting.md) to learn how to generate predictions and confidence intervals.