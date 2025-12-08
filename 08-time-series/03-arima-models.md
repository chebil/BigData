# ARIMA Models

## Introduction

ARIMA (Autoregressive Integrated Moving Average) is a powerful and flexible class of models for time series forecasting. ARIMA combines:
- **AR (Autoregressive)**: Uses past values
- **I (Integrated)**: Applies differencing for stationarity  
- **MA (Moving Average)**: Uses past forecast errors

**Notation**: ARIMA(p, d, q)
- **p**: Order of autoregressive component
- **d**: Degree of differencing
- **q**: Order of moving average component

## Autoregressive (AR) Models

### AR(1) Model

**First-order autoregressive**:

\[
y_t = c + \phi_1 y_{t-1} + \varepsilon_t
\]

Where:
- \(c\): Constant (intercept)
- \(\phi_1\): AR coefficient
- \(\varepsilon_t \sim N(0, \sigma^2)\): White noise error

**Mean**: \(\mu = \frac{c}{1 - \phi_1}\)

**Alternative form** (zero-mean):

\[
y_t - \mu = \phi_1(y_{t-1} - \mu) + \varepsilon_t
\]

### Stationarity Condition

**For AR(1) to be stationary**: \(|\phi_1| < 1\)

- \(\phi_1 > 0\): Positive autocorrelation (smooth series)
- \(\phi_1 < 0\): Negative autocorrelation (oscillating series)
- \(\phi_1 \approx 1\): Near random walk (non-stationary)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate AR(1) with different phi values
np.random.seed(42)
n = 200

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
phi_values = [0.9, 0.3, -0.6]

for i, phi in enumerate(phi_values):
    y = np.zeros(n)
    y[0] = np.random.normal(0, 1)
    
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(0, 1)
    
    axes[i].plot(y)
    axes[i].set_title(f'AR(1) with φ₁ = {phi}')
    axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### General AR(p) Model

**Autoregressive of order p**:

\[
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \varepsilon_t
\]

Or using **backshift operator** \(B\) (where \(By_t = y_{t-1}\)):

\[
\Phi(B) y_t = c + \varepsilon_t
\]

Where \(\Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - ... - \phi_p B^p\)

**Stationarity condition**: All roots of \(\Phi(B) = 0\) must lie outside unit circle

### AR(2) Example

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Simulate AR(2): y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + e_t
np.random.seed(42)
n = 500
y_ar2 = np.zeros(n)

for t in range(2, n):
    y_ar2[t] = 0.6 * y_ar2[t-1] - 0.3 * y_ar2[t-2] + np.random.normal(0, 1)

# Fit AR(2) model
model_ar2 = ARIMA(y_ar2, order=(2, 0, 0))
results_ar2 = model_ar2.fit()

print("\nAR(2) Model Results:")
print(results_ar2.summary())
print(f"\nTrue parameters: φ₁=0.6, φ₂=-0.3")
print(f"Estimated parameters: φ₁={results_ar2.params['ar.L1']:.4f}, φ₂={results_ar2.params['ar.L2']:.4f}")
```

## Moving Average (MA) Models

### MA(1) Model

**First-order moving average**:

\[
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}
\]

Where:
- \(\mu\): Mean of the process
- \(\theta_1\): MA coefficient
- \(\varepsilon_t \sim N(0, \sigma^2)\): White noise

**Key insight**: Current value depends on current and past **errors**, not past values

### Invertibility Condition

**For MA(1) to be invertible**: \(|\theta_1| < 1\)

- Invertibility ensures unique AR representation
- Analogous to stationarity for AR models

```python
# Simulate MA(1) with different theta values
from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(42)
n = 200

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
theta_values = [0.8, 0.3, -0.6]

for i, theta in enumerate(theta_values):
    ar_params = np.array([1])  # No AR
    ma_params = np.array([1, theta])  # MA(1)
    
    y = arma_generate_sample(ar_params, ma_params, nsample=n, scale=1)
    
    axes[i].plot(y)
    axes[i].set_title(f'MA(1) with θ₁ = {theta}')
    axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### General MA(q) Model

**Moving average of order q**:

\[
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}
\]

Using backshift operator:

\[
y_t = \mu + \Theta(B) \varepsilon_t
\]

Where \(\Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + ... + \theta_q B^q\)

**Invertibility condition**: All roots of \(\Theta(B) = 0\) must lie outside unit circle

### MA(2) Example

```python
# Simulate MA(2): y_t = e_t + 0.5*e_{t-1} + 0.3*e_{t-2}
np.random.seed(42)
ar_params = np.array([1])
ma_params = np.array([1, 0.5, 0.3])

y_ma2 = arma_generate_sample(ar_params, ma_params, nsample=500, scale=1)

# Fit MA(2) model
model_ma2 = ARIMA(y_ma2, order=(0, 0, 2))
results_ma2 = model_ma2.fit()

print("\nMA(2) Model Results:")
print(results_ma2.summary())
print(f"\nTrue parameters: θ₁=0.5, θ₂=0.3")
print(f"Estimated parameters: θ₁={results_ma2.params['ma.L1']:.4f}, θ₂={results_ma2.params['ma.L2']:.4f}")
```

## ARMA Models

### ARMA(p,q) Definition

**Combines AR and MA components**:

\[
y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}
\]

**Compact notation**:

\[
\Phi(B) y_t = c + \Theta(B) \varepsilon_t
\]

### Why Use ARMA?

**Parsimony**: ARMA(1,1) can represent patterns requiring high-order pure AR or MA

**Example**: 
- AR(∞) might be needed for certain patterns
- ARMA(1,1) achieves same with just 2 parameters

### ARMA(1,1) Example

```python
# Simulate ARMA(1,1): y_t = 0.7*y_{t-1} + e_t + 0.4*e_{t-1}
np.random.seed(42)
ar_params = np.array([1, -0.7])  # Note: statsmodels uses negative convention
ma_params = np.array([1, 0.4])

y_arma = arma_generate_sample(ar_params, ma_params, nsample=500, scale=1)

# Fit ARMA(1,1) model
model_arma = ARIMA(y_arma, order=(1, 0, 1))
results_arma = model_arma.fit()

print("\nARMA(1,1) Model Results:")
print(results_arma.summary())
print(f"\nTrue: φ₁=0.7, θ₁=0.4")
print(f"Estimated: φ₁={results_arma.params['ar.L1']:.4f}, θ₁={results_arma.params['ma.L1']:.4f}")
```

## ARIMA Models

### Adding Integration (I)

**Problem**: Many real series are non-stationary

**Solution**: Difference the series \(d\) times

**ARIMA(p,d,q)**: Apply ARMA(p,q) to differenced series

### First-Order Differencing

\[
\nabla y_t = y_t - y_{t-1}
\]

Using backshift: \(\nabla y_t = (1-B)y_t\)

### Second-Order Differencing

\[
\nabla^2 y_t = \nabla(\nabla y_t) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})
\]

Using backshift: \(\nabla^2 y_t = (1-B)^2 y_t\)

### General ARIMA(p,d,q) Model

**Step 1**: Difference \(d\) times to get \(w_t = \nabla^d y_t\)

**Step 2**: Apply ARMA(p,q) to \(w_t\):

\[
\Phi(B) w_t = c + \Theta(B) \varepsilon_t
\]

**Combined form**:

\[
\Phi(B) (1-B)^d y_t = c + \Theta(B) \varepsilon_t
\]

### Common ARIMA Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **ARIMA(0,1,0)** | Random walk | Stock prices |
| **ARIMA(0,1,1)** | Exponential smoothing | Demand forecasting |
| **ARIMA(1,1,0)** | Differenced AR(1) | Economic indicators |
| **ARIMA(0,2,2)** | Linear trend | Growth series |
| **ARIMA(1,1,1)** | General model | Many applications |

### ARIMA(0,1,1) - Exponential Smoothing

\[
\nabla y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}
\]

Equivalent to:

\[
y_t = y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}
\]

**Simple exponential smoothing** when \(\theta_1 = \alpha - 1\)

```python
# Simulate ARIMA(0,1,1) - Random walk with MA(1) error
np.random.seed(42)
n = 200
e = np.random.normal(0, 1, n)
y_arima011 = np.zeros(n)

theta = 0.5
for t in range(1, n):
    y_arima011[t] = y_arima011[t-1] + e[t] + theta * e[t-1]

plt.figure(figsize=(12, 5))
plt.plot(y_arima011)
plt.title('ARIMA(0,1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.show()

# Fit model
model = ARIMA(y_arima011, order=(0, 1, 1))
results = model.fit()
print(results.summary())
```

## Model Selection

### Information Criteria

**1. Akaike Information Criterion (AIC)**:

\[
\text{AIC} = -2\log(L) + 2k
\]

Where:
- \(L\): Maximum likelihood
- \(k\): Number of parameters (\(p + q + 1\))

**2. Bayesian Information Criterion (BIC)**:

\[
\text{BIC} = -2\log(L) + k \log(n)
\]

Where \(n\) is sample size

**Selection rule**: Choose model with **minimum** AIC or BIC

**BIC vs AIC**:
- BIC penalizes complexity more heavily
- BIC tends to select simpler models
- AIC better for prediction
- BIC better for true model selection

### Grid Search for Best ARIMA

```python
import itertools
import warnings
warnings.filterwarnings('ignore')

def find_best_arima(y, max_p=3, max_d=2, max_q=3):
    """
    Grid search for best ARIMA model
    """
    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = None
    best_order_bic = None
    
    results = []
    
    # Generate all combinations
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        if p == 0 and q == 0:
            continue  # Skip (0,d,0)
        
        try:
            model = ARIMA(y, order=(p, d, q))
            fitted = model.fit()
            
            results.append({
                'order': (p, d, q),
                'AIC': fitted.aic,
                'BIC': fitted.bic,
                'params': len(fitted.params)
            })
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order_aic = (p, d, q)
            
            if fitted.bic < best_bic:
                best_bic = fitted.bic
                best_order_bic = (p, d, q)
                
        except:
            continue
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('AIC').head(10)
    
    print("Top 10 Models by AIC:")
    print(df_results.to_string(index=False))
    print(f"\nBest model by AIC: ARIMA{best_order_aic}")
    print(f"Best model by BIC: ARIMA{best_order_bic}")
    
    return best_order_aic, best_order_bic

# Example usage
np.random.seed(42)
y_test = arma_generate_sample([1, -0.6], [1, 0.4], nsample=200, scale=1)

best_aic, best_bic = find_best_arima(y_test, max_p=3, max_d=1, max_q=3)
```

### Auto ARIMA

**Automated model selection** using stepwise algorithm:

```python
from pmdarima import auto_arima

# Automatic ARIMA model selection
auto_model = auto_arima(
    y_test,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # Auto-detect differencing
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic',
    trace=True
)

print("\nAuto ARIMA Results:")
print(auto_model.summary())
print(f"\nSelected model: ARIMA{auto_model.order}")
```

## Parameter Estimation

### Maximum Likelihood Estimation (MLE)

**Likelihood function** for ARIMA:

\[
L(\phi, \theta, \sigma^2 | y) = \prod_{t=1}^{n} f(y_t | y_{t-1}, ..., y_1; \phi, \theta, \sigma^2)
\]

**Objective**: Maximize \(L\) (or minimize \(-\log L\))

**Optimization**: Numerical methods (BFGS, Newton-Raphson)

### Standard Errors

**Asymptotic standard errors** from Hessian matrix:

\[
\text{SE}(\hat{\theta}) \approx \sqrt{\text{diag}(H^{-1})}
\]

Where \(H\) is the Hessian of \(-\log L\)

### Confidence Intervals

**95% CI for parameters**:

\[
\hat{\theta} \pm 1.96 \times \text{SE}(\hat{\theta})
\]

```python
# Fit model and extract parameter details
model = ARIMA(y_test, order=(1, 0, 1))
results = model.fit()

print("\nParameter Estimates:")
print(results.params)

print("\nStandard Errors:")
print(results.bse)

print("\n95% Confidence Intervals:")
print(results.conf_int())

print("\nt-statistics:")
print(results.tvalues)

print("\np-values:")
print(results.pvalues)
```

## Model Diagnostics

### Residual Analysis

**Good model**: Residuals should be white noise
- Mean ≈ 0
- Constant variance
- No autocorrelation
- Normally distributed

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual plot
residuals = results.resid
axes[0, 0].plot(residuals)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].grid(alpha=0.3)

# Histogram
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Histogram of Residuals')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

### Ljung-Box Test

**Tests for autocorrelation in residuals**:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test:")
print(lb_test)

if (lb_test['lb_pvalue'] > 0.05).all():
    print("\n✓ Residuals appear to be white noise")
else:
    print("\n✗ Significant autocorrelation in residuals")
```

### Normality Tests

```python
from scipy.stats import jarque_bera, shapiro

# Jarque-Bera test
jb_stat, jb_pvalue = jarque_bera(residuals)
print(f"\nJarque-Bera Test:")
print(f"  Statistic: {jb_stat:.4f}")
print(f"  p-value: {jb_pvalue:.4f}")

if jb_pvalue > 0.05:
    print("  ✓ Residuals appear normally distributed")
else:
    print("  ✗ Residuals may not be normally distributed")

# Shapiro-Wilk test
sw_stat, sw_pvalue = shapiro(residuals)
print(f"\nShapiro-Wilk Test:")
print(f"  Statistic: {sw_stat:.4f}")
print(f"  p-value: {sw_pvalue:.4f}")
```

## Summary

**Key Concepts**:

✅ **AR(p)**: Current value = linear combination of past \(p\) values + error
✅ **MA(q)**: Current value = linear combination of past \(q\) errors + error
✅ **ARMA(p,q)**: Combines both AR and MA components
✅ **ARIMA(p,d,q)**: ARMA on differenced series
✅ **Model selection**: Use AIC/BIC or auto_arima
✅ **Diagnostics**: Check residuals for white noise

**Workflow**:
1. Plot data → Visual inspection
2. Check stationarity → ADF test
3. Transform → Differencing if needed
4. Identify orders → ACF/PACF plots
5. Fit candidates → Compare AIC/BIC
6. Diagnostics → Residual analysis
7. Forecast → Generate predictions

## Next Section

Continue to [Seasonal ARIMA](04-seasonal-arima.md) to learn how to handle seasonal patterns in time series data.