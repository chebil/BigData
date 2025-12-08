# Seasonal ARIMA (SARIMA)

## Introduction

Many time series exhibit seasonal patterns that repeat at fixed intervals:
- **Monthly data**: 12-month cycles (retail sales, temperature)
- **Quarterly data**: 4-quarter cycles (GDP, earnings)
- **Daily data**: 7-day cycles (website traffic)
- **Hourly data**: 24-hour cycles (electricity demand)

**Seasonal ARIMA** extends ARIMA to handle these patterns explicitly.

## SARIMA Notation

**SARIMA(p,d,q)(P,D,Q)[s]**

**Non-seasonal components** (lowercase):
- **p**: Non-seasonal AR order
- **d**: Non-seasonal differencing
- **q**: Non-seasonal MA order

**Seasonal components** (uppercase):
- **P**: Seasonal AR order
- **D**: Seasonal differencing
- **Q**: Seasonal MA order
- **[s]**: Seasonal period

**Example**: SARIMA(1,1,1)(1,1,1)[12]
- Non-seasonal: AR(1), 1 difference, MA(1)
- Seasonal: SAR(1), 1 seasonal difference, SMA(1), period=12

## Seasonal Differencing

### First-Order Seasonal Difference

**At lag s**:

\[
\nabla_s y_t = y_t - y_{t-s}
\]

**Examples**:
- Monthly data (s=12): \(y_t - y_{t-12}\) (compare January to January)
- Quarterly (s=4): \(y_t - y_{t-4}\) (compare Q1 to Q1)
- Weekly (s=7): \(y_t - y_{t-7}\) (compare Monday to Monday)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate seasonal data
np.random.seed(42)
n = 144  # 12 years monthly
t = np.arange(n)

# Trend + Seasonal + Noise
trend = 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, n)

y = trend + seasonal + noise

# Apply seasonal differencing
y_diff_seasonal = np.diff(y, n=12)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y)
axes[0].set_title('Original Series with Seasonal Pattern')
axes[0].set_ylabel('Value')
axes[0].grid(alpha=0.3)

axes[1].plot(range(12, len(y)), y_diff_seasonal)
axes[1].set_title('After Seasonal Differencing (lag=12)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Combined Differencing

**Both regular and seasonal**:

\[
\nabla_s \nabla y_t = (1-B^s)(1-B)y_t
\]

Example for monthly data:
\[
(y_t - y_{t-12}) - (y_{t-1} - y_{t-13})
\]

```python
# First difference, then seasonal difference
y_diff1 = np.diff(y, n=1)
y_diff1_seasonal = np.diff(y_diff1, n=12)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(y)
axes[0].set_title('Original Series')
axes[0].grid(alpha=0.3)

axes[1].plot(y_diff1)
axes[1].set_title('After First Differencing')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)

axes[2].plot(range(13, len(y)), y_diff1_seasonal)
axes[2].set_title('After First + Seasonal Differencing')
axes[2].set_xlabel('Time')
axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## SARIMA Model Components

### Seasonal Autoregressive (SAR)

**SAR(P) at lag s**:

\[
y_t = \Phi_1 y_{t-s} + \Phi_2 y_{t-2s} + ... + \Phi_P y_{t-Ps} + \varepsilon_t
\]

**Example SAR(1)[12]** for monthly data:
\[
y_t = \Phi_1 y_{t-12} + \varepsilon_t
\]

January depends on previous January, February on previous February, etc.

### Seasonal Moving Average (SMA)

**SMA(Q) at lag s**:

\[
y_t = \varepsilon_t + \Theta_1 \varepsilon_{t-s} + \Theta_2 \varepsilon_{t-2s} + ... + \Theta_Q \varepsilon_{t-Qs}
\]

**Example SMA(1)[12]**:
\[
y_t = \varepsilon_t + \Theta_1 \varepsilon_{t-12}
\]

### Full SARIMA Model

**General SARIMA(p,d,q)(P,D,Q)[s]**:

\[
\Phi_P(B^s) \phi_p(B) \nabla^d \nabla_s^D y_t = c + \Theta_Q(B^s) \theta_q(B) \varepsilon_t
\]

Where:
- \(\phi_p(B) = 1 - \phi_1 B - ... - \phi_p B^p\): Non-seasonal AR
- \(\theta_q(B) = 1 + \theta_1 B + ... + \theta_q B^q\): Non-seasonal MA
- \(\Phi_P(B^s) = 1 - \Phi_1 B^s - ... - \Phi_P B^{Ps}\): Seasonal AR
- \(\Theta_Q(B^s) = 1 + \Theta_1 B^s + ... + \Theta_Q B^{Qs}\): Seasonal MA
- \(\nabla^d\): d non-seasonal differences
- \(\nabla_s^D\): D seasonal differences at lag s

## Identifying Seasonal Orders

### ACF and PACF Patterns

**Look for patterns at seasonal lags** (s, 2s, 3s, ...):

| Seasonal Component | ACF at lags s, 2s, 3s | PACF at lags s, 2s, 3s |
|-------------------|---------------------|---------------------|
| **SAR(P)** | Decay | Cutoff after Ps |
| **SMA(Q)** | Cutoff after Qs | Decay |
| **Both** | Decay | Decay |

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.datasets import get_rdataset

# Load airline passengers data
data = get_rdataset('AirPassengers').data
data['time'] = pd.date_range('1949-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

# Log transform + differences
ts_log = np.log(ts)
ts_diff = ts_log.diff().dropna()
ts_seasonal = ts_diff.diff(12).dropna()

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(ts_seasonal, lags=36, ax=axes[0])
axes[0].set_title('ACF: Spike at lag 12 (SMA)')
axes[0].axvline(x=12, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=24, color='r', linestyle='--', alpha=0.5)

plot_pacf(ts_seasonal, lags=36, ax=axes[1], method='ywm')
axes[1].set_title('PACF: Spike at lag 12 (SAR)')
axes[1].axvline(x=12, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(x=24, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("Interpretation:")
print("- Spike at lag 12 in both ACF and PACF")
print("- Suggests SAR(1) and/or SMA(1) at seasonal lag")
print("- Candidate: SARIMA(p,1,q)(1,1,1)[12]")
```

## Fitting SARIMA Models

### Using statsmodels

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Fit SARIMA(1,1,1)(1,1,1)[12]
model = SARIMAX(
    ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

print("\nSARIMA(1,1,1)(1,1,1)[12] Results:")
print(results.summary())

# Extract parameters
print("\nParameter Estimates:")
params = results.params
for param, value in params.items():
    print(f"  {param}: {value:.4f}")
```

### Common SARIMA Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **SARIMA(0,1,1)(0,1,1)[12]** | Airline model | Classic seasonal pattern |
| **SARIMA(1,1,1)(1,1,1)[12]** | General seasonal | Many applications |
| **SARIMA(0,1,1)(0,1,1)[7]** | Weekly pattern | Daily data |
| **SARIMA(1,0,0)(1,0,0)[4]** | Quarterly AR | Stationary seasonal |
| **SARIMA(0,1,2)(0,1,1)[12]** | MA with seasonal | Smooth series |

## Model Selection

### Grid Search for SARIMA

```python
import itertools

def sarima_grid_search(ts, max_order=2, seasonal_period=12):
    """
    Find best SARIMA model using AIC
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    
    # Generate combinations
    p = d = q = range(0, max_order + 1)
    P = D = Q = range(0, 2)  # Seasonal orders usually 0 or 1
    
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) 
                    for x in itertools.product(P, D, Q)]
    
    results = []
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(
                    ts,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted = model.fit(disp=False, maxiter=200)
                
                results.append({
                    'order': param,
                    'seasonal_order': param_seasonal,
                    'AIC': fitted.aic,
                    'BIC': fitted.bic
                })
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = param
                    best_seasonal_order = param_seasonal
                    
            except:
                continue
    
    # Display top 10
    df_results = pd.DataFrame(results).sort_values('AIC').head(10)
    print("\nTop 10 SARIMA Models by AIC:")
    print(df_results.to_string(index=False))
    
    print(f"\n\nBest Model:")
    print(f"  SARIMA{best_order}{best_seasonal_order}")
    print(f"  AIC: {best_aic:.2f}")
    
    return best_order, best_seasonal_order

# Run grid search
best_order, best_seasonal = sarima_grid_search(ts, max_order=2, seasonal_period=12)
```

### Auto SARIMA

```python
from pmdarima import auto_arima

# Automatic SARIMA selection
auto_model = auto_arima(
    ts,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    m=12,  # Seasonal period
    seasonal=True,
    d=None,  # Auto-detect
    D=None,  # Auto-detect seasonal
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\n" + "="*50)
print("Auto SARIMA Results")
print("="*50)
print(auto_model.summary())
print(f"\nSelected: SARIMA{auto_model.order}{auto_model.seasonal_order}")
```

## Model Diagnostics

### Residual Analysis

```python
# Fit best model
model = SARIMAX(
    ts,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

# Comprehensive diagnostics plot
fig = results.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.show()

# Manual residual checks
residuals = results.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals over time
axes[0,0].plot(residuals)
axes[0,0].axhline(y=0, color='r', linestyle='--')
axes[0,0].set_title('Residuals Over Time')
axes[0,0].set_ylabel('Residual')
axes[0,0].grid(alpha=0.3)

# 2. Histogram with normal curve
axes[0,1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
from scipy.stats import norm
axes[0,1].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
axes[0,1].set_title('Histogram of Residuals')
axes[0,1].legend()

# 3. ACF of residuals
plot_acf(residuals, lags=36, ax=axes[1,0])
axes[1,0].set_title('ACF of Residuals')

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

### Statistical Tests

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, shapiro

print("\n" + "="*50)
print("Residual Diagnostics")
print("="*50)

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test (No Autocorrelation):")
print(lb_test)
if (lb_test['lb_pvalue'] > 0.05).all():
    print("  ✓ Residuals show no significant autocorrelation")
else:
    print("  ✗ Significant autocorrelation detected")

# Jarque-Bera test for normality
jb_stat, jb_pvalue = jarque_bera(residuals)
print(f"\nJarque-Bera Test (Normality):")
print(f"  Statistic: {jb_stat:.4f}")
print(f"  p-value: {jb_pvalue:.4f}")
if jb_pvalue > 0.05:
    print("  ✓ Residuals appear normally distributed")
else:
    print("  ✗ Residuals may not be normal")

# Shapiro-Wilk
sw_stat, sw_pvalue = shapiro(residuals)
print(f"\nShapiro-Wilk Test (Normality):")
print(f"  Statistic: {sw_stat:.4f}")
print(f"  p-value: {sw_pvalue:.4f}")

# Heteroscedasticity
from statsmodels.stats.diagnostic import het_arch
het_stat, het_pvalue, _, _ = het_arch(residuals, nlags=12)
print(f"\nARCH Test (Homoscedasticity):")
print(f"  Statistic: {het_stat:.4f}")
print(f"  p-value: {het_pvalue:.4f}")
if het_pvalue > 0.05:
    print("  ✓ No significant heteroscedasticity")
else:
    print("  ✗ Heteroscedasticity detected")
```

## Case Study: Airline Passengers

### Complete Analysis

```python
# Load data
data = get_rdataset('AirPassengers').data
data['time'] = pd.date_range('1949-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

print("="*60)
print("SARIMA CASE STUDY: Airline Passengers (1949-1960)")
print("="*60)

# Step 1: EDA
print("\nStep 1: Exploratory Data Analysis")
print(f"  Data points: {len(ts)}")
print(f"  Date range: {ts.index[0]} to {ts.index[-1]}")
print(f"  Min: {ts.min():.0f}, Max: {ts.max():.0f}")
print(f"  Mean: {ts.mean():.2f}, Std: {ts.std():.2f}")

# Step 2: Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(ts, model='multiplicative', period=12)
fig = decomp.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.show()

print("\nStep 2: Seasonal Decomposition")
print("  - Clear upward trend")
print("  - Strong seasonal pattern (period=12)")
print("  - Increasing variance (multiplicative)")

# Step 3: Stationarity tests
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"\n  {name}:")
    print(f"    ADF Statistic: {result[0]:.4f}")
    print(f"    p-value: {result[1]:.4f}")
    return result[1] <= 0.05

print("\nStep 3: Stationarity Tests")
is_stationary = test_stationarity(ts, "Original")
is_stationary = test_stationarity(np.log(ts), "Log-transformed")
is_stationary = test_stationarity(np.log(ts).diff().dropna(), "First difference")
is_stationary = test_stationarity(
    np.log(ts).diff().diff(12).dropna(), 
    "First + Seasonal difference"
)

# Step 4: Model selection
print("\nStep 4: Model Selection")
print("  Trying multiple SARIMA specifications...")

candidates = [
    ((0,1,1), (0,1,1,12)),  # Airline model
    ((1,1,1), (1,1,1,12)),  # General
    ((0,1,2), (0,1,1,12)),  # MA focus
    ((1,1,0), (1,1,0,12)),  # AR focus
]

for order, seasonal_order in candidates:
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)
        print(f"\n  SARIMA{order}{seasonal_order}:")
        print(f"    AIC: {fitted.aic:.2f}")
        print(f"    BIC: {fitted.bic:.2f}")
    except:
        print(f"\n  SARIMA{order}{seasonal_order}: Failed to converge")

print("\n  => Best model: SARIMA(0,1,1)(0,1,1)[12] (Airline Model)")

# Step 5: Fit final model
print("\nStep 5: Final Model")
final_model = SARIMAX(ts, order=(0,1,1), seasonal_order=(0,1,1,12))
final_results = final_model.fit(disp=False)

print(final_results.summary())

# Step 6: Forecast
print("\nStep 6: Forecasting")
forecast_steps = 24
forecast = final_results.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Historical
ax.plot(ts.index, ts, label='Historical', linewidth=2)

# Forecast
forecast_index = pd.date_range(ts.index[-1], periods=forecast_steps+1, freq='M')[1:]
ax.plot(forecast_index, forecast_df['mean'], 
        label='Forecast', color='red', linewidth=2)

# Confidence interval
ax.fill_between(forecast_index,
                forecast_df['mean_ci_lower'],
                forecast_df['mean_ci_upper'],
                color='red', alpha=0.2, label='95% CI')

ax.set_title('SARIMA Forecast: Airline Passengers', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Passengers (thousands)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nForecast for next {forecast_steps} months:")
print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].head(12))
```

## Best Practices

### DO's ✅

1. **Always plot the data first**: Visual inspection is crucial
2. **Check for seasonality**: Look at 2-3 full seasonal cycles
3. **Transform for variance stabilization**: Log transform if variance increases
4. **Difference for stationarity**: Both regular and seasonal if needed
5. **Examine ACF/PACF at seasonal lags**: Multiples of s
6. **Try multiple models**: Don't settle on first candidate
7. **Validate residuals**: Check white noise assumptions
8. **Use out-of-sample validation**: Don't rely on in-sample metrics alone

### DON'Ts ❌

1. **Don't ignore seasonality**: It won't go away
2. **Don't over-difference**: Check stationarity after each difference
3. **Don't use high seasonal orders**: P, D, Q rarely exceed 1
4. **Don't forget business context**: Model should make sense
5. **Don't extrapolate too far**: Uncertainty grows rapidly
6. **Don't ignore diagnostic warnings**: They indicate problems
7. **Don't fit to noise**: Simpler is often better

## Summary

**Key Concepts**:

✅ **SARIMA extends ARIMA** for seasonal patterns
✅ **Seasonal period [s]** must match data frequency
✅ **Seasonal components** (P,D,Q) usually simple (0 or 1)
✅ **ACF/PACF at seasonal lags** guide model selection
✅ **Model diagnostics** essential for validation
✅ **Grid search or auto_arima** for systematic selection

**Typical Workflow**:
1. Visual inspection → Identify seasonality
2. Decomposition → Understand components
3. Stationarity tests → ADF on original and transformed
4. Transformations → Log, differences
5. ACF/PACF → Identify orders
6. Model fitting → Try multiple candidates
7. Diagnostics → Residual analysis
8. Forecast → Generate predictions
9. Validation → Out-of-sample testing

## Next Section

Continue to [Forecasting](06-forecasting.md) to learn how to generate predictions and evaluate forecast accuracy.