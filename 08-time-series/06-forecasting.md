# Forecasting

## Introduction

Forecasting is the primary goal of time series analysis. After fitting an ARIMA model, we use it to predict future values with associated uncertainty (confidence intervals).

**Key concepts**:
- **Point forecast**: Single predicted value
- **Interval forecast**: Range of plausible values
- **Forecast horizon**: How far ahead to predict
- **Forecast error**: Difference between actual and predicted

## Point Forecasts

### One-Step-Ahead Forecast

**Simplest case**: Predict next value

For ARIMA(p,d,q), the forecast is:

\[
\hat{y}_{t+1} = \phi_1 y_t + \phi_2 y_{t-1} + ... + \phi_p y_{t-p+1} + \theta_1 \epsilon_t + ... + \theta_q \epsilon_{t-q+1}
\]

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

# Fit model
model = ARIMA(ts, order=(1, 1, 1))
results = model.fit()

# One-step forecast
forecast_1step = results.forecast(steps=1)
print(f"Next value forecast: {forecast_1step.iloc[0]:.2f}")
```

### Multi-Step Forecasts

**Predict h steps ahead**

```python
# Forecast next 12 periods
forecast = results.forecast(steps=12)
print(forecast)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Historical Data')
plt.plot(forecast, label='Forecast', color='red', marker='o')
plt.legend()
plt.title('ARIMA Forecast')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Forecast Uncertainty

**Forecast error variance increases with horizon**:

\[
\text{Var}(\hat{y}_{t+h}) = \sigma^2 \left(1 + \psi_1^2 + \psi_2^2 + ... + \psi_{h-1}^2\right)
\]

Where \(\psi_i\) are MA representation coefficients.

**Key insight**: Long-term forecasts less reliable

## Confidence Intervals

### Construction

**95% Confidence Interval**:

\[
\hat{y}_{t+h} \pm 1.96 \times SE(\hat{y}_{t+h})
\]

Where \(SE\) is the standard error of forecast.

```python
# Forecast with confidence intervals
forecast_obj = results.get_forecast(steps=12)
forecast_df = forecast_obj.summary_frame()

print(forecast_df[['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']])
```

### Visualization

```python
plt.figure(figsize=(12, 6))

# Historical data
plt.plot(ts, label='Historical', color='blue')

# Forecast
plt.plot(forecast_df['mean'], label='Forecast', color='red')

# Confidence interval
plt.fill_between(
    forecast_df.index,
    forecast_df['mean_ci_lower'],
    forecast_df['mean_ci_upper'],
    color='red',
    alpha=0.2,
    label='95% Confidence Interval'
)

plt.legend()
plt.title('ARIMA Forecast with Confidence Intervals')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Prediction Intervals

**Different confidence levels**:

```python
# 80%, 95%, 99% intervals
for alpha in [0.20, 0.05, 0.01]:
    forecast_obj = results.get_forecast(steps=12)
    forecast_df = forecast_obj.summary_frame(alpha=alpha)
    
    confidence_level = int((1 - alpha) * 100)
    print(f"\n{confidence_level}% Confidence Interval:")
    print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].head())
```

## Forecast Evaluation

### Train-Test Split

```python
# Split data: 80% train, 20% test
train_size = int(0.8 * len(ts))
train = ts[:train_size]
test = ts[train_size:]

print(f"Train: {len(train)} observations")
print(f"Test: {len(test)} observations")

# Fit on train
model = ARIMA(train, order=(1, 1, 1))
results = model.fit()

# Forecast test period
forecast = results.forecast(steps=len(test))

# Align indices
forecast.index = test.index
```

### Forecast Errors

```python
# Calculate errors
errors = test - forecast

print(f"Mean error: {errors.mean():.4f}")
print(f"Std of errors: {errors.std():.4f}")

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual', marker='o')
plt.plot(forecast, label='Forecast', marker='x')
plt.legend()
plt.title('Actual vs Forecast')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Performance Metrics

#### Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
\]

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test, forecast)
print(f"MAE: {mae:.4f}")
```

**Interpretation**: Average absolute forecast error in original units

#### Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.4f}")
```

**Interpretation**: Penalizes large errors more than MAE

#### Mean Absolute Percentage Error (MAPE)

\[
MAPE = \frac{100}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

```python
def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

mape_value = mape(test, forecast)
print(f"MAPE: {mape_value:.2f}%")
```

**Interpretation**: Percentage error (scale-independent)

**Problem**: Undefined when actual = 0

#### Symmetric MAPE (SMAPE)

\[
SMAPE = \frac{100}{n}\sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
\]

```python
def smape(actual, forecast):
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    return np.mean(np.abs(actual - forecast) / denominator) * 100

smape_value = smape(test, forecast)
print(f"SMAPE: {smape_value:.2f}%")
```

**Advantage**: Symmetric, bounded [0, 200]

#### Mean Absolute Scaled Error (MASE)

\[
MASE = \frac{MAE}{\frac{1}{n-1}\sum_{i=2}^n |y_i - y_{i-1}|}
\]

```python
def mase(actual, forecast, train):
    mae = mean_absolute_error(actual, forecast)
    naive_mae = np.mean(np.abs(np.diff(train)))
    return mae / naive_mae

mase_value = mase(test, forecast, train)
print(f"MASE: {mase_value:.4f}")
```

**Interpretation**:
- < 1: Better than naïve forecast
- = 1: Same as naïve
- > 1: Worse than naïve

### Coverage Probability

**Proportion of actuals within confidence intervals**:

```python
# Get forecast with intervals
forecast_obj = results.get_forecast(steps=len(test))
forecast_df = forecast_obj.summary_frame()
forecast_df.index = test.index

# Check coverage
within_interval = (
    (test >= forecast_df['mean_ci_lower']) & 
    (test <= forecast_df['mean_ci_upper'])
)

coverage = within_interval.mean()
print(f"95% CI Coverage: {coverage:.1%}")
print(f"Expected: 95%, Actual: {coverage:.1%}")
```

**Ideal**: Coverage ≈ confidence level

## Rolling Forecast

### Walk-Forward Validation

**More realistic evaluation**:

```python
def rolling_forecast(train, test, order):
    """
    Perform rolling forecast (walk-forward)
    """
    forecasts = []
    history = list(train)
    
    for actual in test:
        # Fit model on current history
        model = ARIMA(history, order=order)
        results = model.fit()
        
        # Forecast one step
        forecast = results.forecast(steps=1).iloc[0]
        forecasts.append(forecast)
        
        # Add actual to history
        history.append(actual)
    
    return pd.Series(forecasts, index=test.index)

# Perform rolling forecast
rolling_fc = rolling_forecast(train, test, order=(1, 1, 1))

# Evaluate
rolling_mae = mean_absolute_error(test, rolling_fc)
print(f"Rolling Forecast MAE: {rolling_mae:.4f}")

# Compare to static forecast
static_fc = results.forecast(steps=len(test))
static_fc.index = test.index
static_mae = mean_absolute_error(test, static_fc)
print(f"Static Forecast MAE: {static_mae:.4f}")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual', marker='o')
plt.plot(static_fc, label='Static Forecast', marker='x')
plt.plot(rolling_fc, label='Rolling Forecast', marker='s')
plt.legend()
plt.title('Static vs Rolling Forecast')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Rolling forecast** usually more accurate (model updated with new data)

## Complete Example: Retail Sales Forecasting

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic retail sales data
np.random.seed(42)
months = pd.date_range('2015-01', periods=120, freq='M')
t = np.arange(120)

# Components
trend = 100 + 2 * t  # Growing trend
seasonal = 20 * np.sin(2 * np.pi * t / 12)  # Yearly seasonality
noise = np.random.normal(0, 5, 120)
sales = trend + seasonal + noise

ts = pd.Series(sales, index=months, name='Sales')

print("=" * 60)
print("RETAIL SALES FORECASTING CASE STUDY")
print("=" * 60)

# Step 1: Visualize
print("\n1. Data Visualization")
plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('Monthly Retail Sales (2015-2024)', fontsize=14, fontweight='bold')
plt.ylabel('Sales ($1000s)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 2: Train-test split
print("\n2. Train-Test Split")
train_size = 96  # 8 years train, 2 years test
train = ts[:train_size]
test = ts[train_size:]
print(f"   Training: {train.index[0]} to {train.index[-1]} ({len(train)} months)")
print(f"   Testing:  {test.index[0]} to {test.index[-1]} ({len(test)} months)")

# Step 3: Model identification
print("\n3. Model Identification")

# Check stationarity
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(train)
print(f"   ADF test p-value: {adf_result[1]:.4f}")
if adf_result[1] > 0.05:
    print("   => Series is non-stationary, will difference")

# Difference
train_diff = train.diff().dropna()
adf_result = adfuller(train_diff)
print(f"   After differencing, p-value: {adf_result[1]:.4f}")

if adf_result[1] <= 0.05:
    print("   => Series is now stationary")

# ACF/PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(train_diff, lags=20, ax=axes[0])
plot_pacf(train_diff, lags=20, ax=axes[1])
plt.suptitle('ACF and PACF of Differenced Series', fontweight='bold')
plt.tight_layout()
plt.show()

# Step 4: Model selection
print("\n4. Model Selection (Grid Search)")

models_to_test = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2)
]

results_list = []
for order in models_to_test:
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        results_list.append({
            'Order': order,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
            'Log-Likelihood': fitted.llf
        })
    except:
        continue

results_df = pd.DataFrame(results_list).sort_values('AIC')
print("\n   Model Comparison (sorted by AIC):")
print(results_df.to_string(index=False))

best_order = results_df.iloc[0]['Order']
print(f"\n   ✓ Best model: ARIMA{best_order}")

# Step 5: Fit best model
print("\n5. Fitting Best Model")
model = ARIMA(train, order=best_order)
results = model.fit()
print(results.summary())

# Step 6: Diagnostics
print("\n6. Model Diagnostics")
results.plot_diagnostics(figsize=(12, 8))
plt.suptitle(f'Diagnostic Plots for ARIMA{best_order}', fontweight='bold')
plt.tight_layout()
plt.show()

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(results.resid, lags=10)
print(f"   Ljung-Box test (p-value): {lb_test['lb_pvalue'].iloc[-1]:.4f}")
if lb_test['lb_pvalue'].iloc[-1] > 0.05:
    print("   ✓ Residuals are white noise")

# Step 7: Forecast
print("\n7. Forecasting")
forecast_obj = results.get_forecast(steps=len(test))
forecast_df = forecast_obj.summary_frame()
forecast_df.index = test.index

# Plot forecast
plt.figure(figsize=(14, 7))

# Historical data
plt.plot(train, label='Training Data', color='blue', linewidth=2)
plt.plot(test, label='Actual Test Data', color='green', linewidth=2, marker='o')

# Forecast
plt.plot(forecast_df['mean'], label='Forecast', color='red', linewidth=2, marker='s')

# Confidence interval
plt.fill_between(
    forecast_df.index,
    forecast_df['mean_ci_lower'],
    forecast_df['mean_ci_upper'],
    color='red',
    alpha=0.2,
    label='95% Confidence Interval'
)

plt.axvline(train.index[-1], color='black', linestyle='--', alpha=0.5)
plt.text(train.index[-1], plt.ylim()[1]*0.95, ' Train/Test Split', 
         rotation=90, verticalalignment='top')

plt.title(f'ARIMA{best_order} Forecast vs Actual', fontsize=14, fontweight='bold')
plt.ylabel('Sales ($1000s)')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 8: Evaluate
print("\n8. Forecast Evaluation")
forecast_values = forecast_df['mean']

mae = mean_absolute_error(test, forecast_values)
rmse = np.sqrt(mean_squared_error(test, forecast_values))
mape_val = np.mean(np.abs((test - forecast_values) / test)) * 100

print(f"   MAE:  ${mae:.2f}k")
print(f"   RMSE: ${rmse:.2f}k")
print(f"   MAPE: {mape_val:.2f}%")

# Coverage
within_ci = (
    (test >= forecast_df['mean_ci_lower']) & 
    (test <= forecast_df['mean_ci_upper'])
)
coverage = within_ci.mean()
print(f"   95% CI Coverage: {coverage:.1%}")

# Error distribution
errors = test - forecast_values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(errors, marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.title('Forecast Errors Over Time')
plt.ylabel('Error ($1000s)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(errors, bins=15, edgecolor='black')
plt.axvline(0, color='r', linestyle='--', linewidth=2)
plt.title('Distribution of Forecast Errors')
plt.xlabel('Error ($1000s)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("FORECAST SUMMARY")
print("=" * 60)
print(f"Model: ARIMA{best_order}")
print(f"Forecast Horizon: {len(test)} months")
print(f"Mean Absolute Error: ${mae:.2f}k")
print(f"Accuracy (100% - MAPE): {100 - mape_val:.2f}%")
print("\nConclusion: Model provides reliable forecasts for retail")
print("sales planning with ~{:.0f}% accuracy.".format(100 - mape_val))
print("=" * 60)
```

## Best Practices

### Forecast Horizon

✅ **Short-term (1-3 steps)**: Usually reliable
✅ **Medium-term (4-12 steps)**: Use with caution
❌ **Long-term (>12 steps)**: High uncertainty, often revert to mean

### Model Updating

**Retrain regularly**:
- Weekly for daily data
- Monthly for weekly data
- Quarterly for monthly data

### Ensemble Forecasting

**Combine multiple models**:

```python
# Fit multiple models
model1 = ARIMA(train, order=(1, 1, 1)).fit()
model2 = ARIMA(train, order=(2, 1, 2)).fit()

# Forecast
f1 = model1.forecast(steps=len(test))
f2 = model2.forecast(steps=len(test))

# Simple average
ensemble = (f1 + f2) / 2
ensemble.index = test.index

# Evaluate
mae_ensemble = mean_absolute_error(test, ensemble)
print(f"Ensemble MAE: {mae_ensemble:.4f}")
```

## Summary

**Forecasting Checklist**:

1. ✅ **Split data** into train/test
2. ✅ **Fit model** on training data only
3. ✅ **Generate forecasts** for test period
4. ✅ **Calculate metrics** (MAE, RMSE, MAPE)
5. ✅ **Check coverage** of confidence intervals
6. ✅ **Visualize** actual vs forecast
7. ✅ **Consider rolling forecast** for realistic evaluation
8. ✅ **Document assumptions** and limitations

**Key Takeaways**:

- **Uncertainty increases** with forecast horizon
- **Confidence intervals** are essential
- **Multiple metrics** provide complete picture
- **Out-of-sample testing** prevents overfitting
- **Regular retraining** maintains accuracy
- **Business context** matters more than statistical perfection

## Next Steps

This completes Chapter 8 on Time Series Analysis! You now understand:
- Stationarity and transformations
- ACF/PACF interpretation
- ARIMA modeling
- Forecasting with confidence intervals
- Evaluation metrics

**Continue to**: [Chapter 9: Text Analytics](../09-text-analytics/index.md) for natural language processing and text mining.