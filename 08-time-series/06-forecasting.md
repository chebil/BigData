# Time Series Forecasting

## Introduction

Forecasting is the primary goal of time series analysis. After building and validating an ARIMA/SARIMA model, we generate predictions for future values with associated uncertainty (confidence intervals).

**Key Questions**:
- How far ahead can we forecast reliably?
- How do we quantify forecast uncertainty?
- How do we validate forecast accuracy?
- When should we update the model?

## Point Forecasts

### h-Step Ahead Forecast

**Forecast at time \(t\) for \(t+h\)**:

\[
\hat{y}_{t+h|t} = E[y_{t+h} | y_1, y_2, ..., y_t]
\]

**Notation**:
- \(\hat{y}_{t+h|t}\): Forecast made at time \(t\) for \(t+h\)
- Based on information up to time \(t\)

### Example: AR(1) Forecasting

**Model**: \(y_t = \phi y_{t-1} + \varepsilon_t\)

**1-step ahead** (\(h=1\)):
\[
\hat{y}_{t+1|t} = \phi y_t
\]

**2-step ahead** (\(h=2\)):
\[
\hat{y}_{t+2|t} = \phi \hat{y}_{t+1|t} = \phi^2 y_t
\]

**General h-step**:
\[
\hat{y}_{t+h|t} = \phi^h y_t
\]

**Key insight**: For \(|\phi| < 1\), forecasts converge to mean as \(h \to \infty\)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate AR(1) and forecast
np.random.seed(42)
phi = 0.8
n = 100
h_max = 20

# Generate data
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + np.random.normal(0, 1)

# Generate forecasts
forecasts = np.zeros(h_max)
forecasts[0] = phi * y[-1]
for h in range(1, h_max):
    forecasts[h] = phi * forecasts[h-1]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(n), y, label='Historical', linewidth=2)
ax.plot(range(n, n+h_max), forecasts, 
        label='Forecast', color='red', linewidth=2, linestyle='--')
ax.axvline(x=n-1, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title(f'AR(1) Forecasting (φ={phi})')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print("Forecast values converge to mean (0):")
for h in [1, 5, 10, 20]:
    print(f"  h={h:2d}: {forecasts[h-1]:.4f}")
```

## Forecast Intervals

### Prediction Uncertainty

**Forecast error**:
\[
e_{t+h|t} = y_{t+h} - \hat{y}_{t+h|t}
\]

**Forecast variance** (for AR(1)):
\[
\text{Var}(e_{t+h|t}) = \sigma^2 \frac{1 - \phi^{2h}}{1 - \phi^2}
\]

**Grows with horizon**: Uncertainty increases as we forecast further

### Confidence Intervals

**95% prediction interval**:

\[
\hat{y}_{t+h|t} \pm 1.96 \times \sqrt{\text{Var}(e_{t+h|t})}
\]

**Properties**:
- Width increases with \(h\)
- Based on assumption of normally distributed errors
- Contains true value ~95% of the time

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Fit AR(1) model
model = ARIMA(y, order=(1,0,0))
results = model.fit()

# Generate forecast with confidence intervals
forecast = results.get_forecast(steps=h_max)
forecast_df = forecast.summary_frame()

# Plot with confidence intervals
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(range(n), y, label='Historical', linewidth=2)
ax.plot(range(n, n+h_max), forecast_df['mean'], 
        label='Forecast', color='red', linewidth=2)
ax.fill_between(range(n, n+h_max),
                forecast_df['mean_ci_lower'],
                forecast_df['mean_ci_upper'],
                color='red', alpha=0.2, label='95% CI')

ax.axvline(x=n-1, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('AR(1) Forecast with 95% Confidence Intervals')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print("\nForecast with 95% Confidence Intervals:")
print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].head(10))
```

## ARIMA/SARIMA Forecasting

### General Procedure

**For ARIMA(p,d,q)**:

1. **Generate forecasts** for differenced series
2. **Invert differences** to get original scale
3. **Compute prediction intervals** accounting for differencing

```python
from statsmodels.datasets import get_rdataset

# Load airline data
data = get_rdataset('AirPassengers').data
data['time'] = pd.date_range('1949-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

# Split into train/test
train = ts[:'1958']
test = ts['1959':'1960']

print(f"Training: {len(train)} observations")
print(f"Testing: {len(test)} observations")

# Fit SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

print("\nModel Summary:")
print(results.summary())

# Forecast
forecast_steps = len(test)
forecast = results.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(train.index, train, label='Train', linewidth=2)
ax.plot(test.index, test, label='Test (Actual)', linewidth=2, color='green')
ax.plot(test.index, forecast_df['mean'], 
        label='Forecast', linewidth=2, color='red', linestyle='--')
ax.fill_between(test.index,
                forecast_df['mean_ci_lower'],
                forecast_df['mean_ci_upper'],
                color='red', alpha=0.2, label='95% CI')

ax.set_title('SARIMA Forecast: Airline Passengers')
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Forecast Evaluation

### Error Metrics

**1. Mean Absolute Error (MAE)**:
\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

**2. Root Mean Squared Error (RMSE)**:
\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

**3. Mean Absolute Percentage Error (MAPE)**:
\[
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

**4. Symmetric MAPE (SMAPE)**:
\[
\text{SMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
\]

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted):
    """
    Calculate forecast accuracy metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(predicted - actual) / 
                    (np.abs(actual) + np.abs(predicted))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}

# Calculate metrics
metrics = calculate_metrics(test.values, forecast_df['mean'].values)

print("\n" + "="*50)
print("Forecast Accuracy Metrics")
print("="*50)
for metric, value in metrics.items():
    print(f"{metric:10s}: {value:8.2f}")

# Mean Forecast Error (bias)
mfe = np.mean(test.values - forecast_df['mean'].values)
print(f"\nMean Forecast Error (Bias): {mfe:.2f}")
if abs(mfe) < 5:
    print("  ✓ Low bias - forecasts are unbiased")
else:
    print("  ✗ Significant bias detected")
```

### Coverage Probability

**Proportion of actual values within confidence intervals**:

```python
# Check if actual values fall within 95% CI
within_ci = ((test.values >= forecast_df['mean_ci_lower'].values) & 
             (test.values <= forecast_df['mean_ci_upper'].values))

coverage = within_ci.mean() * 100

print(f"\nCoverage Probability:")
print(f"  {coverage:.1f}% of actual values within 95% CI")
print(f"  Expected: 95%")

if coverage >= 90:
    print("  ✓ Good coverage")
else:
    print("  ✗ Poor coverage - intervals may be too narrow")
```

## Rolling Window Validation

### Walk-Forward Validation

**Procedure**:
1. Train on initial window
2. Forecast next step
3. Add actual value to training set
4. Retrain and repeat

```python
def rolling_forecast(ts, train_size, horizon=1):
    """
    Perform rolling window forecast
    """
    n = len(ts)
    predictions = []
    actuals = []
    
    for i in range(train_size, n - horizon + 1):
        # Train data
        train = ts[:i]
        
        # Fit model
        model = SARIMAX(
            train,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12)
        )
        results = model.fit(disp=False)
        
        # Forecast
        forecast = results.get_forecast(steps=horizon)
        pred = forecast.predicted_mean.iloc[-1]
        
        predictions.append(pred)
        actuals.append(ts.iloc[i + horizon - 1])
        
        if (i - train_size) % 12 == 0:
            print(f"  Progress: {i - train_size + 1} forecasts")
    
    return np.array(actuals), np.array(predictions)

print("\nPerforming rolling forecast validation...")
train_size = 108  # 9 years
actuals, predictions = rolling_forecast(ts, train_size, horizon=1)

# Calculate metrics
roll_metrics = calculate_metrics(actuals, predictions)

print("\n" + "="*50)
print("Rolling Forecast Metrics (1-step ahead)")
print("="*50)
for metric, value in roll_metrics.items():
    print(f"{metric:10s}: {value:8.2f}")

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(actuals, label='Actual', linewidth=2)
ax.plot(predictions, label='Forecast', linewidth=2, alpha=0.7)
ax.set_title('Rolling Forecast Validation')
ax.set_xlabel('Forecast Step')
ax.set_ylabel('Passengers')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

## Multi-Step Forecasting

### Strategies

**1. Direct Multi-Step**:
- Build separate model for each horizon
- Pros: Tailored to each horizon
- Cons: Many models to maintain

**2. Recursive (Iterative)**:
- Use 1-step forecast as input for next step
- Pros: Single model
- Cons: Errors accumulate

**3. DirRec (Hybrid)**:
- Combines both approaches

```python
# Compare 1-step vs multi-step accuracy
horizons = [1, 3, 6, 12]
results_by_horizon = []

for h in horizons:
    model = SARIMAX(
        train,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12)
    )
    fitted = model.fit(disp=False)
    
    forecast = fitted.get_forecast(steps=h)
    pred = forecast.predicted_mean.iloc[-1]
    actual = test.iloc[h-1]
    error = abs(actual - pred)
    
    results_by_horizon.append({
        'Horizon': h,
        'Forecast': pred,
        'Actual': actual,
        'Error': error,
        'MAPE': 100 * error / actual
    })

df_horizons = pd.DataFrame(results_by_horizon)
print("\nAccuracy by Forecast Horizon:")
print(df_horizons.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_horizons['Horizon'], df_horizons['MAPE'], 
        marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Forecast Horizon (months)')
ax.set_ylabel('MAPE (%)')
ax.set_title('Forecast Accuracy vs Horizon')
ax.grid(alpha=0.3)
plt.show()
```

## Forecast Updating

### When to Update

**Triggers for model update**:
1. **Regular schedule**: Weekly, monthly, quarterly
2. **Performance degradation**: MAPE exceeds threshold
3. **Structural breaks**: Major events, policy changes
4. **New data availability**: Additional observations

### Online Updating

```python
def update_forecast(model_results, new_data, forecast_horizon):
    """
    Update forecast with new observations
    """
    # Extend the model with new data
    updated = model_results.append(new_data, refit=True)
    
    # Generate new forecast
    forecast = updated.get_forecast(steps=forecast_horizon)
    
    return updated, forecast

# Example: Update monthly
print("\nForecast Update Example:")
print("-" * 50)

# Initial model
initial_model = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,12))
initial_results = initial_model.fit(disp=False)

print(f"Initial training end: {train.index[-1]}")

# Add one month and update
new_observation = test.iloc[:1]
print(f"New observation: {new_observation.index[0]} = {new_observation.values[0]:.0f}")

updated_results, new_forecast = update_forecast(
    initial_results, 
    new_observation, 
    forecast_horizon=12
)

print(f"Updated model training end: {train.index[-1] + pd.DateOffset(months=1)}")
print(f"Next 12-month forecast generated")
```

## Practical Guidelines

### Forecast Horizon Selection

**Rule of thumb**: Reliable forecasts typically extend to:
- **Short-term**: 1-3 periods ahead (high accuracy)
- **Medium-term**: 4-12 periods ahead (moderate accuracy)
- **Long-term**: >12 periods ahead (low accuracy, high uncertainty)

### Best Practices

✅ **DO**:
1. Use multiple models and ensemble
2. Always include confidence intervals
3. Validate out-of-sample
4. Update models regularly
5. Monitor forecast errors
6. Document assumptions
7. Consider business constraints
8. Communicate uncertainty

❌ **DON'T**:
1. Over-rely on point forecasts
2. Extrapolate too far ahead
3. Ignore structural breaks
4. Use only in-sample fit
5. Forget to check residuals
6. Assume stationarity without testing
7. Ignore domain knowledge
8. Hide forecast uncertainty

## Summary

**Key Concepts**:

✅ **Point forecasts** are conditional expectations
✅ **Prediction intervals** quantify uncertainty
✅ **Uncertainty grows** with forecast horizon
✅ **Multiple metrics** assess different aspects
✅ **Rolling validation** mimics real-world usage
✅ **Regular updates** maintain accuracy
✅ **Communication** of uncertainty is crucial

**Typical Workflow**:
1. Fit model on training data
2. Generate forecasts with intervals
3. Evaluate on test data
4. Calculate accuracy metrics
5. Check coverage probability
6. Perform rolling validation
7. Update model periodically
8. Monitor ongoing performance

**Forecast Quality Indicators**:
- MAE/RMSE within acceptable range
- Low bias (MFE ≈ 0)
- Coverage probability ≈ 95%
- Residuals are white noise
- Consistent performance over time

## Next Section

Continue to [Advanced Methods](07-advanced-methods.md) to explore Prophet, exponential smoothing, and machine learning approaches for time series forecasting.

---

**Chapter 8 Complete!** You now understand:
- Stationarity and transformations
- ACF/PACF for model identification
- ARIMA and SARIMA models
- Model selection and diagnostics
- Forecasting with confidence intervals
- Forecast evaluation and validation

**Next Chapter**: [Chapter 9: Text Analytics](../09-text-analytics/index.md)