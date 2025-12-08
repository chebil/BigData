# ARIMA and Forecasting

## Learning Objectives

- Understand ARIMA model components
- Build and tune ARIMA models
- Apply SARIMA for seasonal data
- Use Prophet for robust forecasting
- Evaluate forecast accuracy
- Implement production forecasting pipelines

## ARIMA Components

**ARIMA(p, d, q)**: AutoRegressive Integrated Moving Average

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

print("""
ARIMA MODEL COMPONENTS:

AR(p) - AutoRegressive:
  - Uses past values to predict future
  - y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ... + φₚ·y(t-p) + ε(t)
  - p = number of lag observations
  - PACF cuts off after lag p

I(d) - Integrated:
  - Number of differencing operations
  - Makes series stationary
  - d = 0: no differencing
  - d = 1: first difference
  - d = 2: second difference

MA(q) - Moving Average:
  - Uses past forecast errors
  - y(t) = μ + ε(t) + θ₁·ε(t-1) + θ₂·ε(t-2) + ... + θ₆·ε(t-q)
  - q = number of lagged forecast errors
  - ACF cuts off after lag q

ARIMA(p,d,q):
  - Combines all three components
  - Flexible and powerful
  - Most widely used time series model
""")
```

## Building ARIMA Models

### Preparing Data

```python
# Load or generate data
np.random.seed(42)
n = 365 * 3
dates = pd.date_range('2021-01-01', periods=n, freq='D')

# Generate time series with trend and seasonality
trend = np.linspace(100, 150, n)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 3, n)
y = trend + seasonality + noise

ts = pd.Series(y, index=dates, name='value')

# Train/test split (80/20)
train_size = int(0.8 * len(ts))
train = ts[:train_size]
test = ts[train_size:]

print(f"Total samples: {len(ts)}")
print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Train', linewidth=2)
plt.plot(test.index, test, label='Test', linewidth=2)
plt.axvline(train.index[-1], color='red', linestyle='--', label='Train/Test Split')
plt.legend()
plt.title('Time Series: Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Determining ARIMA Parameters

```python
from statsmodels.tsa.stattools import adfuller

# Check stationarity
def check_stationarity(series, name='Series'):
    result = adfuller(series.dropna())
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    is_stationary = result[1] < 0.05
    print(f"  Stationary: {is_stationary}")
    return is_stationary

check_stationarity(train, 'Original')

# Difference if needed
train_diff = train.diff().dropna()
check_stationarity(train_diff, 'First Difference')

# ACF and PACF for parameter selection
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Original
plot_acf(train, lags=40, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Original')

plot_pacf(train, lags=40, ax=axes[0, 1], method='ywm')
axes[0, 1].set_title('PACF - Original')

# Differenced
plot_acf(train_diff, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('ACF - First Difference')

plot_pacf(train_diff, lags=40, ax=axes[1, 1], method='ywm')
axes[1, 1].set_title('PACF - First Difference')

plt.tight_layout()
plt.show()

print("""
PARAMETER SELECTION GUIDELINES:

p (AR order):
  - Look at PACF
  - Number of significant lags before cutoff
  
q (MA order):
  - Look at ACF
  - Number of significant lags before cutoff
  
d (Differencing):
  - Apply until stationary (ADF test p < 0.05)
  - Usually d=0, 1, or 2
  - Start with d=1 if trend present
""")
```

### Manual ARIMA Model

```python
# Fit ARIMA model
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()

print("\n" + "="*70)
print("ARIMA(2,1,2) MODEL SUMMARY")
print("="*70)
print(model_fit.summary())

# Diagnostics
fig = model_fit.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.show()

print("""
DIAGNOSTIC PLOTS:

1. Standardized Residuals:
   - Should look like white noise
   - No patterns or trends
   
2. Histogram + KDE:
   - Should be normally distributed
   - Centered at zero
   
3. Q-Q Plot:
   - Points should follow red line
   - Indicates normality
   
4. Correlogram (ACF):
   - No significant autocorrelation
   - All within confidence bands
""")
```

### Grid Search for Best Parameters

```python
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define parameter ranges
p_values = range(0, 4)
d_values = range(0, 3)
q_values = range(0, 4)

# Generate all combinations
parameter_combinations = list(itertools.product(p_values, d_values, q_values))

print(f"\nTesting {len(parameter_combinations)} parameter combinations...")

# Grid search
results = []

for params in parameter_combinations:
    try:
        model = ARIMA(train, order=params)
        model_fit = model.fit()
        
        # Calculate AIC
        aic = model_fit.aic
        bic = model_fit.bic
        
        results.append({
            'order': params,
            'AIC': aic,
            'BIC': bic
        })
    except:
        continue

# Sort by AIC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')

print("\nTop 10 Models by AIC:")
print(results_df.head(10).to_string(index=False))

# Best model
best_order = results_df.iloc[0]['order']
print(f"\n\nBest ARIMA order: {best_order}")
print(f"AIC: {results_df.iloc[0]['AIC']:.2f}")
print(f"BIC: {results_df.iloc[0]['BIC']:.2f}")
```

### Auto ARIMA

```python
# Install: pip install pmdarima
from pmdarima import auto_arima

print("\nRunning Auto ARIMA...")

# Automatic model selection
auto_model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # Let it determine d
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True
)

print("\n" + "="*70)
print("AUTO ARIMA RESULTS")
print("="*70)
print(auto_model.summary())

print(f"\nSelected order: {auto_model.order}")
```

## Forecasting

```python
# Fit best model on train data
best_model = ARIMA(train, order=best_order)
best_fit = best_model.fit()

# Forecast
n_forecast = len(test)
forecast = best_fit.forecast(steps=n_forecast)
forecast_index = test.index

# Confidence intervals
forecast_df = best_fit.get_forecast(steps=n_forecast)
forecast_ci = forecast_df.conf_int()

# Visualize
plt.figure(figsize=(14, 6))

# Historical data
plt.plot(train.index, train, label='Train', linewidth=2)
plt.plot(test.index, test, label='Test (Actual)', linewidth=2)

# Forecast
plt.plot(forecast_index, forecast, label='Forecast', linewidth=2, color='red')

# Confidence interval
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 alpha=0.3, color='red', label='95% Confidence')

plt.axvline(train.index[-1], color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.title(f'ARIMA{best_order} Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Forecast accuracy
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("\n" + "="*70)
print("FORECAST ACCURACY")
print("="*70)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Residual analysis
residuals = test - forecast

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Residuals over time
axes[0, 0].plot(residuals)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Forecast Residuals')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].grid(alpha=0.3)

# Histogram
axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].grid(alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(alpha=0.3)

# ACF of residuals
plot_acf(residuals, lags=20, ax=axes[1, 1])
axes[1, 1].set_title('Residual ACF')

plt.tight_layout()
plt.show()
```

## SARIMA - Seasonal ARIMA

**SARIMA(p,d,q)(P,D,Q)m**: Adds seasonal components

```python
print("""
SARIMA PARAMETERS:

Non-Seasonal: (p, d, q)
  p: AR order
  d: Differencing order  
  q: MA order

Seasonal: (P, D, Q, m)
  P: Seasonal AR order
  D: Seasonal differencing order
  Q: Seasonal MA order
  m: Seasonal period (12 for monthly, 4 for quarterly, etc.)

Example: SARIMA(1,1,1)(1,1,1,12)
  - Non-seasonal: AR(1), I(1), MA(1)
  - Seasonal: AR(1), I(1), MA(1) with period 12
""")

# Generate seasonal data
np.random.seed(42)
n_monthly = 365
dates_monthly = pd.date_range('2021-01-01', periods=n_monthly, freq='D')

trend = np.linspace(100, 150, n_monthly)
seasonality_yearly = 15 * np.sin(2 * np.pi * np.arange(n_monthly) / 365)
seasonality_weekly = 5 * np.sin(2 * np.pi * np.arange(n_monthly) / 7)
noise = np.random.normal(0, 2, n_monthly)

y_seasonal = trend + seasonality_yearly + seasonality_weekly + noise
ts_seasonal = pd.Series(y_seasonal, index=dates_monthly)

# Split
train_s = ts_seasonal[:int(0.8*len(ts_seasonal))]
test_s = ts_seasonal[int(0.8*len(ts_seasonal)):]

# Fit SARIMA
sarima_model = SARIMAX(
    train_s,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)

print("\n" + "="*70)
print("SARIMA MODEL SUMMARY")
print("="*70)
print(sarima_fit.summary())

# Forecast
forecast_s = sarima_fit.forecast(steps=len(test_s))

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(train_s, label='Train')
plt.plot(test_s, label='Test')
plt.plot(test_s.index, forecast_s, label='SARIMA Forecast', linewidth=2)
plt.axvline(train_s.index[-1], color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.title('SARIMA Forecast with Weekly Seasonality')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Accuracy
mae_s = mean_absolute_error(test_s, forecast_s)
rmse_s = np.sqrt(mean_squared_error(test_s, forecast_s))

print(f"\nSARIMA Forecast Accuracy:")
print(f"MAE:  {mae_s:.2f}")
print(f"RMSE: {rmse_s:.2f}")
```

## Prophet for Robust Forecasting

```python
# Install: pip install prophet
from prophet import Prophet

# Prepare data for Prophet (needs 'ds' and 'y' columns)
df_prophet = pd.DataFrame({
    'ds': ts.index,
    'y': ts.values
})

train_prophet = df_prophet[:train_size]
test_prophet = df_prophet[train_size:]

print("\nFitting Prophet model...")

# Initialize and fit
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

m.fit(train_prophet)

# Create future dataframe
future = m.make_future_dataframe(periods=len(test), freq='D')

# Predict
forecast_prophet = m.predict(future)

# Visualize
fig1 = m.plot(forecast_prophet)
plt.axvline(train.index[-1], color='red', linestyle='--', label='Train/Test Split')
plt.legend()
plt.title('Prophet Forecast')
plt.tight_layout()
plt.show()

# Components
fig2 = m.plot_components(forecast_prophet)
plt.tight_layout()
plt.show()

# Extract forecast for test period
forecast_test = forecast_prophet.tail(len(test))['yhat'].values

# Accuracy
mae_prophet = mean_absolute_error(test, forecast_test)
rmse_prophet = np.sqrt(mean_squared_error(test, forecast_test))

print("\n" + "="*70)
print("PROPHET FORECAST ACCURACY")
print("="*70)
print(f"MAE:  {mae_prophet:.2f}")
print(f"RMSE: {rmse_prophet:.2f}")

# Compare all models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10}")
print("-" * 40)
print(f"{'ARIMA':<20} {mae:<10.2f} {rmse:<10.2f}")
print(f"{'SARIMA':<20} {mae_s:<10.2f} {rmse_s:<10.2f}")
print(f"{'Prophet':<20} {mae_prophet:<10.2f} {rmse_prophet:<10.2f}")
```

## Production Forecasting Pipeline

```python
import joblib

class TimeSeriesForecaster:
    def __init__(self, model_type='arima', **kwargs):
        self.model_type = model_type
        self.model = None
        self.model_fit = None
        self.kwargs = kwargs
    
    def fit(self, train_data):
        """Fit the model"""
        if self.model_type == 'arima':
            order = self.kwargs.get('order', (1, 1, 1))
            self.model = ARIMA(train_data, order=order)
            self.model_fit = self.model.fit()
        
        elif self.model_type == 'sarima':
            order = self.kwargs.get('order', (1, 1, 1))
            seasonal_order = self.kwargs.get('seasonal_order', (1, 1, 1, 7))
            self.model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            self.model_fit = self.model.fit(disp=False)
        
        elif self.model_type == 'prophet':
            df = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
            self.model = Prophet(**self.kwargs)
            self.model.fit(df)
        
        return self
    
    def forecast(self, steps):
        """Generate forecast"""
        if self.model_type in ['arima', 'sarima']:
            forecast = self.model_fit.forecast(steps=steps)
            return forecast
        
        elif self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=steps, freq='D')
            forecast = self.model.predict(future)
            return forecast.tail(steps)['yhat'].values
    
    def save(self, filename):
        """Save model"""
        joblib.dump(self, filename)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load(filename):
        """Load model"""
        return joblib.load(filename)

# Example usage
forecaster = TimeSeriesForecaster(model_type='arima', order=(2, 1, 2))
forecaster.fit(train)
forecast = forecaster.forecast(steps=len(test))

print(f"\nForecasted {len(forecast)} steps")
print(f"Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")

# Save
forecaster.save('time_series_model.pkl')

# Load and use
loaded_model = TimeSeriesForecaster.load('time_series_model.pkl')
new_forecast = loaded_model.forecast(steps=30)
print(f"\nNew forecast: {len(new_forecast)} steps")
```

## Key Takeaways

:::{admonition} Summary
:class: note

1. **ARIMA(p,d,q)** models non-seasonal time series
2. **p**: AR order (PACF), **q**: MA order (ACF), **d**: differencing
3. **Auto ARIMA** automates parameter selection
4. **SARIMA** extends ARIMA for seasonal patterns
5. **Prophet** handles holidays, multiple seasonality, robustly
6. **Grid search** finds optimal parameters
7. **Diagnostics** check model assumptions
8. **Residuals** should be white noise
9. **Multiple metrics** (MAE, RMSE, MAPE) evaluate forecasts
10. **Production pipelines** ensure reproducibility
:::

## Further Reading

- Box, G. & Jenkins, G. (1970). "Time Series Analysis: Forecasting and Control"
- Hyndman, R.J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice"
- Prophet Documentation: [Prophet](https://facebook.github.io/prophet/)
- pmdarima: [Auto ARIMA](http://alkaline-ml.com/pmdarima/)
