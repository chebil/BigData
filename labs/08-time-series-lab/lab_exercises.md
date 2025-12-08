# Lab 8: Time Series Forecasting - Sales Prediction

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
```

## Part 1: Data Exploration (20 points)

### Exercise 1.1: Load and Visualize (10 points)

```python
# Load sample sales data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')

# Generate synthetic sales data
trend = np.linspace(1000, 1500, len(dates))
seasonality_yearly = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
seasonality_weekly = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = np.random.normal(0, 30, len(dates))

sales = trend + seasonality_yearly + seasonality_weekly + noise
sales_df = pd.DataFrame({'date': dates, 'sales': sales})
sales_df.set_index('date', inplace=True)

print("Sales Data:")
print(sales_df.head())
print(f"\nShape: {sales_df.shape}")
print(f"Date range: {sales_df.index.min()} to {sales_df.index.max()}")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(sales_df.index, sales_df['sales'], linewidth=1)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Daily Sales Data')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q1: What patterns do you observe?
- Q2: Is there a trend?
- Q3: Do you see seasonality?

### Exercise 1.2: Decomposition (10 points)

**TODO:** Decompose the time series

```python
# Decompose
decomposition = seasonal_decompose(sales_df['sales'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

decomposition.observed.plot(ax=axes[0])
axes[0].set_ylabel('Observed')

decomposition.trend.plot(ax=axes[1])
axes[1].set_ylabel('Trend')

decomposition.seasonal.plot(ax=axes[2])
axes[2].set_ylabel('Seasonal')

decomposition.resid.plot(ax=axes[3])
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.show()
```

**Questions:**
- Q4: Describe the trend component
- Q5: What is the seasonal period?
- Q6: How large is the residual component?

---

## Part 2: Stationarity Testing (20 points)

### Exercise 2.1: ADF Test (10 points)

```python
def check_stationarity(series, name=''):
    result = adfuller(series.dropna())
    
    print(f"ADF Test Results for {name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  ✓ STATIONARY (p < 0.05)")
    else:
        print(f"  ✗ NON-STATIONARY (p ≥ 0.05)")
    return result

# Test original
check_stationarity(sales_df['sales'], 'Original Series')

# TODO: Test differenced series
sales_diff = sales_df['sales'].diff().dropna()
check_stationarity(sales_diff, 'First Difference')
```

**Questions:**
- Q7: Is the original series stationary?
- Q8: Does differencing make it stationary?
- Q9: Why is stationarity important?

### Exercise 2.2: ACF and PACF (10 points)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(sales_diff, lags=50, ax=axes[0])
axes[0].set_title('ACF')

plot_pacf(sales_diff, lags=50, ax=axes[1], method='ywm')
axes[1].set_title('PACF')

plt.tight_layout()
plt.show()
```

**Questions:**
- Q10: What AR order would you suggest?
- Q11: What MA order would you suggest?

---

## Part 3: ARIMA Modeling (30 points)

### Exercise 3.1: Train/Test Split (5 points)

```python
# Split data (80/20)
train_size = int(0.8 * len(sales_df))
train = sales_df[:train_size]
test = sales_df[train_size:]

print(f"Train: {len(train)} samples")
print(f"Test: {len(test)} samples")

plt.figure(figsize=(14, 6))
plt.plot(train.index, train['sales'], label='Train')
plt.plot(test.index, test['sales'], label='Test')
plt.axvline(train.index[-1], color='red', linestyle='--')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Train/Test Split')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Exercise 3.2: Fit ARIMA (15 points)

**TODO:** Fit ARIMA model

```python
# Choose order based on ACF/PACF
order = (1, 1, 1)  # (p, d, q) - adjust based on your analysis

model = ARIMA(train['sales'], order=order)
model_fit = model.fit()

print("Model Summary:")
print(model_fit.summary())

# Diagnostics
fig = model_fit.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.show()
```

**Questions:**
- Q12: Are the residuals white noise?
- Q13: What is the AIC of your model?
- Q14: Try different orders - which is best?

### Exercise 3.3: Forecasting (10 points)

```python
# Forecast
forecast_steps = len(test)
forecast = model_fit.forecast(steps=forecast_steps)

# Get confidence intervals
forecast_df = model_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast_df.conf_int()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['sales'], label='Train')
plt.plot(test.index, test['sales'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red', linewidth=2)
plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                 alpha=0.3, color='red', label='95% CI')
plt.axvline(train.index[-1], color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f'ARIMA{order} Forecast')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test['sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
mape = np.mean(np.abs((test['sales'] - forecast) / test['sales'])) * 100

print(f"\nForecast Metrics:")
print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
```

**Questions:**
- Q15: How accurate is the forecast?
- Q16: Does the forecast capture seasonality?
- Q17: How would you improve the model?

---

## Part 4: Prophet (20 points)

### Exercise 4.1: Fit Prophet (10 points)

```python
# Prepare data for Prophet
train_prophet = train.reset_index()
train_prophet.columns = ['ds', 'y']

# Fit Prophet
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(train_prophet)

# Make future dataframe
future = m.make_future_dataframe(periods=len(test), freq='D')

# Predict
forecast_prophet = m.predict(future)

# Plot
fig = m.plot(forecast_prophet)
plt.axvline(train.index[-1], color='red', linestyle='--')
plt.title('Prophet Forecast')
plt.tight_layout()
plt.show()

# Components
fig = m.plot_components(forecast_prophet)
plt.tight_layout()
plt.show()
```

**Questions:**
- Q18: What components did Prophet identify?
- Q19: How does seasonality look?

### Exercise 4.2: Evaluate Prophet (10 points)

```python
# Extract test predictions
forecast_test = forecast_prophet.tail(len(test))['yhat'].values

mae_prophet = mean_absolute_error(test['sales'], forecast_test)
rmse_prophet = np.sqrt(mean_squared_error(test['sales'], forecast_test))
mape_prophet = np.mean(np.abs((test['sales'] - forecast_test) / test['sales'])) * 100

print(f"\nProphet Metrics:")
print(f"  MAE: {mae_prophet:.2f}")
print(f"  RMSE: {rmse_prophet:.2f}")
print(f"  MAPE: {mape_prophet:.2f}%")

# Compare models
print(f"\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
print("-" * 50)
print(f"{'ARIMA':<20} {mae:<10.2f} {rmse:<10.2f} {mape:<10.2f}%")
print(f"{'Prophet':<20} {mae_prophet:<10.2f} {rmse_prophet:<10.2f} {mape_prophet:<10.2f}%")
```

**Questions:**
- Q20: Which model performs better?
- Q21: When would you choose ARIMA vs Prophet?

---

## Part 5: Production Forecast (10 points)

**TODO:** Create production forecast for next 30 days

```python
# Retrain on all data
full_data_prophet = sales_df.reset_index()
full_data_prophet.columns = ['ds', 'y']

# Fit final model
final_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
final_model.fit(full_data_prophet)

# Forecast next 30 days
future_30 = final_model.make_future_dataframe(periods=30, freq='D')
final_forecast = final_model.predict(future_30)

# Plot
fig = final_model.plot(final_forecast)
plt.title('30-Day Sales Forecast')
plt.tight_layout()
plt.show()

# Export forecast
final_30_days = final_forecast.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("\nNext 30 Days Forecast:")
print(final_30_days)

# Save
final_30_days.to_csv('sales_forecast_30days.csv', index=False)
print("\nForecast saved to 'sales_forecast_30days.csv'")
```

**Final Questions:**
- Q22: What is the expected sales for next week?
- Q23: What is the confidence interval?
- Q24: How would you monitor forecast accuracy?
- Q25: When should the model be retrained?

Good luck! 📈
