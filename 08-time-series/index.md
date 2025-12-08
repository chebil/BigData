# Chapter 8: Time Series Analysis

## Introduction

Time series analysis examines data collected over time to identify patterns, trends, and underlying structures. Unlike cross-sectional data (observations at a single point in time), time series data exhibits temporal dependencies where current observations are influenced by past values.

**Definition**: A **time series** is a sequence of observations \(y_1, y_2, ..., y_n\) ordered by time \(t_1, t_2, ..., t_n\), typically at equally spaced intervals.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand time series components**: Identify trend, seasonality, cyclical, and random components
2. **Test for stationarity**: Apply statistical tests (ADF, KPSS) and visual inspection
3. **Transform non-stationary data**: Use differencing, detrending, and logarithmic transformations
4. **Interpret ACF and PACF plots**: Identify AR, MA, and ARMA patterns
5. **Build ARIMA models**: Select appropriate orders (p, d, q) using Box-Jenkins methodology
6. **Handle seasonality**: Implement seasonal ARIMA models with period s
7. **Evaluate models**: Use AIC, BIC, residual diagnostics, and out-of-sample validation
8. **Generate forecasts**: Produce point predictions with confidence intervals
9. **Apply modern methods**: Use exponential smoothing and Prophet for real-world forecasting
10. **Implement in Python**: Use statsmodels, pmdarima, and Prophet libraries

## Time Series Applications

### Financial Markets
- Stock price prediction
- Volatility forecasting (GARCH models)
- Algorithmic trading strategies
- Risk management and VaR estimation

### Retail and E-commerce
- Sales forecasting for inventory management
- Demand planning across product lines
- Seasonal promotion effectiveness
- Customer traffic prediction

### Energy and Utilities
- Electricity load forecasting
- Renewable energy production prediction
- Smart grid optimization
- Energy price modeling

### Healthcare
- Disease outbreak prediction
- Patient admission forecasting
- Medical resource allocation
- Epidemic modeling (COVID-19)

### Manufacturing and Supply Chain
- Production planning
- Spare parts demand forecasting
- Quality control monitoring
- Supply chain disruption detection

### Climate and Environment
- Temperature and precipitation forecasting
- Climate change modeling
- Air quality prediction
- Natural disaster early warning systems

## Time Series Components

A time series \(y_t\) can be decomposed into:

\[
y_t = T_t + S_t + C_t + R_t
\]

Where:
- **\(T_t\)**: **Trend** - Long-term increase or decrease
- **\(S_t\)**: **Seasonality** - Fixed, periodic fluctuations
- **\(C_t\)**: **Cyclical** - Non-fixed periodic patterns (business cycles)
- **\(R_t\)**: **Random** (Irregular) - Unpredictable noise

### Decomposition Models

**1. Additive Model**:
\[
y_t = T_t + S_t + C_t + R_t
\]

- Use when seasonal variation is constant over time
- Appropriate when magnitude of fluctuations doesn't grow with level

**2. Multiplicative Model**:
\[
y_t = T_t \times S_t \times C_t \times R_t
\]

- Use when seasonal variation increases with level
- Common in economic and financial data
- Can be converted to additive via log transformation:

\[
\log(y_t) = \log(T_t) + \log(S_t) + \log(C_t) + \log(R_t)
\]

## Chapter Structure

This chapter is organized into the following sections:

1. **[Stationarity](01-stationarity.md)**: Tests and transformations
2. **[Autocorrelation](02-autocorrelation.md)**: ACF and PACF interpretation
3. **[ARIMA Models](03-arima-models.md)**: AR, MA, ARMA, and ARIMA
4. **[Seasonal ARIMA](04-seasonal-arima.md)**: SARIMA models
5. **[Model Selection](05-model-selection.md)**: Box-Jenkins methodology
6. **[Forecasting](06-forecasting.md)**: Prediction and confidence intervals
7. **[Advanced Methods](07-advanced-methods.md)**: Exponential smoothing, Prophet, LSTM

## Quick Start Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load example data: Monthly airline passengers
from statsmodels.datasets import get_rdataset
data = get_rdataset('AirPassengers').data
data['time'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data.set_index('time', inplace=True)
ts = data['value']

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Monthly Airline Passengers (1949-1960)')
plt.ylabel('Thousands of Passengers')
plt.xlabel('Year')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Decompose
decomp = seasonal_decompose(ts, model='multiplicative', period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# Build SARIMA model
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

print(results.summary())

# Forecast 24 months ahead
forecast = results.get_forecast(steps=24)
forecast_df = forecast.summary_frame()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Historical')
plt.plot(forecast_df['mean'], label='Forecast', color='red')
plt.fill_between(forecast_df.index, 
                 forecast_df['mean_ci_lower'], 
                 forecast_df['mean_ci_upper'], 
                 color='red', alpha=0.2)
plt.legend()
plt.title('SARIMA Forecast')
plt.ylabel('Passengers')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Key Concepts Preview

### Stationarity

A time series is **stationary** if:
1. Mean is constant over time
2. Variance is constant over time
3. Covariance between periods depends only on lag, not on time

**Why important?** Most time series models assume stationarity

### Autocorrelation

**Autocorrelation**: Correlation between \(y_t\) and \(y_{t-k}\) (lag k)

**ACF (Autocorrelation Function)**: Shows all autocorrelations
**PACF (Partial Autocorrelation Function)**: Shows direct correlations, removing indirect effects

### ARIMA Components

- **AR(p)**: Autoregressive - Uses p past values
- **I(d)**: Integrated - Applies d differences
- **MA(q)**: Moving Average - Uses q past errors

**ARIMA(p,d,q)**: Combines all three

### Box-Jenkins Methodology

**Iterative process**:
1. **Identification**: Plot data, check stationarity, examine ACF/PACF
2. **Estimation**: Fit ARIMA model, estimate parameters
3. **Diagnostic**: Check residuals, validate assumptions
4. **Forecast**: Generate predictions if model is adequate

## Common Pitfalls

❌ **Ignoring non-stationarity**: Leads to spurious regression
❌ **Over-differencing**: Unnecessarily increases variance
❌ **Ignoring seasonality**: Systematic patterns left in residuals
❌ **Using in-sample metrics only**: Overfitting risk
❌ **Long-term forecasts without caution**: Uncertainty grows rapidly
❌ **Not checking residuals**: Violations of white noise assumption
❌ **Assuming linearity**: Real data may have regime changes

## Best Practices

✅ **Always plot the data first**: Visual inspection is crucial
✅ **Test for stationarity**: Use ADF test, not just eyeballing
✅ **Start simple**: Try ARIMA(1,1,1) before complex models
✅ **Use information criteria**: AIC/BIC for model comparison
✅ **Validate out-of-sample**: Rolling window or walk-forward
✅ **Check residuals thoroughly**: ACF, normality, heteroscedasticity
✅ **Consider domain knowledge**: Incorporate business constraints
✅ **Report uncertainty**: Always include confidence intervals
✅ **Monitor model performance**: Retrain regularly with new data
✅ **Document assumptions**: Stationarity, no structural breaks, etc.

## Software Tools

### Python
- **statsmodels**: Comprehensive time series analysis
- **pmdarima**: Automated ARIMA modeling (auto_arima)
- **Prophet**: Facebook's forecasting tool
- **sktime**: Unified ML framework for time series
- **tslearn**: Machine learning for time series

### R
- **forecast**: Rob Hyndman's comprehensive package
- **tseries**: Time series analysis and tests
- **vars**: Vector autoregression
- **rugarch**: GARCH modeling
- **prophet**: Facebook Prophet

### Specialized Tools
- **MATLAB**: Econometrics Toolbox
- **SAS**: SAS/ETS
- **SPSS**: Time Series Modeler
- **Stata**: Time series commands

## Data Requirements

### Minimum Requirements
- **At least 40-50 observations**: For reliable estimation
- **For seasonality**: At least 2-3 full seasonal cycles
- **Equal spacing**: ARIMA assumes equal time intervals
- **No missing values**: Must be handled (imputation or interpolation)

### Data Quality Considerations
- **Outliers**: Can distort parameter estimates
- **Structural breaks**: Regime changes, policy changes
- **Calendar effects**: Trading days, holidays
- **Level shifts**: Sudden permanent changes

## Performance Metrics

### In-Sample Metrics
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion
- **Log-likelihood**: Goodness of fit

### Out-of-Sample Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric MAPE
- **Coverage**: Proportion of actuals within confidence intervals

## Case Study Preview

### Retail Sales Forecasting

**Problem**: Forecast next 12 months of retail sales

**Data**: 5 years of monthly sales (60 observations)

**Approach**:
1. Decompose: Identify trend + seasonality
2. Transform: Log transformation to stabilize variance
3. Difference: Once for trend, once at lag 12 for seasonality
4. Model: SARIMA(1,1,1)(1,1,1)[12]
5. Validate: Out-of-sample on last 12 months
6. Forecast: Next 12 months with 95% CI

**Result**: MAPE < 5% on test set

Detailed implementation in [Forecasting Section](06-forecasting.md).

## Advanced Topics (Not Covered)

- **Multivariate Time Series**: VAR, VECM
- **State Space Models**: Kalman filtering
- **GARCH**: Volatility modeling
- **Regime Switching**: Markov switching models
- **Neural Networks**: LSTM, GRU for time series
- **Causal Inference**: Granger causality, intervention analysis

## Further Reading

### Books
- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos (free online)
- "Time Series Analysis and Its Applications" by Shumway & Stoffer
- "Introductory Time Series with R" by Cowpertwait & Metcalfe
- "Analysis of Financial Time Series" by Tsay

### Papers
- Box & Jenkins (1970): Original ARIMA methodology
- Dickey & Fuller (1979): Unit root tests
- Hyndman et al. (2008): Automatic ARIMA
- Taylor & Letham (2018): Facebook Prophet

### Online Resources
- Penn State STAT 510: Applied Time Series Analysis
- Duke STATS 170: Forecasting
- Coursera: Practical Time Series Analysis

## Summary

Time series analysis is essential for:
- Understanding temporal patterns
- Making data-driven forecasts
- Planning and resource allocation
- Detecting anomalies and interventions

The ARIMA framework provides a flexible approach to modeling various time series patterns. Modern methods like Prophet and deep learning extend capabilities for complex real-world scenarios.

**Ready to start?** Continue to [Stationarity](01-stationarity.md) to learn the foundation of time series modeling.

---

## Next Chapter

After mastering time series analysis, proceed to:
- **[Chapter 9: Text Analytics](../09-text-analytics/index.md)** - Natural language processing and text mining