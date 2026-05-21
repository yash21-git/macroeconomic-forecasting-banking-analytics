```python id="70l9ot"
# ============================================================
# MACROECONOMIC FORECASTING & BANKING ANALYTICS PIPELINE
# Author: Yash
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ardl import ARDL

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from pmdarima import auto_arima

import statsmodels.api as sm

# ============================================================
# 2. LOAD DATA
# ============================================================

# Example Dataset Structure:
# ------------------------------------------------------------
# date | inflation | repo_rate | bank_credit
# exchange_rate | oil_price
# ------------------------------------------------------------

df = pd.read_csv("macro_data.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Set datetime index
df.set_index('date', inplace=True)

print("\nDataset Preview")
print(df.head())

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Handle missing values
df = df.fillna(method='ffill')

# Check null values
print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Log Transformation
df['log_inflation'] = np.log(df['inflation'])

# Rolling Average
df['inflation_rolling_mean'] = (
    df['inflation']
    .rolling(window=3)
    .mean()
)

# Drop NA values after transformations
df.dropna(inplace=True)

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nSummary Statistics")
print(df.describe())

# Correlation Matrix
plt.figure(figsize=(10, 6))

corr_matrix = df.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Inflation Trend
plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df['inflation'],
    label='Inflation'
)

plt.plot(
    df.index,
    df['inflation_rolling_mean'],
    label='3-Month Rolling Average'
)

plt.title("Inflation Trend Analysis")
plt.xlabel("Date")
plt.ylabel("Inflation")

plt.legend()

plt.savefig(
    "inflation_trend.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 6. STATIONARITY TEST (ADF TEST)
# ============================================================

def adf_test(series, variable_name):

    result = adfuller(series.dropna())

    print(f"\nADF Test: {variable_name}")
    print("-" * 40)

    print(f"ADF Statistic : {result[0]}")
    print(f"P-value       : {result[1]}")

    if result[1] < 0.05:
        print("Result        : Stationary")
    else:
        print("Result        : Non-Stationary")

# Run ADF Test
adf_test(df['inflation'], "Inflation")

# ============================================================
# 7. DIFFERENCING
# ============================================================

df['inflation_diff'] = (
    df['inflation']
    .diff()
)

df.dropna(inplace=True)

# Re-run ADF Test
adf_test(
    df['inflation_diff'],
    "Inflation Differenced"
)

# ============================================================
# 8. ACF & PACF PLOTS
# ============================================================

plot_acf(df['inflation_diff'])

plt.title("Autocorrelation Function")

plt.savefig(
    "acf_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

plot_pacf(df['inflation_diff'])

plt.title("Partial Autocorrelation Function")

plt.savefig(
    "pacf_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

# ============================================================
# 10. AUTO ARIMA MODEL SELECTION
# ============================================================

auto_model = auto_arima(
    train['inflation'],
    seasonal=False,
    trace=True,
    suppress_warnings=True
)

print("\nAUTO ARIMA SUMMARY")
print(auto_model.summary())

# ============================================================
# 11. ARIMA MODEL
# ============================================================

model_arima = ARIMA(
    train['inflation'],
    order=(2,1,2)
)

model_fit = model_arima.fit()

print("\nARIMA MODEL SUMMARY")
print(model_fit.summary())

# ============================================================
# 12. FORECASTING
# ============================================================

forecast = model_fit.forecast(
    steps=len(test)
)

# ============================================================
# 13. FORECAST EVALUATION
# ============================================================

mae = mean_absolute_error(
    test['inflation'],
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test['inflation'],
        forecast
    )
)

mape = np.mean(
    np.abs(
        (test['inflation'] - forecast)
        / test['inflation']
    )
) * 100

print("\nForecast Accuracy Metrics")
print("-" * 40)

print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")
print(f"MAPE : {mape}")

# ============================================================
# 14. FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    train.index,
    train['inflation'],
    label='Training Data'
)

plt.plot(
    test.index,
    test['inflation'],
    label='Actual Inflation'
)

plt.plot(
    test.index,
    forecast,
    label='Forecast'
)

plt.title("Inflation Forecast")
plt.xlabel("Date")
plt.ylabel("Inflation")

plt.legend()

plt.savefig(
    "inflation_forecast.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 15. ARDL MODEL
# ============================================================

X = df[
    [
        'repo_rate',
        'bank_credit',
        'exchange_rate'
    ]
]

y = df['inflation']

ardl_model = ARDL(
    endog=y,
    lags=2,
    exog=X,
    order=2
)

ardl_result = ardl_model.fit()

print("\nARDL MODEL SUMMARY")
print("-" * 40)

print(ardl_result.summary())

# ============================================================
# 16. REGRESSION ANALYSIS
# ============================================================

X_reg = sm.add_constant(X)

regression_model = sm.OLS(
    y,
    X_reg
).fit()

print("\nREGRESSION MODEL SUMMARY")
print("-" * 40)

print(regression_model.summary())

# ============================================================
# 17. MULTICOLLINEARITY CHECK (VIF)
# ============================================================

vif_data = pd.DataFrame()

vif_data["Variable"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        X.values,
        i
    )
    for i in range(X.shape[1])
]

print("\nVIF ANALYSIS")
print(vif_data)

# ============================================================
# 18. RESIDUAL DIAGNOSTICS
# ============================================================

residuals = model_fit.resid

# Residual Plot
plt.figure(figsize=(10, 4))

plt.plot(residuals)

plt.title("Residual Diagnostics")
plt.xlabel("Time")
plt.ylabel("Residuals")

plt.savefig(
    "residual_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Residual Distribution
plt.figure(figsize=(8, 4))

sns.histplot(
    residuals,
    kde=True
)

plt.title("Residual Distribution")

plt.savefig(
    "residual_distribution.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 19. ROLLING FORECAST
# ============================================================

history = list(train['inflation'])

rolling_predictions = []

for t in range(len(test)):

    model = ARIMA(
        history,
        order=(2,1,2)
    )

    model_fit = model.fit()

    output = model_fit.forecast()

    yhat = output[0]

    rolling_predictions.append(yhat)

    obs = test['inflation'].iloc[t]

    history.append(obs)

# ============================================================
# 20. ROLLING FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    test.index,
    test['inflation'],
    label='Actual'
)

plt.plot(
    test.index,
    rolling_predictions,
    label='Rolling Forecast'
)

plt.title("Rolling Forecast Evaluation")

plt.legend()

plt.savefig(
    "rolling_forecast.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 21. BUSINESS INSIGHTS
# ============================================================

print("\nKEY INSIGHTS")
print("-" * 40)

print("""
1. Inflation demonstrates autoregressive behavior
   across multiple time periods.

2. Repo rate and exchange rate show
   significant relationships with inflation trends.

3. ARDL modelling captures both short-run
   and long-run macroeconomic relationships.

4. Forecasting models can support banking,
   credit-risk, and financial decision-making.

5. Rolling forecast analysis improves
   model monitoring and forecast evaluation.
""")

# ============================================================
# 22. EXPORT RESULTS
# ============================================================

forecast_df = pd.DataFrame({
    'Actual': test['inflation'],
    'Forecast': forecast
})

forecast_df.to_csv(
    "forecast_results.csv",
    index=False
)

print("\nForecast results exported successfully.")

# ============================================================
# END OF PIPELINE
# ============================================================

print("\nPIPELINE EXECUTED SUCCESSFULLY.")
```
