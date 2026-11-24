#  Time Series Analysis and Forecasting - Onion prices at Pune APMC
 
# 1. Aim
# To perform time series analysis on daily onion prices of 
# Pune APMC (2012 to 2026) and build a SARIMA model that forecasts the weekly modal price.

# python -m pip install statsmodels
# Data: CEDA / Agmarknet daily onion prices (2012-2026), from the mini project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

df = pd.read_csv('onion_pune_mumbai_2012_2026.csv')
df = df[df['market'] == 'Pune'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').dropna(subset=['modal_price'])

print('Pune rows:', len(df), ' From',
      df['date'].min().date(), 'to', df['date'].max().date())

# Resample the irregular daily data to a weekly series
weekly = df.set_index('date')['modal_price'].resample('W').mean()
print('Weekly points:', len(weekly),
      ' Empty weeks filled:', weekly.isnull().sum())
weekly = weekly.interpolate()

# Stationarity test
adf = adfuller(weekly)
print('\nADF statistic: %.3f   p-value: %.4f' % (adf[0], adf[1]))
print('Series is', 'stationary' if adf[1] < 0.05 else 'not stationary')

# Monthly seasonality
print('\nMedian price by month (Rs/quintal):')
monthly = weekly.groupby(weekly.index.month).median().round(0)
monthly.index = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]
print(monthly.to_string())

# Train / test split by time (last 52 weeks for testing)
train, test = weekly[:-52], weekly[-52:]
print('\nTrain weeks:', len(train), ' Test weeks:', len(test))

# Rolling 13-week forecasts from 4 quarterly origins
# (only past data is used each time)
H = 13
origins = [
    train.index[-1] + pd.Timedelta(weeks=H * i)
    for i in range(4)
]
sarima_fc, naive_fc = [], []

for origin in origins:
    history = weekly[:origin]
    weeks = test.index[test.index > origin][:H]

    fit = SARIMAX(
        np.log(history),
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    sarima_fc.append(
        pd.Series(
            np.exp(fit.forecast(len(weeks))).values,
            index=weeks
        )
    )
    naive_fc.append(
        pd.Series(history.iloc[-1], index=weeks)
    )

forecast = pd.concat(sarima_fc)
naive = pd.concat(naive_fc)

for name, pred in [('Naive', naive), ('SARIMA', forecast)]:
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = np.mean(np.abs((test - pred) / test)) * 100
    print(f'{name:7s} MAE: {mae:8.1f}   RMSE: {rmse:8.1f}   MAPE: {mape:5.1f}%')

# Final model on all data, forecast next 13 weeks
final = SARIMAX(
    np.log(weekly),
    order=(1, 0, 1),
    seasonal_order=(1, 0, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future = np.exp(final.get_forecast(steps=13).predicted_mean)

print('\nForecast for the next 13 weeks (Rs/quintal):')
print(future.round(0).to_string())

decomp = seasonal_decompose(weekly, model='additive', period=52)

fig, axes = plt.subplots(3, 1, figsize=(11, 9))

axes[0].plot(weekly, linewidth=1, label='Weekly price')
axes[0].plot(weekly.rolling(52).mean(),
             color='red', label='52-week mean')
axes[0].set_title('Weekly Modal Price of Onion - Pune APMC')
axes[0].legend()

axes[1].plot(decomp.seasonal[-156:], color='purple')
axes[1].set_title('Seasonal Component (last 3 years)')

axes[2].plot(weekly[-104:], color='gray', label='Actual')
axes[2].plot(forecast, color='blue', label='SARIMA test forecast')
axes[2].plot(future, color='green', linestyle='--',
             label='Future forecast')
axes[2].set_title('Forecast vs Actual')
axes[2].legend()

for ax in axes:
    ax.set_ylabel('Rs / quintal')

plt.tight_layout()
plt.savefig('figs/exp10_timeseries.png', dpi=110)
plt.show()
