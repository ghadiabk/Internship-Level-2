import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
output_dir = 'Level-2/plots/t3'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('Level-2/Sentiment_Data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

ts = df.resample('M', on='Timestamp').size()

decomposition = seasonal_decompose(ts, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(ts, label='Original', color='blue')
axes[0].set_title('Time Series Decomposition')
axes[0].legend(loc='upper left')

axes[1].plot(decomposition.trend, label='Trend', color='red')
axes[1].legend(loc='upper left')

axes[2].plot(decomposition.seasonal, label='Seasonality', color='green')
axes[2].legend(loc='upper left')

axes[3].plot(decomposition.resid, label='Residuals', color='orange')
axes[3].legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ts_decomposition.png'))

ts_ma = ts.rolling(window=3).mean()
model_ses = SimpleExpSmoothing(ts).fit(smoothing_level=0.3, optimized=False)
ts_ses = model_ses.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original Data', alpha=0.4)
plt.plot(ts_ma, label='3-Month Moving Average', color='red', linewidth=2)
plt.plot(ts_ses, label='Exponential Smoothing (SES)', color='green', linestyle='--')
plt.title('Smoothing Techniques: MA vs SES')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ts_smoothing.png'))

train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model_sarima.fit(disp=False)

forecast_obj = model_fit.get_forecast(steps=len(test))
forecast_mean = forecast_obj.summary_frame()['mean']
conf_int = forecast_obj.summary_frame()[['mean_ci_lower', 'mean_ci_upper']]

rmse = np.sqrt(mean_squared_error(test, forecast_mean))
print(f"Model Evaluation (RMSE): {rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Test Data', color='gray', alpha=0.7)
plt.plot(forecast_mean, label='SARIMA Forecast', color='red')
plt.fill_between(conf_int.index, conf_int['mean_ci_lower'], 
                 conf_int['mean_ci_upper'], color='pink', alpha=0.3, label='Confidence Interval')
plt.title(f'Sentiment Volume Forecast (RMSE: {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Monthly Post Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ts_forecast.png'))