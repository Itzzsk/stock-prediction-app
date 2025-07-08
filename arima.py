from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import os

# Load API key from environment
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

def fetch_data_alpha(symbol):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

        if data.empty:
            raise ValueError("No data returned from Alpha Vantage. Check symbol or rate limit.")

        # Rename columns for consistency
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)

        data.reset_index(inplace=True)
        return data

    except Exception as e:
        print(f"🔴 Data fetch failed: {e}")
        return pd.DataFrame()

def predict_stock_action(symbol):
    data = fetch_data_alpha(symbol)

    if data.empty or len(data) < 10:
        raise ValueError("⚠️ Not enough data for ARIMA model.")

    # Sort by date (ascending)
    data.sort_values('date', inplace=True)

    # Use 'Close' price for prediction
    data['Price'] = data['Close']
    data.dropna(inplace=True)

    # Fit ARIMA model
    try:
        model = ARIMA(data['Price'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted_price = forecast.iloc[0]
    except Exception as e:
        raise RuntimeError(f"ARIMA model failed: {e}")

    # Last actual price
    last_price = data['Price'].iloc[-1]

    # Suggest action
    action = "Buy" if predicted_price > last_price else "Sell"

    return last_price, predicted_price, action, data
