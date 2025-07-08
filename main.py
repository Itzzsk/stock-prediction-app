import streamlit as st
import os
import pandas as pd
import locale
from datetime import datetime
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import google.generativeai as genai

# ------------------ SETUP ------------------
load_dotenv()
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# API KEYS
GEMINI_API_KEY = os.getenv("API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Gemini Model Init
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Sentiment Analyzer Init
analyzer = SentimentIntensityAnalyzer()

# ------------------ AI CALL ------------------
def ai_call(stock, actual_price, predicted_price, sentiment_score=0):
    prompt = f"""
    Write a short paragraph advising whether to BUY or SELL {stock} stock.
    - Current price: ₹{actual_price}
    - Predicted price: ₹{predicted_price}
    - Market sentiment score: {sentiment_score}
    Keep it concise and helpful.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "AI response failed."

# ------------------ SENTIMENT ------------------
def get_sentiment_review(text):
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']
    score = int((compound + 1) * 50)
    if compound >= 0.05:
        return score, "Positive"
    elif compound <= -0.05:
        return score, "Negative"
    else:
        return score, "Neutral"

# ------------------ STYLE ------------------
def indicator(label, color):
    html = f"""
    <div style="text-align:center;">
        <button style="
            background-color:{color};
            color:white;
            width:100%;
            border:none;
            padding:10px;
            font-size:18px;
            border-radius:8px;">
            {label}
        </button>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ------------------ GET SYMBOL ------------------
def get_stock_symbol(user_input, exchange):
    suffix = "BSE" if exchange == "BSE" else "NS"
    return f"{user_input.upper()}.{suffix}"

# ------------------ FETCH DAILY DATA ------------------
def fetch_data_alpha(symbol):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

        if data.empty:
            raise ValueError("No data returned from Alpha Vantage. Check symbol or rate limit.")

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
        st.error(f"🔴 Data fetch failed: {e}")
        return pd.DataFrame()

# ------------------ ARIMA MODEL ------------------
def predict_stock_action(symbol):
    data = fetch_data_alpha(symbol)

    if data.empty or len(data) < 10:
        raise ValueError("⚠️ Not enough data for ARIMA model.")

    data.sort_values('date', inplace=True)
    data['Price'] = data['Close']
    data.dropna(inplace=True)

    try:
        model = ARIMA(data['Price'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predicted_price = forecast.iloc[0]
    except Exception as e:
        raise RuntimeError(f"ARIMA model failed: {e}")

    last_price = data['Price'].iloc[-1]
    action = "Buy" if predicted_price > last_price else "Sell"
    return last_price, predicted_price, action, data

# ------------------ PLOT & DISPLAY ------------------
def plot_graph(symbol):
    try:
        last_price, predicted_price, action, df = predict_stock_action(symbol)
        formatted_price = locale.currency(last_price, grouping=True)

        fig = go.Figure(data=[go.Candlestick(
            x=df['date'],
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f'{symbol} - Daily Candlestick',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            xaxis_rangeslider_visible=False
        )

        st.header(f"📈 Live Stock Price: {formatted_price}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Buy/Sell Recommendation")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Actual Price:** ₹{round(last_price, 2)}")
        with c2:
            st.write(f"**Predicted Price:** ₹{round(predicted_price, 2)}")

        advice = ai_call(symbol, round(last_price, 2), round(predicted_price, 2))
        score, review = get_sentiment_review(advice)

        indicator("BUY" if action == "Buy" else "SELL", "#1dcf46" if action == "Buy" else "red")

        st.subheader("🧠 Market Sentiment")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"**Score:** {score}")
        with s2:
            st.markdown(f"**Review:** {review}")

        st.subheader("✨ AI Investment Suggestion")
        st.write(advice)

        st.subheader("📄 Raw Data (Latest 5)")
        st.dataframe(df.tail())

    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="InvestIQ", layout="wide")
st.title("📊 InvestIQ")
st.caption("Smart Insights, Smarter Investments")
st.divider()

col1, col2 = st.columns(2)
stock_name = col1.text_input("Enter stock name (e.g., TCS, RELIANCE)", "")
exchange = col2.selectbox("Select exchange", ["BSE", "NSE"])

symbol = ''
if st.button("🔍 Analyze"):
    if not stock_name.strip():
        st.warning("Please enter a valid stock name.")
    else:
        symbol = get_stock_symbol(stock_name.strip(), exchange)

if symbol:
    st.write("")
    plot_graph(symbol)

st.divider()
st.subheader("Project by ")
st.markdown("\n Harish")
