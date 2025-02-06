import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Set the title
st.title('Stock Price Viewer')

# User input for selecting multiple stocks
stock_symbols = st.text_input('Enter stock symbols separated by commas (e.g., TCS.NS, RELIANCE.NS, INFY.NS)', 'TCS.NS, RELIANCE.NS')

# Split the input string into a list of symbols
stock_list = [symbol.strip() for symbol in stock_symbols.split(',')]

# Number of columns to display graphs side by side
cols_per_row = 2

# Create columns dynamically based on the number of stocks
columns = st.columns(cols_per_row)

# Fetch and plot data for each selected stock
for i, ticker in enumerate(stock_list):
    if ticker:  # Ensure the ticker symbol is not empty
        with columns[i % cols_per_row]:  # Arrange plots in columns
            st.write(f'## {ticker} Stock Price')

            # Fetch real-time data
            data = yf.Ticker(ticker)
            df = data.history(period='1d', start='2020-01-01', end=None)

            # Plot the data
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax.legend()

            # Display the plot
            st.pyplot(fig)
