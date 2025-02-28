import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import datetime

# Function to fetch stock data from Yahoo Finance (Last 1 Year)
def get_stock_data(ticker):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)  # 1 year back

    stock = yf.download(ticker, start=start, end=end)

    if stock.empty or 'Close' not in stock.columns:
        st.error("❌ No valid stock data found. Please check the ticker symbol.")
        return None  

    return add_technical_indicators(stock)

# Function to add technical indicators
def add_technical_indicators(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.error("❌ DataFrame is empty or invalid.")
        return None  
    
    if 'Close' not in df.columns:
        st.error("❌ 'Close' price column is missing.")
        return None  

    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Drop rows where 'Close' is NaN
    df = df.dropna(subset=['Close'])

    if df.empty:
        st.error("❌ No valid closing price data available.")
        return None  

    # Apply technical indicators
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])

    return df

# Streamlit UI
st.title("📈 Stock Research & Analysis (Last 1 Year)")

ticker = st.text_input("🔍 Enter Stock Ticker", "AAPL")

if st.button("🚀 Analyze"):
    stock_data = get_stock_data(ticker)
    
    if stock_data is not None:
        st.subheader("📊 Stock Data (Last 1 Year)")
        st.dataframe(stock_data.tail())  # Show latest stock data
        st.line_chart(stock_data[['Close', 'SMA_50', 'SMA_200']])  # Plot price & moving averages
