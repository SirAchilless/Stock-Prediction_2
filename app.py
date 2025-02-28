import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime

# Function to get stock data from Yahoo Finance (Last 1 Year)
def get_stock_data(ticker):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)  # 1 year back

    stock = yf.download(ticker, start=start, end=end)

    if stock.empty:
        st.error("❌ No stock data found. Please check the ticker symbol.")
        return None  

    stock = stock.fillna(method="ffill").dropna()  # Fill missing values and remove any remaining NaN
    return add_technical_indicators(stock)

# Function to add technical indicators
def add_technical_indicators(df):
    df = df.copy()
    
    # Ensure 'Close' column exists and has valid data
    if 'Close' not in df.columns or df['Close'].dropna().empty:
        st.error("❌ Error: 'Close' price data is missing or invalid.")
        return None  
    
    # Convert 'Close' column to numeric (handles any data type issues)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Apply technical indicators
    try:
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
    except Exception as e:
        st.error(f"❌ Error in technical indicators: {str(e)}")
        return None

    # Replace remaining NaN values with zero
    df = df.fillna(0)

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
