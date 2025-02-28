import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    
    if stock.empty:
        st.error("❌ Error: No stock data found. Please check the ticker symbol.")
        return None  # Prevent further processing if no data is found

    stock = stock.fillna(method="ffill").dropna()  # Fill missing values and drop any remaining NaN
    return add_technical_indicators(stock)

# Function to add technical indicators
def add_technical_indicators(df):
    df = df.copy()
    
    # Ensure 'Close' column exists and has valid data
    if 'Close' not in df.columns or df['Close'].isnull().all():
        st.error("❌ Error: 'Close' price data is missing or invalid.")
        return df

    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])

    df = df.fillna(0)  # Replace any remaining NaN values with zero
    return df

# Streamlit UI
st.title("Stock Prediction with AI & Sentiment Analysis")

ticker = st.text_input("Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button("Analyze"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        st.subheader("Stock Data")
        st.dataframe(stock_data.tail())  # Show latest stock data
