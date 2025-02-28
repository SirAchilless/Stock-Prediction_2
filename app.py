import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import ta  # Using ta instead of TA-Lib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return add_technical_indicators(stock)

# Function to add technical indicators using `ta`
def add_technical_indicators(df):
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    return df

# Function to fetch market sentiment from news
def fetch_news_sentiment(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=YOUR_API_KEY"
    response = requests.get(url).json()
    articles = response.get('articles', [])

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles]
    
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Function to train Prophet model
def train_prophet_model(df):
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Function to prepare data for LSTM
def prepare_data(df):
    df = df[['Close']].dropna()
    data = df.values
    X, y = [], []
    for i in range(50, len(data)):
        X.append(data[i-50:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Function to train LSTM model
def train_lstm_model(df):
    X, y = prepare_data(df)
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model

# Function to predict using LSTM
def predict_lstm(model, df):
    X, _ = prepare_data(df)
    return model.predict(X)

# Streamlit UI
st.title("Stock Prediction with AI & Sentiment Analysis")

ticker = st.text_input("Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button("Analyze"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    sentiment_score = fetch_news_sentiment(ticker)
    
    prophet_forecast = train_prophet_model(stock_data)
    lstm_model = train_lstm_model(stock_data)
    lstm_predictions = predict_lstm(lstm_model, stock_data)

    st.subheader("Prophet Prediction")
    st.line_chart(prophet_forecast.set_index('ds')['yhat'])

    st.subheader("LSTM Prediction")
    st.line_chart(lstm_predictions)

    st.subheader("Market Sentiment Score")
    st.write(f"Sentiment Score: {sentiment_score}")

    st.subheader("Technical Indicators")
    st.line_chart(stock_data[['Close', 'SMA_50', 'SMA_200']])
