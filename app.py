import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import nltk

# Download NLTK corpora (required for TextBlob)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('wordnet')

# Function to fetch real-time stock data
def fetch_realtime_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    return data

# Function to fetch news and analyze sentiment
def fetch_news_sentiment(query="India"):
    api_key = "a0d81686b96d49bb9b6bf37b1db7b12c"  # Your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    
    sentiments = []
    for article in articles:
        text = article.get("title", "") + " " + article.get("description", "")
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment

# Function to preprocess data for LSTM
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to build and train LSTM model
def build_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10)
    return model

# Function to predict future prices using LSTM
def predict_future_prices(model, data, scaler, future_days=30):
    last_60_days = data['Close'][-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    
    predictions = []
    for _ in range(future_days):
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        predictions.append(pred_price[0, 0])
        
        # Update last_60_days_scaled
        last_60_days_scaled = np.append(last_60_days_scaled[1:], pred_price)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit App
st.title("Enhanced Stock Price Prediction App 📈")
st.write("This app predicts future stock prices using LSTM, real-time data, and market sentiment.")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, INFY):", "INFY")

# Fetch real-time stock data
if st.button("Analyze"):
    st.write(f"Fetching real-time data for {ticker}...")
    stock_data = fetch_realtime_stock_data(ticker)
    st.write("### Real-Time Stock Data")
    st.write(stock_data)
    
    # Plot real-time data
    st.write("### Real-Time Stock Price Chart")
    st.line_chart(stock_data['Close'])
    
    # Fetch news sentiment
    st.write("### Market Sentiment Analysis")
    sentiment = fetch_news_sentiment()
    st.write(f"Average Sentiment Score (India-related news): {sentiment:.2f}")
    
    # Preprocess data and train LSTM model
    st.write("### Training LSTM Model...")
    X, y, scaler = preprocess_data(stock_data)
    model = build_lstm_model(X, y)
    
    # Predict future prices
    st.write("### Future Stock Price Prediction")
    future_days = 30
    predictions = predict_future_prices(model, stock_data, scaler, future_days)
    
    # Create future dates
    future_dates = pd.date_range(stock_data.index[-1], periods=future_days + 1)[1:]
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions.flatten()
    })
    st.write(predictions_df)
    
    # Plot future predictions
    st.write("### Predicted Stock Price Chart")
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label="Historical Prices")
    ax.plot(future_dates, predictions, label="Predicted Prices", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
