# data_collection.py
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_data(stock_symbol, period='1y', interval='1d'):
    """
    Fetches stock data from Yahoo Finance.
    """
    stock_data = yf.download(stock_symbol, period=period, interval=interval, auto_adjust=True)
    
    # Check if we got data
    if stock_data.empty:
        raise ValueError(f"No data found for {stock_symbol} with period={period} and interval={interval}")
    
    stock_data = stock_data.ffill()  # Forward fill missing values
    return stock_data

def preprocess_data(stock_data):
    """
    Scales the 'Close' price.
    """
    data = stock_data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    """
    Creates input-output datasets.
    """
    if len(data) <= time_step:
        raise ValueError("Not enough data to create input sequences. Try a longer period or smaller time_step.")

    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y