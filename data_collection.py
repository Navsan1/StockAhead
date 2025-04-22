import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_data(stock_symbol, period='1y', interval='1d'):
    """
    Fetches stock data from Yahoo Finance.
    :param stock_symbol: The stock symbol (e.g., 'AAPL')
    :param period: The time period for the data (e.g., '1y', '6mo')
    :param interval: The time interval (e.g., '1d', '1wk')
    :return: stock_data (Pandas DataFrame)
    """
    stock_data = yf.download(stock_symbol, period=period, interval=interval)
    return stock_data

def preprocess_data(stock_data):
    """
    Preprocesses the stock data by scaling the 'Close' price.
    :param stock_data: The stock data to preprocess
    :return: scaled_data (numpy array), scaler (MinMaxScaler object)
    """
    data = stock_data[['Close']]  # We are only using 'Close' price for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    """
    Creates datasets for model training with a sliding window of 60 days.
    :param data: The scaled stock data
    :param time_step: Number of days used to predict the next day's price
    :return: X (input data), y (output data)
    """
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Previous 60 days as input
        y.append(data[i, 0])  # The next day's price as output
    return np.array(X), np.array(y)
