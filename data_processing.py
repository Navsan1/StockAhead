from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(stock_data):
    """
    Preprocesses stock data by scaling the features.
    :param stock_data: The stock data with added features
    :return: scaled_data (numpy array), scaler (MinMaxScaler object)
    """
    features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']
    data = stock_data[features]
    
    # Normalize the features
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
        X.append(data[i-time_step:i, :])  # Previous 60 days as input (with all features)
        y.append(data[i, 0])  # The next day's closing price as output
    return np.array(X), np.array(y)
