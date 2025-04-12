import argparse
import numpy as np
from data_collection import fetch_data, preprocess_data, create_dataset
from model import build_model
from utils import plot_results, evaluate_model
from sklearn.model_selection import train_test_split

def predict_and_visualize(stock_symbol, period='1y', interval='1d'):
    """
    Fetches, preprocesses the stock data, trains the model, and visualizes the prediction.
    :param stock_symbol: The stock symbol (e.g., 'AAPL','GOOG','MSFT')
    :param period: The time period for the data (e.g., '1y', '6mo')
    :param interval: The time interval (e.g., '1d', '1wk')
    """
    # Step 1: Fetch and preprocess data
    stock_data = fetch_data(stock_symbol, period=period, interval=interval)
    scaled_data, scaler = preprocess_data(stock_data)

    # Step 2: Create datasets
    X, y = create_dataset(scaled_data)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Step 4: Build and train the model
    model = build_model(X_train, y_train)

    # Step 5: Predict on test data
    predicted_stock_price = model.predict(X_test)

    # Step 6: Inverse transform the scaled data
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Step 7: Plot the results
    plot_results(y_test_actual, predicted_stock_price, stock_symbol)

    # Step 8: Evaluate the model
    evaluate_model(y_test_actual, predicted_stock_price)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction")
    parser.add_argument('stock_symbol', type=str, help="Stock symbol (e.g., 'AAPL')")
    parser.add_argument('--period', type=str, default='1y', help="Period for stock data (e.g., '1y', '6mo')")
    parser.add_argument('--interval', type=str, default='1d', help="Interval for stock data (e.g., '1d', '1wk')")
    args = parser.parse_args()

    predict_and_visualize(args.stock_symbol, period=args.period, interval=args.interval)
