# main.py
import argparse
from data_collection import fetch_data, preprocess_data, create_dataset
from model import build_model
from utils import plot_results, evaluate_model

def predict_stock(stock_symbol, period='1y', interval='1d'):
    print(f"Fetching data for {stock_symbol}...")
    stock_data = fetch_data(stock_symbol, period, interval)
    scaled_data, scaler = preprocess_data(stock_data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Split into train and test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    if len(X_test) == 0:
        raise ValueError("Not enough data left for testing. Try a longer period or smaller interval.")

    model = build_model(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    # Inverse scale
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

    evaluate_model(y_test_actual, y_pred_actual)
    plot_results(y_test_actual, y_pred_actual, stock_symbol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('stock_symbol', type=str, help="Stock symbol like AAPL, MSFT, TSLA")
    parser.add_argument('--period', type=str, default='1y', help="Data period (e.g., 1y, 6mo)")
    parser.add_argument('--interval', type=str, default='1d', help="Data interval (e.g., 1d, 1wk)")
    args = parser.parse_args()

    predict_stock(args.stock_symbol, args.period, args.interval)