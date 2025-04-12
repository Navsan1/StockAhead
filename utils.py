import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_results(y_test_actual, predicted_stock_price, stock_symbol):
    """
    Plots the actual vs. predicted stock prices.
    :param y_test_actual: Actual stock prices
    :param predicted_stock_price: Predicted stock prices
    :param stock_symbol: The stock symbol
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, color='blue', label=f'Actual {stock_symbol} Stock Price')
    plt.plot(predicted_stock_price, color='red', label=f'Predicted {stock_symbol} Stock Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def evaluate_model(y_test_actual, predicted_stock_price):
    """
    Evaluates the model by calculating Mean Absolute Error and Mean Squared Error.
    :param y_test_actual: Actual stock prices
    :param predicted_stock_price: Predicted stock prices
    """
    mae = mean_absolute_error(y_test_actual, predicted_stock_price)
    mse = mean_squared_error(y_test_actual, predicted_stock_price)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
