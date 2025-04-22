import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_results(y_actual, y_predicted, stock_symbol):
    """
    Plots actual vs predicted prices.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label="Actual Price", color='blue')
    plt.plot(y_predicted, label="Predicted Price", color='red')
    plt.title(f"{stock_symbol} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def evaluate_model(y_actual, y_predicted):
    """
    Prints MAE and MSE.
    """
    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    print(f"Mean Absolute Error: {mae:.5f}")
    print(f"Mean Squared Error: {mse:.5f}")