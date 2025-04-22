from flask import Flask, render_template, request
from data_collection import fetch_data, preprocess_data, create_dataset
from model import build_model
from utils import plot_results, evaluate_model
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    stock_symbol = ""
    
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        period = request.form.get('period', '1y')
        interval = request.form.get('interval', '1d')

        (result, error_message) = predict_and_visualize(stock_symbol, period, interval)
        
        if result:
            y_test_actual, predicted_stock_price = result
            plot_results(y_test_actual, predicted_stock_price, stock_symbol)
            prediction = True

    return render_template('index.html', prediction=prediction, stock_symbol=stock_symbol, error_message=error_message)

def predict_and_visualize(stock_symbol, period='1y', interval='1d'):
    stock_data = fetch_data(stock_symbol, period=period, interval=interval)

    if stock_data is None or stock_data.empty:
        return None, "Error: No data found for given stock and period."

    scaled_data, scaler = preprocess_data(stock_data)

    time_step = 60
    if len(scaled_data) <= time_step:
        return None, "Error: Not enough data to predict. Please select longer period."

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - time_step:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    if X_train is None or X_test is None:
        return None, "Error: Not enough data to create datasets."

    model = build_model(X_train, y_train)

    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return (y_test_actual, predicted_stock_price), None

if __name__ == '__main__':
    app.run(debug=True)
