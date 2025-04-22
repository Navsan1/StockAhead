# app.py
from flask import Flask, render_template, request
from data_collection import fetch_data, preprocess_data, create_dataset
from model import build_model
from utils import evaluate_model
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def predict_stock(stock_symbol, period='1y', interval='1d'):
    stock_data = fetch_data(stock_symbol, period, interval)
    scaled_data, scaler = preprocess_data(stock_data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    if len(X) == 0:
        raise ValueError("Not enough data to predict. Try a longer period or different interval.")

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    if len(X_test) == 0:
        raise ValueError("Not enough data left for testing.")

    model = build_model(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

    mae, mse = evaluate_model(y_test_actual, y_pred_actual)

    # Plotting
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(y_test_actual, label='Actual Price', color='blue')
    ax.plot(y_pred_actual, label='Predicted Price', color='red')
    ax.set_title(f'{stock_symbol} Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()

    plt.close()

    return plot_data, mae, mse

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        period = request.form.get("period", "1y")
        interval = request.form.get("interval", "1d")
        try:
            plot_data, mae, mse = predict_stock(stock_symbol, period, interval)
            return render_template("index.html", plot_data=plot_data, mae=mae, mse=mse, stock_symbol=stock_symbol)
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)