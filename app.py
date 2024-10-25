from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('D:/Workspace/Projects/Assignments/ML/model/lstm_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Get user input
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = stock_data['Close']

    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    time_step = 60
    X_test = []
    for i in range(time_step, len(scaled_data)):
        X_test.append(scaled_data[i-time_step:i, 0])

    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predict stock prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Prepare data for plotting
    actual_prices = close_prices[time_step:].values
    predicted_prices = predictions.flatten()
    dates = stock_data.index[time_step:].strftime('%Y-%m-%d').tolist()  # Convert to string format

    return render_template('evaluation.html',
                           ticker=ticker,
                           actual_prices=actual_prices.tolist(),
                           predicted_prices=predicted_prices.tolist(),
                           dates=dates)


if __name__ == '__main__':
    app.run(debug=True)
