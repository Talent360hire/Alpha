# Minimal Flask API for Render
# Save this as app.py

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and train model
try:
    data = pd.read_csv('sample_stock_data.csv')
    data['Prev_Close'] = data['Close'].shift(1)
    data = data.dropna()
    X = data[['Prev_Close']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
except Exception as e:
    model = None
    print(f"Model training failed: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained'}), 500
    req_data = request.get_json()
    prev_close = req_data.get('prev_close')
    if prev_close is None:
        return jsonify({'error': 'Missing prev_close'}), 400
    pred = model.predict([[prev_close]])
    return jsonify({'predicted_close': float(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
