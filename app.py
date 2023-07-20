# app.py
import yfinance as yf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model.load('stockPrediction1.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (historical stock prices)
        data = request.json

        # Preprocess the data as needed (e.g., convert to proper format)
        # ...

        # Make predictions using the model
        predictions = model.predict(data)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
