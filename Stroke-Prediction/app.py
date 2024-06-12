from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

model = load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract feature data from the request
    data = request.get_json(force=True)['features']
    
    # Create a DataFrame using the input data;
    features_df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(features_df)
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
