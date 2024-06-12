from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import __main__ 
from deployment.feature_engineering import feature_engineering, label_encode, label_encode_transform


__main__.feature_engineering = feature_engineering
__main__.label_encode = label_encode
__main__.label_encode_transform = label_encode_transform

app = Flask(__name__)

pipeline = joblib.load('lgbm_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)

    df = pd.DataFrame(data['features'])
    print("DataFrame columns:", df.columns)
    
    df = feature_engineering(df)
    
    predictions_proba = pipeline.predict_proba(df)[:, 1]
    return jsonify({'predictions': predictions_proba.tolist()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
