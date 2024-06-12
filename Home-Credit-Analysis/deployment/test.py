import requests
import pandas as pd
import numpy

df = pd.read_csv('test.csv', index_col= 0)

data = df.iloc[:1].to_dict(orient='records')

payload = {'features': data}

response = requests.post('https://predict0610v1-p7ebgcjdvq-lm.a.run.app/predict', json=payload)

try:
    response_data = response.json()
    print(response_data)
except requests.exceptions.JSONDecodeError as e:
    print(f"Failed to decode JSON response: {e}")
    print(f"Response text: {response.text}")
