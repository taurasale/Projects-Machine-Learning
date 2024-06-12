import requests

# Example raw data for prediction
data = {
    "gender": "Male",
    "age": 28.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Rural",
    "avg_glucose_level": 200.69,
    "bmi": 30.5,
    "smoking_status": "smokes"
}

response = requests.post('http://127.0.0.1:5000/predict', json={'features': data})
print(response.json())
