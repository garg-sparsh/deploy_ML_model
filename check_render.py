import requests
import json


df  = json.dumps({'age': 25, 'workclass':'Self-emp-not-inc', 'fnlgt': 176756, 'education': 'HS-grad', 
    'education_num': 9, 'marital_status': 'Never-married', 'occupation': 'Farming-fishing', 'relationship': 'Own-child', 'race':'White', 
    'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week':35, 'native_country': 'United-States'})

r = requests.post("https://deploy-ml-model-tcb6.onrender.com/predict", data=df)

print("status_code",r.status_code)
print("result",r.json())

