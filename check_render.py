import requests


df = {
    "age": 38,
    "workclass": "Private",
    "fnlgt": 28887,
    "education": "11th",
    "education_num": 7,
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Sales",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain":0,
    "capital_loss":0,
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
    }
r = requests.post('https://nd0821-c3-starter-code-master-1.onrender.com', json=df)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

