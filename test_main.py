import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_post_below():
    data = json.dumps({'age': 25, 'workclass':'Self-emp-not-inc', 'fnlgt': 176756, 'education': 'HS-grad', 
    'education_num': 9, 'marital_status': 'Never-married', 'occupation': 'Farming-fishing', 'relationship': 'Own-child', 'race':'White', 
    'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week':35, 'native_country': 'United-States'})
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}

def test_post_above():
    data = json.dumps({'age': 42, 'workclass':'Private', 'fnlgt': 116632, 'education': 'Doctorate', 
    'education_num': 16, 'marital_status': 'Married-civ-spouse', 'occupation': 'Prof-specialty', 'relationship': 'Husband', 'race':'White', 
    'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week':45, 'native_country': 'United-States'})
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Message": "Welcome"}

