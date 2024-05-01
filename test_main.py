import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_post_below():
    data = json.dumps({'age': 37, 'workclass':'Private', 'fnlgt': 284582, 'education': 'Masters', 
    'education_num': 14, 'marital_status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Wife', 'race':'White', 
    'sex': 'Female', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week':40, 'native_country': 'United-States'})
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


if __name__ == '__main__':
    test_post_below()
    test_post_above()
    test_get()

