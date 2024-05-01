# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import data_process
import pickle
import pandas as pd
import model
import numpy as np

app = FastAPI()
loaded_model = joblib.load('model/random_forest.pkl')
lb = pickle.load(open('model/binarizer.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
cat_features = ["workclass", "education",  "marital-status", "occupation",  "relationship",  "race", "sex", "native-country"]

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.post('/predict')
async def predict(data: InputData):
    # Extract features from input data
    features = np.array([[data.age, data.workclass, data.fnlgt, data.education, data.education_num, data.marital_status, data.occupation, data.relationship, data.race, data.sex, data.capital_gain, data.capital_loss, data.hours_per_week, data.native_country]]) 
     # Adjust as needed
    df_temp = pd.DataFrame(data=features, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country",
    ])
    encoder = pickle.load(open("model/encoder.pkl", "rb"))
    lb = pickle.load(open("model/binarizer.pkl", "rb"))
    #features = pd.DataFrame(features)
    #features.columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain','capital-loss','hours-per-week','native-country']
    print('features:', features)
    X_test, _,_,_ = data_process.process_data(df_temp, categorical_features=cat_features, label=None, training=False, 
    encoder=encoder, lb=lb)
    # Perform inference using the loaded model
    prediction = model.inference(loaded_model, X_test)
    prediction = lb.inverse_transform(prediction)[0]
    # Return the prediction
    return {'prediction': prediction}

@app.get("/")
def welcome(str = "Welcome"):
    return {"Message": str}