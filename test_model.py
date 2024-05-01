from model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pytest
import data_process
import joblib

dataset = pd.read_csv('data/census.csv', sep=', ')
label = 'salary'
# Optional enhancement, use K-fold cross validation instead of a train-test split.




# Add the necessary imports for the starter code.

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(dataset, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = data_process.process_data(
    train, categorical_features=cat_features, label="salary", training=True)

loaded_model = joblib.load('model/random_forest.pkl')
lb = pickle.load(open('model/binarizer.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))
X_test, y_test, encoder, lb = data_process.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

@pytest.mark.parametrize("train_model",[train_model])
def test_train(train_model):
    
    try :
        model = train_model(X_train, y_train)
        
    except Exception as err:
        raise err
    
        
@pytest.mark.parametrize("compute_model_metrics",[compute_model_metrics])
def test_metrics(compute_model_metrics):
    
    try :
        preds = loaded_model.predict(X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        outs = np.array([precision, recall, fbeta])
     
    except Exception as err:
        raise err
        
    assert len(outs)==3
    
    for o in outs:
        assert type(o)==np.float64
        assert 0<=o<=1

@pytest.mark.parametrize("inference",[inference])
def test_inference(inference):
    
    try :
        preds = inference(loaded_model, X_test)
        
    except Exception as err:
        raise err
    
    assert type(preds)==np.ndarray
    
    assert len(preds)==len(X_test)
    
    assert ((preds==1)|(preds==0)).all()
    