# Script to train machine learning model.
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import data_process
import model
# Add the necessary imports for the starter code.

# Add code to load in the data.
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

trained_model = model.train_model(X_train, y_train)
joblib.dump(trained_model, 'model/random_forest.pkl')

with open("model/encoder.pkl", "wb") as f: 
    pickle.dump(encoder, f)
f.close()

with open("model/binarizer.pkl", "wb") as f: 
    pickle.dump(lb, f)
f.close()


X_test, y_test, encoder, lb = data_process.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

y_preds = trained_model.predict(X_test)
precision, recall, fbeta = model.compute_model_metrics(y_test, y_preds)
print(precision, recall, fbeta)

# Optional: implement hyperparameter tuning.
## For inference
#X_test, y_test, encoder, lb = data_process.process_data(
#    test, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)

loaded_model = joblib.load('model/random_forest.pkl')
preds = model.inference(loaded_model, X_test)

print(preds)

slices_results = []

for feature in cat_features:
    for val in test[feature].unique():
        
        filtered = test[feature]==val

        precision, recall, fbeta = model.compute_model_metrics(y_test[filtered],
                                                         preds[filtered])
        
        slices_results.append({"feature":feature,"val":val, "precision":precision,
                   "recall": recall, "fbeta":fbeta})
        
slices = pd.DataFrame(slices_results)

slices.to_csv("./slice_output.txt", index=None)