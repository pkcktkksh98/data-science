from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
app = FastAPI()

model  = joblib.load("model/iris_mdl_prob.pkl")
labels = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']

class Features(BaseModel):
    sepalL:float
    sepalW:float
    petalL:float
    petalW:float

@app.get("/ping")
def ping():
    return{"message":"Working!"}

@app.post("/predict/")
def predict(features:Features):
    features_input = np.array([features.sepalL,
                         features.sepalW,
                         features.petalL,
                         features.petalW]).reshape(1,-1)
    pred = model.predict(features_input)
    proba = model.predict_proba(features_input)
    pred_labels = labels[pred[0]]
    proba = max(proba[0])

    return{"prediction":pred_labels,
           "prediction score":proba}
