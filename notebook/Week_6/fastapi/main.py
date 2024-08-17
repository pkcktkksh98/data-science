from fastapi import FastAPI
import joblib
import numpy as np
app = FastAPI()

model  = joblib.load("model/iris_mdl.pkl")
labels = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']

@app.get("/ping")
def ping():
    return{"message":"Working!"}

@app.post("/predict/")
def predict(data:dict):
    features = np.array(data['features']).reshape(1,-1)
    pred = model.predict(features)
    pred_labels = labels[pred[0]]

    return{"prediction":pred_labels}
