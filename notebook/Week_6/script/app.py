import streamlit as st
import joblib
import numpy as np

model = joblib.load('model/iris_mdl.pkl')
labels = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']

st.title("Iris Flower Classification - Prediction APP")

SepalL=st.number_input("Sepal Length in cm: ")
SepalW=st.number_input("Sepal Width in cm: ")
PetalL=st.number_input("Petal Length in cm: ")
PetalW=st.number_input("Petal Width in cm: ")

button = st.button("Prediction")

if button:
    pred = model.predict([[SepalL,SepalW,PetalL,PetalW]])
    pred_label=labels[pred[0]]
    st.subheader(f'Class prediction: {pred_label}')