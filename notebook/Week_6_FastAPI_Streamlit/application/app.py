import streamlit as st
import joblib
import numpy as np

model = joblib.load('model\iris_mdl.pkl')
labels = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']

st.title("Iris Flower Classification - Prediction APP")

# SepalL=st.number_input("Sepal Length in cm: ")
SepalL = float(st.slider("Sepal Length in cm:", min_value=4.0, max_value=8.0, value=6.0, step=0.01))
# SepalW=st.number_input("Sepal Width in cm: ")
SepalW = float(st.slider("Sepal Width in cm:", min_value=1.5, max_value=5.0, value=3.0, step=0.01))
# PetalL=st.number_input("Petal Length in cm: ")
PetalL = float(st.slider("Petal Length in cm:", min_value=0.5, max_value=7.0, value=4.5,step=0.01))
# PetalW=st.number_input("Petal Width in cm: ")
PetalW = float(st.slider("Petal Width in cm:", min_value=0.05, max_value=3.0, value=1.5, step=0.01))

button = st.button("Prediction")

if button:
    pred = model.predict([[SepalL,SepalW,PetalL,PetalW]])
    pred_label=labels[pred[0]]
    st.subheader(f'Class prediction: {pred_label}')

    if pred_label=='Iris-setosa':
        st.image('img/setosa.jpg')
    elif pred_label=='Iris-versicolor':
        st.image('img/versicolor.jpg')
    else:
        st.image('img/virginica.jpg')