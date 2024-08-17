import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("My First Streamlit Application")
x = st.slider("Select a value")
squared = x*x

st.write(f"The input number: {x}")
st.write(f"Squared value: {squared}")

st.title("Line Plot Example")
data = pd.DataFrame(
    {
        "x":[1,2,3,4],
        "y":[10,20,30,40]
    }
)
check_plot = st.checkbox("Show plot", value = True)
if check_plot == False:
    plt.close()

else: 
    #plot data
    fig, ax = plt.subplots()
    plt.plot(data["x"],data["y"], linestyle='-')
    st.pyplot(fig)


num = st.slider("Number of points:", min_value=100, max_value=1000, value=200, step=100)

st.write(num)

name=st.text_input("Enter your name:")
st.write(f'Your Name is: {name}')

color = st.selectbox("Plot color",["blue", "red", "green"])

fig, ax = plt.subplots()
ax.scatter(data["x"],data["y"], color=color)
st.pyplot(fig)


# st.camera_input("Take a photo")