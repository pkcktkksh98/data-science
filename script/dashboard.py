import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'],delimiter=',', index_col=False)
    return df

def main():
    st.title("Streamlit Dashboard")

    data =load_data()
    # st.dataframe(data)
    page = st.sidebar.selectbox("Select a page:",["Homepage",'Exploration', "Modelling"])

    if page == "Homepage":
        st.title("Homepage")
        st.dataframe(data)
    elif page == "Exploration":
        st.title("Exploratory Data")
        st.header("Correlation Coefficient")
        fig,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(abs(data.corr()),annot=True,ax=ax)
        st.pyplot(fig)
        st.text("Pairplot with different classes")
        fig = sns.pairplot(data,vars = ["magnesium", "flavanoids","nonflavanoid_phenols","proline"], hue = 'alcohol')
        st.pyplot(fig)

    else:
        st.title("Modelling")

if __name__=='__main__':
    main()