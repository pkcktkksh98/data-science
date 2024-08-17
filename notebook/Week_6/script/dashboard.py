import streamlit as st

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from module.load_data import load_data
from module.train_model import train_model
from module.plot_corr import plot_corr


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

        fig1,fig2=plot_corr(data)
        st.pyplot(fig1)
        st.text("Pairplot with different classes")
        st.pyplot(fig2)

    else:
        st.title("Modelling")
        #Declare models
        svm_clf = SVC()
        knn_clf = KNeighborsClassifier()
        rf_clf  = RandomForestClassifier()

        models = [svm_clf,knn_clf,rf_clf]
        
        for model in models:
            trained_model,score_rf = train_model(model,data)
            st.write(f"{model} Model Accuracy is {score_rf:.3f}")

if __name__=='__main__':
    main()