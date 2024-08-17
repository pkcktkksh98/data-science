from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def train_model(model, data):
    #features seperator and label seperator

    x = data.drop("alcohol", axis=1)
    y = data.alcohol

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
    
    #model training
    model.fit(x_train,y_train)
    score=model.score(x_test,y_test)

    return model,score