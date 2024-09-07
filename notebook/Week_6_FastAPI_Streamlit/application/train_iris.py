import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("..\..\..\dataset\Iris.csv")

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df.Species

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

model=SVC(probability=True)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)

joblib.dump(model, "model/iris_mdl_prob.pkl")