import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.regularizers import l2


df = pd.read_csv("..\..\..\dataset\Iris.csv")

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df.Species

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

model = Sequential([
    Flatten(input_shape=(4,)),
    Dense(128,activation='relu', kernel_regularizer=l2(0.001)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(3,activation='softmax')
])
print(model.summary())

model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics =['accuracy'])

history=model.fit(x_train,y_train, batch_size=10, epochs= 5, validation_split=0.1)

plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.show()

score = model.evaluate(x_test,y_test, verbose=0)
print("Test loss", score[0])
print("Test Accuracy",score[1]) 