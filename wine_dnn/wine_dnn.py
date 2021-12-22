import pandas as pd
import numpy as np

cols = ["Class","Alcohol",
 	 "Malic acid",
 	 "Ash",
	 "Alcalinity of ash",  
 	 "Magnesium",
	"Total phenols",
 	"Flavanoids",
 	"Nonflavanoid phenols",
 	"Proanthocyanins",
	"Color intensity",
 	"Hue",
 	"OD280/OD315 of diluted wines",
 	"Proline" ]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
w = pd.read_csv(url, names=cols, sep=",", index_col = False)

X = w.drop('Class', axis=1)
y = w['Class']

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

from tensorflow import keras
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Dense(60, activation='relu', input_shape=[13]),
     keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])


history = model.fit(train_data, train_labels,epochs = 50, validation_data = (test_data, test_labels), verbose = 0)

result = np.round(model.predict_classes(test_data))

df = pd.read_excel (r'A2Q2.xlsx')
df.set_index('Attribute\Wine sample',inplace=True)
df_new= df.transpose()
