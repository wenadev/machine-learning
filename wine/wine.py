import pandas as pd

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
auto = pd.read_csv(url, names=cols, sep=",", index_col = False)


X = auto.drop('Class', axis=1)
y = auto['Class']
y_new = pd.get_dummies(auto, columns=['Class'])
labels = y_new.loc[:,['Class_1','Class_2','Class_3']]

#split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size=0.5, random_state=42)

#scale data
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler().fit(X_train)
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[13]),
    keras.layers.Dense(180, activation="relu"),
    keras.layers.Dense(3, activation='sigmoid')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

history = model.fit(
    X_train, y_train, 
    epochs=10, 
    validation_data=(X_test, y_test), 
    verbose=0)

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

import numpy as np

result = np.round(model.predict(X_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy", train_accuracy)
print("Test Accuracy: ", test_accuracy)

df = pd.read_excel (r'A2Q2.xlsx')
df.set_index('Attribute\Wine sample',inplace=True)
df_new= df.transpose()

#new samples
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_new)
ew = scaler.transform(df_new)
class_result = np.round(model.predict(ew))
sample = class_result.astype(int)

for i in range(9):
    print ("i",i, sample[i])
