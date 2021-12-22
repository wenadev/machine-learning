# -*- coding: utf-8 -*-
"""
MSE for poor model (KNN): 20.2565 

MSE for good model (Lasso): 3.8466458475472365
    
    
Created on Mon Sep 27 06:13:55 2021

@author: Temitayo Hayes
Student Number: 100794977
Course: Machine Learning MITS
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

cols = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin', 'Name']

DOWN = "https://archive.ics.uci.edu/ml/"

NEW_path = os.path.join("machine-learning-databases", "auto-mpg")
NAMES_path = os.path.join("machine-learning-databases", "auto-mpg")

NEW_url = DOWN + "machine-learning-databases/auto-mpg/auto-mpg.data"
NAMES_url = DOWN + "machine-learning-databases/auto-mpg/auto-mpg.names"


def fetch_auto_data(auto_url=NEW_url, auto_path=NEW_path,  names_url=NAMES_url,  names_path = NAMES_path ):
    if (not os.path.isdir(auto_path)) and (not os.path.isdir(names_path)):
        os.makedirs(auto_path)
        
    auto_data_path = os.path.join(auto_path, "auto-mpg.data")
    urllib.request.urlretrieve(auto_url, auto_data_path)
    
fetch_auto_data()


def load_auto_data(auto_path=NEW_path, names_path= NAMES_path):
    auto_data = os.path.join(auto_path, "auto-mpg.data")
    
    df = pd.read_csv(auto_data, names=cols, na_values = "?", comment = '\t', skipinitialspace=True, delim_whitespace=True)
    df_data_auto = df.copy()
    
    return df_data_auto

auto = load_auto_data()

#Exploratory Data Analysis
auto.shape
auto.head()
auto.describe()


#drop car name attribute
auto = auto.drop(['Name'], axis=1)


#clean dataset and fix missing values
auto.isnull().sum()
condition= auto[auto["Horsepower"].isnull()][["Cylinders", "Horsepower"]] # selects cylinders where horsepower is null
cylinder_number = condition.Cylinders.unique() #selects single values cylinder numbers

ch= auto.groupby("Cylinders")["Horsepower"].mean().reset_index(name='Horsepower') #average of horsepower by respective cylinders
ch_new = ch[ch["Cylinders"].isin(cylinder_number)].round(2) #only selects average of cylinder_number indicated to be nan above
avg_dict =dict(zip(ch_new['Cylinders'], ch_new['Horsepower'])) #assigning avg of cylinders of respective cylinders in a dict

bool_mask = auto['Horsepower'].isna()
auto.loc[bool_mask, 'Horsepower'] = auto.loc[bool_mask, 'Cylinders'].map(avg_dict) #fill in average of respective rows according to cylinder number


#check correlation with target
#sns.pairplot(auto, x_vars=auto.drop(['MPG'], axis=1, inplace=False).columns, y_vars= ['MPG'])
corr_matrix = auto.corr()
corr_matrix['MPG'].sort_values(ascending=True)


X= auto.iloc[: , 4:5]
y = auto.iloc[:, 0:1]


#split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#scale data
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
scaler2.fit(X_train)
X_train_sc = scaler2.transform(X_train)
X_test_sc = scaler2.transform(X_test)


#training poor model that overfits with 
#training accuracy higher than validation accuracy
from sklearn.neighbors import KNeighborsRegressor
knn= knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)

#KNN Training accuracy
knn.score(X_train, y_train)

y_knn = knn.predict(X_test)

#KNN Test accuracy with test set
knn.score(X_test, y_test)

k_mse = mean_squared_error(y_test, y_knn)
print("MSE for KNN: ",k_mse, "\n")


#training better model with good generalization of data
from sklearn.linear_model import Lasso
lasso2 = Lasso(alpha=0.1, random_state=42)
lasso2.fit(X_train_sc, y_train)

#Lasso Training accuracy
lasso2.score(X_train_sc, y_train)

y_lasso = lasso2.predict(X_test_sc)

#Lasso Test accuracy with test set
lasso2.score(X_test_sc, y_test)

l_mse = np.sqrt(mean_squared_error(y_test, y_lasso))
print("MSE for Lasso: ",l_mse)


import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#to make the overfit model's prediction line smoother
param = np.linspace(0, 1, X_test.size)
spl = make_interp_spline(param, np.c_[X_test,y_knn], k=2)
xnew, y_smooth = spl(np.linspace(0, 1, X_test.size * 100)).T


plt.scatter(X, y, color = 'skyblue', label="Actual data values")
plt.plot(xnew, y_smooth, '--', color="green", label="KNN predictions", lw =1)
plt.plot(X_test, y_lasso, color="darkorange", label="Lasso predictions", lw =2)
plt.title("City-cycle Fuel Consumption")

plt.xlabel("Car Weight") 
plt.ylabel("Miles per Gallon")
plt.xticks([2000,3000,4000,5000])
plt.yticks([10,20,30,40,50])
plt.legend()
plt.show()