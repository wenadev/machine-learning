import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import statistics

data = pd.read_csv('gld_price_data.csv')

 # EDA
 #graphs to explore data and gain insights

 #get statistical analysis of data
data.head()
data.shape
data.describe()


 GRAPH 1
 #create a year column to visualize the data

 #Converting the date column to the proper format
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year

 # Here we can inspect how the gold price performed between 2008 to 2018
data.groupby('Year').max()['GLD'].plot(color='coral')
plt.title('Gold\'s Price over 10 years')
plt.show()

 #GRAPH 2
 #heat map to check correlations
data = data.drop(['Date'], axis =1)
corr = data.corr()

 #put correlation in report
print(corr)

 # the data is negatively and positively correlated
 #the red-ish squares represent positive correlation ex: SPX and Year
 #the blue-ish squares represent negative correlation ex: GLD and USO
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), cbar=True, square=True, annot=True,fmt='.2f', annot_kws={'size':7}, cmap='vlag')
plt.title('Correlation of Features', y = 1.05, size=20)


 #GRAPH 3
 #visualize distribution of the features
sns.distplot(data['GLD'], color='olive')
plt.figure(figsize = (20,60), facecolor='white')
plotnumber = 1

features = ['SPX', 'SLV', 'EUR/USD', 'USO']

for f in features:
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(data[f], color='olive');
    plotnumber += 1
    

 #plt.title('Gold Distribution')

 #correlation of gold with other features
print(corr['GLD'])


 #GRAPH 4
 #correlation of gold and stock index 
sns.jointplot(x=data['SPX'], y = data['GLD'], color = 'lightblue')

 #GRAPH 5
 #correlation of gold and oil price
sns.jointplot(x =data['USO'], y = data['GLD'], color = 'coral')


 # prepare data to expose patterns

 #checking number of missing values- there is none
data.isnull().sum()

 #assign features and target variable
X = data.drop(['GLD'], axis = 1)

y = data['GLD'].values


 #split data into train and test sets
X_tr, X_ts, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 3)
 #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

 #drop year
X_train = X_tr.drop(['Year'], axis =1)
X_test = X_ts.drop(['Year'], axis =1)

 #scale data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train.values)
X_train_sc = scaler.transform(X_train.values)
X_test_sc = scaler.transform(X_test.values)

 #creating a function to report error
def error_report(rmse_list, mae_list, model):
    
    min_rmse = min(rmse_list)
    max_rmse = max(rmse_list)
    mean_rmse = statistics.mean(rmse_list)
    
    min_mae = min(mae_list)
    max_mae = max(mae_list)
    mean_mae = statistics.mean(mae_list)
 
    print("%s model \nMax RMSE: %s || Min RMSE: %s || Mean RMSE: %s" %(model.capitalize(),max_rmse, min_rmse, mean_rmse))
    print("Max MAE: %s || Min MAE: %s || Mean MAE: %s" %(max_mae, min_mae, mean_mae))


 #MODEL 1- SGD

rmse_list = []
mae_list = []

for i in range(10):
    from sklearn.linear_model import SGDRegressor
    sgd = SGDRegressor()
        
     #fit Stochastic Gradient classifier to train set
    sgd.fit(X_train_sc, y_train)
    
     # predict test set result
    y_sgd = sgd.predict(X_test_sc)

     #calculating rmse and mae errors of the lasso model
    rmse = mean_squared_error(y_test,y_sgd, squared=False)
    mae = mean_absolute_error(y_test,y_sgd)
    
    s_rmse_list.append(rmse)
    s_mae_list.append(mae)

 #print maximum, minimum and mean values of RMSE and MAE for Random Forest regression
print(error_report(s_rmse_list, s_mae_list, "SGD"))

 #visualize accuracy of Lasso's predicted result
plt.figure(figsize=(10,8))
plt.plot(y_test, color = 'darkgreen', label = 'Actual')
plt.plot(y_sgd, color = 'cornflowerblue', label = 'Predicted values')
plt.grid(0.3)
 #plt.title('SGD Model')
plt.xlabel('Number of Values')
plt.ylabel('GLD')
plt.legend()
plt.show()

 #MODEL 2- Random Forest

f_rmse_list = []
f_mae_list = []

for i in range(10):

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100)
        
     #fit Random Forest classifier to train set
    regressor.fit(X_train_sc, y_train)
    
     # predict test set result
    y_forest = regressor.predict(X_test_sc)
    
     #calculating rmse and mae errors of the random forest model
    rmse = mean_squared_error(y_test,y_forest, squared=False)
    mae = mean_absolute_error(y_test,y_forest)
    
    f_rmse_list.append(rmse)
    f_mae_list.append(mae)

 #print maximum, minimum and mean values of RMSE and MAE for Random Forest regression
print(error_report(f_rmse_list, f_mae_list, "random forest"))


 #visualize accuracy of Random Forest's predicted result
plt.plot(y_test, color = 'darkgreen', label = 'Actual')
plt.plot(y_forest, color = 'cornflowerblue', label = 'Predicted values')
plt.grid(0.3)

plt.xlabel('Number of Values')
plt.ylabel('Gold Value')
plt.legend()
plt.show()



 #MODEL 3- DNN
nn_rmse_list = []
nn_mae_list = []

import tensorflow as tf
from tensorflow import keras

def model_builder(hp):
     model = keras.Sequential()
     model.add(keras.layers.Flatten(input_shape=[4]))
     
     hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
     model.add(keras.layers.Dense(units=hp_units, activation='relu'))
     model.add(keras.layers.Dense(units=hp_units, activation='relu'))
     model.add(keras.layers.Dense(1))
     
     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

     model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mae',
                metrics=['mse'])
     return model

import keras_tuner as kt
import IPython
tuner = kt.Hyperband(model_builder,
                     objective='val_mse',
                     max_epochs=10,
                     factor=3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

class ClearTrainingOutput(tf.keras.callbacks.Callback):

    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

tuner.search(X_train, y_train, 
             validation_data = (X_test, y_test),
             epochs=10, callbacks=[ClearTrainingOutput()], verbose =0)

 #optimal parameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

 #build model
nn = tuner.hypermodel.build(best_hps)

for i in range(10):
    history = nn.fit(X_train_sc, y_train, validation_data = (X_test_sc, y_test),  epochs = 10, verbose = 0)
    tr_mae, ts_mae = nn.evaluate(X_train_sc, y_train)
    y_nn = nn.predict(X_test_sc)

    nn_mae, nn_mse = nn.evaluate(X_test_sc, y_test)
    
     #calculating rmse and mae errors of the lasso model
    rmse = mean_squared_error(y_test,y_nn, squared=False)
    mae = mean_absolute_error(y_test,y_nn)
    
    nn_rmse_list.append(rmse)
    nn_mae_list.append(mae)

 #print maximum, minimum and mean values of RMSE and MAE for Neural Network regression
print(error_report(nn_rmse_list, nn_mae_list, "deep neural network"))

 #visualize accuracy of the DNN's predicted result
plt.plot(y_test, color = 'darkgreen', label = 'Actual')
plt.plot(y_nn, color = 'cornflowerblue', label = 'Predicted values')
plt.grid(0.3)
plt.title('Deep Neural Network Model')
plt.xlabel('Number of Values')
plt.xticks(rotation = '60');
plt.ylabel('Gold Value')
plt.legend()
plt.show()

 # Best performing algorithm is the Random Forest Model

 #plot gold prediction against all features

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(30, 25))
axes = axes.flatten()

for i, v in enumerate(X_ts.columns):
    
     # seclect the column to be plotted
    feature = X_ts[v]
    axes[i].scatter(x=feature, y= y_test, s=40, ec='white', label='actual')
    
     # plot the predicted gold values against the features
    axes[i].scatter(x=feature, y= y_forest, c='deeppink', s=20, ec='white', alpha=0.5, label='predicted')

     # set the title and ylabel
    axes[i].set(title=f'Feature: {v}', ylabel='Gold Price')
    
axes[4].legend(title='Price', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Best Performing Model predictions against Features')

for v in range(5, 6):
    fig.delaxes(axes[v])
    
