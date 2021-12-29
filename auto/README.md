<!-- GETTING STARTED -->
## Predicting Fuel Consumption

The program:
- Cleans the data and fixes the missing values by replacing them with the average horsepower for the respective number of cylinders. 
- Trains a poor model to intentionally overfit the training data, evident in its predictions 
- Trains a better model that does not overfit the training data.  
- Generates a graph showing both models (poor and good), with the highest negatively correlated attribute and the target variable (MPG)

![graph](auto.jpeg)

- Tests both models (poor and good) using the test set and reports the MSE for both models  


### Built With
* [Dataset](https://archive.ics.uci.edu/ml/datasets/Auto+MPG ) 
* [Python](https://reactjs.org/)
* [Numpy](https://github.com/facebook/create-react-app)
* [Pandas](https://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Sklearn](https://scikit-learn.org/)
