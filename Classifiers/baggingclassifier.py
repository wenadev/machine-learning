# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 02:22:22 2021

@author: Temitayo

I, Temitayo Mawuena Hayes, hereby give my word of honour that I 
am the sole author of the work as submitted in the answers to this 
midterm examination. I didn’t collaborate with anyone and didn’t 
let anybody copy my work. If I used code written by someone else, 
I clearly and explicitly acknowledged the source using comments 
within the Python code, and I am aware that I will not receive any 
credits solely for code written by someone else. I didn’t post or 
submit, and will never post or submit, questions or answers, or 
parts of questions or answers of this midterm to a third party 
(except for my submission in Canvas for this midterm).

"""
import pandas as pd
data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',
        header=None)

# Create training/testing datasets
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=11)
y_train = train.iloc[:,-1]
X_train = train.iloc[:,0:-1]
y_test = test.iloc[:,-1]
X_test = test.iloc[:,0:-1]


#split data into train and test
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

est = DecisionTreeClassifier()

#use 100 decision tree estimators
clf = BaggingClassifier(base_estimator=est, 
                        n_estimators=100, bootstrap=True, oob_score=True)

#fit classifier and predict data
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score

#print accuracy on test data
print('Bagging score', accuracy_score(y_test, y_pred))

#print accuracy on out of bag samples
print('Out-of-sample score', clf.oob_score_)

