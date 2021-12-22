# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 02:19:53 2021

@author: Temitayo

I, Temitayo Mawuena Hayes, hereby give my word of honour that I am 
the sole author of the work as submitted in the answers to this 
midterm examination. I didn’t collaborate with anyone and didn’t 
let anybody copy my work. If I used code written by someone else, 
I clearly and explicitly acknowledged the source using comments 
within the Python code, and I am aware that I will not receive 
any credits solely for code written by someone else. I didn’t 
post or submit, and will never post or submit, questions or answers, 
or parts of questions or answers of this midterm to a third party 
(except for my submission in Canvas for this midterm)
"""

import pandas as pd
data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',
        header=None)

data.replace({'R': 0, 'M': 1}, inplace=True)

# Create training/testing datasets
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=11)
y_train = train.iloc[:,-1]
X_train = train.iloc[:,0:-1]
y_test = test.iloc[:,-1]
X_test = test.iloc[:,0:-1]


import tensorflow as tf
from tensorflow import keras

#import necessary functions

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#define estimators
dtc_clf = DecisionTreeClassifier(max_depth=5)
gb_clf = GaussianNB()
svm_clf = SVC(gamma=0.5, C=100)


voting_clf = VotingClassifier(estimators=[('gb', gb_clf), ('svm', svm_clf), ('dtc', dtc_clf)], voting='hard')

voting_clf.fit(X_train, y_train)


#print accuracy
from sklearn.metrics import accuracy_score
for clf in (gb_clf, svm_clf, dtc_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_result = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_result))
    
    
    
    
