#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:27:10 2020

@author: ducvm
"""

import numpy as np
import keras as ks
import pandas as pd
import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression
import csv

d = pd.read_csv("train.csv")
d1 = pd.read_csv("test.csv")
train = pd.DataFrame(data = d)
test = pd.DataFrame(data = d1)



dummy_gender = pd.get_dummies(train["Sex"])
train = pd.concat([train, dummy_gender], axis = 1)
train = train.drop(["Sex"], axis = 1)

dummy_embarked = pd.get_dummies(train["Embarked"])
train = pd.concat([train, dummy_embarked], axis = 1)
train = train.drop(["Embarked"], axis = 1)


dummy_gender = pd.get_dummies(test["Sex"])
test = pd.concat([test, dummy_gender], axis = 1)
test = test.drop(["Sex"], axis = 1)

dummy_embarked = pd.get_dummies(test["Embarked"])
test = pd.concat([test, dummy_embarked], axis = 1)
test = test.drop(["Embarked"], axis = 1)


train = train.drop(["Cabin", "Name", "Ticket"], axis = 1)
train = train.fillna(value = 29.699118)


test = test.drop(["Cabin", "Name", "Ticket"], axis = 1)
test = test.fillna(value =  30.272590)



"""
#Heatmap feature selection
corrmat = df.corr()
#plot heat map
sns.set(font_scale = 1.8)
ax = plt.subplot(111)
plt.figure(figsize=(50,50))
heatmap = sns.heatmap(df.corr(),annot = True,cmap="RdYlGn", vmin = 0, vmax = 1)
plt.show()
"""
orig_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

x_train = train[orig_features]
y_train = train[["Survived"]]
#y_train = y_train.reshape(y_train.shape[0], )

x_test = test[orig_features]


"""
regressor = RandomForestRegressor(n_estimators = 200)
regressor.fit(x_data, y_data)

sorted_indices = np.argsort(regressor.feature_importances_)[::-1]

for index in sorted_indices:
    print(f"{sorted_indices[index]}: {regressor.feature_importances_[index]}")
    
   Most important features: #Fare, Pclass, Parch, SibSp, Age
"""

linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
predictions = linear_regressor.predict(x_test)

survived = []
for i in predictions:
    if i > 0.5:
        i = 1
        survived.append(i)
    else:
        i = 0
        survived.append(i)

survived.insert(0, "Survived")

PassengerId = [test.PassengerId[i] for i in range(len(test.PassengerId))]
PassengerId.insert(0, "PassengerId")

rows = zip(PassengerId, survived)

with open('submission.csv', 'w') as f:
    wr = csv.writer(f)
    for row in rows:
        wr.writerow(row)
        
