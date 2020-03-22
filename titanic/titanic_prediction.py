#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Explore feature engineering options: family size (Parch + Sibsp) 

"""
Created on Wed Mar 11 10:27:10 2020

@author: ducvm
"""

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
import csv

d = pd.read_csv("train.csv")
d1 = pd.read_csv("test.csv")
#train = pd.DataFrame(data = d)
#test = pd.DataFrame(data = d1)
whole = pd.concat([d, d1], sort = False).reset_index(drop=True)


"""
age_by_pclass_sex = whole.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(whole['Age'].median()))
"""


whole["Age"] = whole.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


whole["Embarked"] = whole["Embarked"].fillna("S")


median_fare = whole.groupby(["Pclass", "Parch", "SibSp"]).Fare.median()[3][0][0] 
whole['Fare'] = whole['Fare'].fillna(median_fare)

whole['Deck'] = whole['Cabin'].apply(lambda x: x[0].capitalize() if pd.notnull(x) else 'M')

idx = whole[whole['Deck'] == 'T'].index
whole.loc[idx, 'Deck'] = 'A'

dummy_gender = pd.get_dummies(whole["Sex"])
whole = pd.concat([whole, dummy_gender], axis = 1)
whole = whole.drop(["Sex"], axis = 1)

dummy_embarked = pd.get_dummies(whole["Embarked"])
whole = pd.concat([whole, dummy_embarked], axis = 1)
whole = whole.drop(["Embarked"], axis = 1)

whole.drop(['Cabin'], inplace=True, axis=1)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

train, test = divide_df(whole)

def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

dfs = [train, test]

for df in dfs:
    print('{}'.format(df))
    display_missing(df)

"""
#Heatmap feature selection
corrmat = df.corr()
#plot heat map
sns.set(font_scale = 1.8)
ax = plt.subplot(111)
plt.figure(figsize=(50,50))
heatmap = sns.heatmap(df.corr(),annot = True,cmap="RdYlGn", vmin = 0, vmax = 1)
plt.show()


orig_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "female", "male", "S", "Q", "C"]

x_train = train[orig_features].to_numpy()
y_train = train[["Survived"]].to_numpy()
#y_train = y_train.reshape(y_train.shape[0], )

x_test = test[orig_features]



regressor = RandomForestRegressor(n_estimators = 200)
regressor.fit(x_data, y_data)

sorted_indices = np.argsort(regressor.feature_importances_)[::-1]

for index in sorted_indices:
    print(f"{sorted_indices[index]}: {regressor.feature_importances_[index]}")
    
   Most important features: #Fare, Pclass, Parch, SibSp, Age



pd.crosstab(train.Pclass, train.Survived).plot(kind='bar')
plt.title("Surviving citizens and their class")
plt.xlabel('Passenger Class')
plt.ylabel('Surviving citizens')


linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
predictions = linear_regressor.predict(x_test)



model = Sequential()

model.add(Dense(150, input_dim = 7,activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs = 500, shuffle = False, verbose = 2)
predictions = model.predict(x_test)


decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(x_train,y_train)
predictions = decision_tree.predict(x_test)


#Append predictions to csv file with Id of passenger and whether they survived (0 = no, 1 = yes)
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
"""       
