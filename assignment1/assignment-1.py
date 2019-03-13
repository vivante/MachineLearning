#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('50Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy='mean', axis=0)
imputer = imputer.fit(x[:, 0:3])
x[:, 0:3] = imputer.transform(x[:, 0:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
x[:, -1] = labelencoder.fit_transform(x[:, -1])  # transform only specified colums
onehotencoder = OneHotEncoder(categorical_features=[-1])
x = onehotencoder.fit_transform(x).toarray()  # columns are specified in constructor hence transform entire thing


# split data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Linear regression
from sklearn import linear_model

reg1 = linear_model.LinearRegression()
reg1.fit(x_train, y_train)

reg2 = linear_model.Ridge(alpha=.005)
reg2.fit(x_train, y_train)

reg3 = linear_model.Lasso(alpha=.005)
reg3.fit(x_train, y_train)

reg4 = linear_model.Ridge(alpha=.01)
reg4.fit(x_train, y_train)

reg5 = linear_model.Lasso(alpha=.01)
reg5.fit(x_train, y_train)

# testing for accuracy
from sklearn.metrics import explained_variance_score

y_pred1 = reg1.predict(x_test)
y_pred2 = reg2.predict(x_test)
y_pred3 = reg3.predict(x_test)
y_pred4 = reg4.predict(x_test)
y_pred5 = reg5.predict(x_test)

print(explained_variance_score(y_test, y_pred1))
print(explained_variance_score(y_test, y_pred2))
print(explained_variance_score(y_test, y_pred3))
print(explained_variance_score(y_test, y_pred4))
print(explained_variance_score(y_test, y_pred5))

plt.plot(np.arange(0, 17), y_test, marker='o', linestyle='dashed', color='blue')
plt.plot(np.arange(0, 17), y_pred1, marker='o', linestyle='dashed', color='green')
plt.plot(np.arange(0, 17), y_pred2, marker='o', linestyle='dashed', color='red')
plt.plot(np.arange(0, 17), y_pred3, marker='o', linestyle='dashed', color='black')
plt.plot(np.arange(0, 17), y_pred4, marker='o', linestyle='dashed', color='orange')
plt.plot(np.arange(0, 17), y_pred5, marker='o', linestyle='dashed', color='purple')
