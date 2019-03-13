#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# In[3]:


# Import dataset
dataset = pd.read_csv('50Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset


# In[134]:


# Handle missing data
imputer = SimpleImputer(missing_values=0, strategy='mean')
x[:, :3] = imputer.fit_transform(x[:, :3])


# In[4]:


# Encode categorical data 
ct = ColumnTransformer([ ('one-hot', OneHotEncoder(categories='auto'), [-1]) ])
encoded_cols = ct.fit_transform(x)
x = np.concatenate((x[:, :-1], encoded_cols[:, :-1]), axis=1)  # non-categ cols ++ categ cols with dummy trap removed


# In[139]:


# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

