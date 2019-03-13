import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./Admission_Predict.csv')
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
c = data.iloc[:, 2].values

plt.scatter(x, y, c=c)