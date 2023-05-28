import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv("4HousingData.csv")
print(data.head())

print(data.tail())
print("The shape of data : ", data.shape)

data.isnull().sum()
data.fillna(0, inplace=True)
data.isnull().sum()

X = data.iloc[:,0:13]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
model.fit(X_train, y_train)

print(model.score(X_test, y_test)*100)