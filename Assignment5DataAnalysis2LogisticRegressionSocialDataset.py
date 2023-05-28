import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('5social_Network_Ads.csv')
print(df.shape)
print(df.head())

df.drop(['User ID'], axis=1, inplace=True)
print(df.head())

print(df.isnull().sum())

scaler = MinMaxScaler()
X = df[['Age', 'EstimatedSalary']]
X_scaled = scaler.fit_transform(X)
Y = df['Purchased']
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
print("Training and testing split was successful.")

basemodel = LogisticRegression()
basemodel.fit(X_train, y_train)
print("Training accuracy:", basemodel.score(X_train, y_train) * 100)

y_predict = basemodel.predict(X_test)
print("Testing accuracy:", basemodel.score(X_test, y_test) * 100)

Acc = accuracy_score(y_test, y_predict)
print(Acc)

cm = confusion_matrix(y_test, y_predict)
print(cm)

prf = precision_recall_fscore_support(y_test, y_predict)
print('precision:', prf[0])
print('Recall:', prf[1])
print('fscore:', prf[2])
print('support:', prf[3])
