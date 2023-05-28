import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

data=pd.read_csv('iris.csv')
print(data.head())

#splitting data in independent and dependent variable
X=data.iloc[:, :4].values
Y=data['class'].values

#splitting the dataset into training set and test  set
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=82)

#feature scalling to bring the variable in a single scale
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print("\n",X_train)

#fitting Naive Bayse Classification to the traning set with linear kernel
nvclassifier=GaussianNB()
nvclassifier.fit(X_train,y_train)

#predicting the Test set result
y_pred=nvclassifier.predict(X_test)
print("\n",y_pred)

#Let see the actual and predicted value side by side
y_compare=np.vstack((y_test,y_pred)).T

#printing the top 5 values
print("\n",y_compare[:5,:])

#making confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("\n",cm)
TP=cm[0,0]
TN=cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]
FP=cm[1,0]+cm[2,0]
FN=cm[0,1]+cm[0,2]

#printing the confusion matrix
print("Confusion Matrix\n\n",cm)
print("\n True Positives (TP)=" ,TP)
print("\n True Negatives (TN)=", TN)
print("\n False Positives (TP)=" ,FP)
print("\n False Positives (TP)=" ,FN)

print(classification_report(y_test,y_pred))

#finding accuracy
accuracy=(TP+TN)/float(TP+TN+FP+FN)
print("\n Accuracy = ",accuracy)

#finding error
error_rate=1-accuracy
print("\n Error Rate",error_rate)

#finding recall
recall=TP/float(TP+FN)
print("\n Recall = ",recall)

#finding precision score
precision = TP/float(TP+FP)
print("\n Precision = ",precision)

