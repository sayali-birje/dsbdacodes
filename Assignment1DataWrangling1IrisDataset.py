import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data = pd.read_csv("C:\\Users\Sayali Birje\Downloads\iris.csv")
data = pd.read_csv("iris.csv")

print(data)
data.head()
a= data.sample(10)
print(a)

print("max value sepal length ",data['sepal_length'].max())

data.fillna(0,inplace= True)
print("no null value sepal width", data['sepal_length'])
print("***********")
print(data.describe())
specific_data=data[["sepal_length","sepal_width"]]
print ("specified data",specific_data)
print("statistic calculation")
sum_data=data["sepal_length"].sum()
mean_data=data["sepal_length"].mean()
median_data=data["sepal_length"].median()

print("Sum:",sum_data,"\nmean:",mean_data, "\nmedian :",median_data)


data.describe()
data.info()
plt.figure(figsize = (151, 5))
x = data["sepal_length"]

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("sepal_length")
plt.ylabel("Count")
plt.show()
