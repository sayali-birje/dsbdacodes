import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('iris.csv')

print(df.head())
print(df.tail())
print(df.info())
print(df.shape)
print(df.dtypes)
print(df.describe())

# df.hist()
# plt.show()

# df.boxplot()
# plt.show()

df.isnull().sum()

sns.histplot(x = df['sepal_length'], kde=True)
plt.show()

sns.histplot(x = df['sepal_width'], kde=True)
plt.show()

sns.histplot(x = df['petal_length'], kde=True)
plt.show()

sns.histplot(x = df['petal_width'], kde=True)
plt.show()

sns.boxplot(df['sepal_length'])
plt.show()

sns.boxplot(df['petal_length'])
plt.show()

sns.boxplot(df['petal_width'])
plt.show()

sns.boxplot(x='sepal_length',y='class',data=df)
plt.show()

sns.boxplot(x='petal_length',y='class',data=df)
plt.show()
