import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv("C:\\Users\Sayali Birje\Downloads\iris.csv")
print(df.head())

#'Iris-setosa'
setosa = df['class'] == 'Iris-setosa'
print(df[setosa].describe())
#'Iris-versicolor'
versicolor = df['class'] == 'Iris-versicolor'
print(df[versicolor].describe())
#'Iris-virginica'
virginica = df['class'] == 'Iris-virginica'
print(df[virginica].describe())
print(df.dtypes)
print(df.dtypes.value_counts())
