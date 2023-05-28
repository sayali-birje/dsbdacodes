import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset =pd.read_csv('8,9titanic.csv')
print(dataset)
print(dataset.head())
sns.histplot(dataset['Fare'], kde=True, linewidth=0);
sns.jointplot(x='Age', y='Fare', data=dataset);
plt.show()
