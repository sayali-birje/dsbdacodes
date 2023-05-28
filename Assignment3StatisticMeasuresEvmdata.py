import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("3emp.csv")
data2=np.array([7,5,4,9,12,45])
print("Sum: ",data["EmployeeNumber"].sum())
print("Mean: ",data["EmployeeNumber"].mean())
print()
print("Maximum: ",data["EmployeeNumber"].max())
print("Minimum: ",data["EmployeeNumber"].min())
print("Standard deviation: ", statistics.stdev(data["EmployeeNumber"]))

print("Standard Deviation of the sample is % s " % (statistics.stdev(data2)))
print("Mean of the sample is % s " % (statistics.mean(data2)))
