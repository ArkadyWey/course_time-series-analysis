import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.stats import boxcox

df = pd.read_csv("airline_passengers.csv", index_col="Month", parse_dates=True)

print(df.head())

df["Passengers"].plot(figsize=(20,8))
plt.show()

df["SqrtPassengers"] = np.sqrt(df["Passengers"]) 
df["SqrtPassengers"].plot()
plt.show()

df["LogPassengers"] = np.log(df["Passengers"])
df["LogPassengers"].plot()
plt.show()

data, lam = boxcox(df["Passengers"])

print(lam)

df["BoxCoxPassengers"] = data 
df["BoxCoxPassengers"].plot()
plt.show()

df["Passengers"].hist(bins=20)
plt.show()

